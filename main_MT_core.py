"""Core Mean Teacher training code for SS-SENet.

This version preserves the original training logic and removes only
evaluation, logging, checkpoint management, and unused experimental code.
"""

import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import models.improved_sudormrf as improved_sudormrf
import semi_supervised.semi_utils.semi_cmd_parser as parser
import semi_supervised.semi_utils.semi_dataset_setup as dataset_setup
import utils.mixture_consistency as mixture_consistency
from utils.losses import pairwise_neg_sisdr
from utils.utils import update_ema


args = parser.get_args()
hparams = vars(args)
generators = dataset_setup.semisupervised_setup(hparams)

os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
    [cad for cad in hparams["cuda_available_devices"]]
)


def get_new_student(hparams, depth_growth, model):
    student = improved_sudormrf.SuDORMRF(
        out_channels=hparams["out_channels"],
        in_channels=hparams["in_channels"],
        num_blocks=int(depth_growth * hparams["num_blocks"]),
        upsampling_depth=hparams["upsampling_depth"],
        enc_kernel_size=hparams["enc_kernel_size"],
        enc_num_basis=hparams["enc_num_basis"],
        num_sources=2,
    )
    return student


def freeze_model(model):
    for f in model.parameters():
        if f.requires_grad:
            f.requires_grad = False


def apply_output_transform(
    rec_sources_wavs, input_mix_std, input_mix_mean, input_mom, hparams
):
    if hparams["rescale_to_input_mixture"]:
        rec_sources_wavs = (rec_sources_wavs * input_mix_std) + input_mix_mean
    if hparams["apply_mixture_consistency"]:
        rec_sources_wavs = mixture_consistency.apply(rec_sources_wavs, input_mom)
    return rec_sources_wavs


class ExponentialWarmup(object):
    def __init__(
        self,
        optimizer,
        max_lr,
        rampup_length=40,
        exponent=-5.0,
        patience=15,
        divide_lr_by=3,
        logger=None,
    ):
        self.optimizer = optimizer
        self.rampup_length = rampup_length
        self.max_lr = max_lr
        self.step_num = 1
        self.exponent = exponent
        self.patience = patience
        self.divide_lr_by = divide_lr_by
        self.logger = logger
        self._warmup_ended_logged = False
        self._last_decay_step = -1

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _get_lr(self):
        if self.step_num <= self.rampup_length:
            return self.max_lr * self._get_scaling_factor()
        decay_steps = (self.step_num - self.rampup_length) // self.patience
        decay_factor = (1 / self.divide_lr_by) ** decay_steps
        return self.max_lr * decay_factor

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def step(self):
        prev_step = self.step_num
        self.step_num += 1
        lr = self._get_lr()
        self._set_lr(lr)

        if self.logger and not self._warmup_ended_logged:
            if prev_step == self.rampup_length:
                self.logger.info(
                    f"[Warmup Completed], Reached max LR: {self.max_lr:.2e}"
                )
                self._warmup_ended_logged = True

        if self.logger and self.step_num > self.rampup_length:
            current_decay_step = self.step_num - self.rampup_length - 1
            decay_interval = self.patience
            if decay_interval > 0 and current_decay_step % decay_interval == 0:
                if current_decay_step != self._last_decay_step:
                    self.logger.info(
                        f"[LR Decay], New LR: {lr:.2e} "
                        f"(decayed {current_decay_step // decay_interval} times)"
                    )
                    self._last_decay_step = current_decay_step

    def _get_scaling_factor(self):
        if self.rampup_length == 0:
            return 1.0
        current = np.clip(self.step_num, 0.0, self.rampup_length)
        phase = 1.0 - current / self.rampup_length
        return float(np.exp(self.exponent * phase * phase))


cons_func = nn.MSELoss()

student = get_new_student(hparams, depth_growth=1, model="student")
teacher = get_new_student(hparams, depth_growth=1, model="teacher")

for param in student.parameters():
    assert param.requires_grad

opt = torch.optim.Adam(student.parameters(), lr=hparams["learning_rate"])

warmup_steps = hparams["n_epochs_warmup"] * len(generators["train"][0])
patience = hparams["patience"] * len(generators["train"][0])
scheduler = ExponentialWarmup(
    opt,
    max_lr=hparams["learning_rate"],
    rampup_length=warmup_steps,
    exponent=-5.0,
    patience=patience,
    divide_lr_by=hparams["divide_lr_by"],
)

student = torch.nn.DataParallel(student).cuda()
teacher = torch.nn.DataParallel(teacher).cuda()
freeze_model(teacher)

initial_seed = 17

for i in range(hparams["n_epochs"]):
    torch.manual_seed(initial_seed + i)
    np.random.seed(initial_seed + i)

    label_train_data = generators["train"][0]
    unlabel_train_data = generators["train"][1]
    label_train_tqdm = tqdm(label_train_data, desc="Label Training")
    unlabel_train_tqdm = tqdm(unlabel_train_data, desc="Unlabel Training")

    student.train()
    teacher.eval()

    for label, unlabel in zip(label_train_tqdm, unlabel_train_tqdm):
        opt.zero_grad()

        label_speakers, label_noise = label
        unlabel_mix = unlabel
        unlabel_input_mix = unlabel_mix.unsqueeze(1).cuda()

        # Normalize the labeled input mixture.
        label_speaker_input_mix = label_speakers.sum(1, keepdims=True).cuda()
        label_noise = label_noise.cuda()
        label_input_mix = label_noise + label_speaker_input_mix
        label_input_std = label_input_mix.std(-1, keepdims=True)
        label_input_mean = label_input_mix.mean(-1, keepdims=True)
        label_input_mix = (label_input_mix - label_input_mean) / (
            label_input_std + 1e-9
        )

        # Normalize the unlabeled input mixture.
        unlabel_mix_std = unlabel_input_mix.std(-1, keepdims=True)
        unlabel_mix_mean = unlabel_input_mix.mean(-1, keepdims=True)
        unlabel_input_mix = (unlabel_input_mix - unlabel_mix_mean) / (
            unlabel_mix_std + 1e-9
        )

        # Generate detached teacher pseudo-targets and remixed mixtures.
        with torch.no_grad():
            teacher_estimates_unlabel = teacher(input_wav=unlabel_input_mix)
            teacher_estimates_unlabel = apply_output_transform(
                teacher_estimates_unlabel,
                unlabel_mix_std,
                unlabel_mix_mean,
                unlabel_input_mix,
                hparams,
            )
            t_est_speech_unlabel = teacher_estimates_unlabel[:, 0:1].detach()
            t_est_noise_unlabel = teacher_estimates_unlabel[:, 1:].detach()

            batch_size, _, _ = t_est_noise_unlabel.shape
            permuted_t_est_noise = t_est_noise_unlabel[
                torch.randperm(batch_size)
            ]
            bootstrapped_mix = t_est_speech_unlabel + permuted_t_est_noise
            bootstrapped_mix_std = bootstrapped_mix.std(-1, keepdim=True)
            bootstrapped_mix_mean = bootstrapped_mix.mean(-1, keepdim=True)
            bootstrapped_mix = (bootstrapped_mix - bootstrapped_mix_mean) / (
                bootstrapped_mix_std + 1e-9
            )

        # Obtain student estimates for labeled and remixed unlabeled data.
        student_estimates_label = student(input_wav=label_input_mix)
        student_estimates_unlabel = student(input_wav=bootstrapped_mix)

        student_estimates_label = apply_output_transform(
            student_estimates_label,
            label_input_std,
            label_input_mean,
            label_input_mix,
            hparams,
        )
        s_est_speech_label = student_estimates_label[:, 0:1]
        s_est_noise_label = student_estimates_label[:, 1:]

        student_estimates_unlabel = apply_output_transform(
            student_estimates_unlabel,
            bootstrapped_mix_std,
            bootstrapped_mix_mean,
            bootstrapped_mix,
            hparams,
        )
        s_est_speech_unlabel = student_estimates_unlabel[:, 0:1]
        s_est_noise_unlabel = student_estimates_unlabel[:, 1:]

        # Consistency loss for unlabeled data.
        w_cons = 0.3 * scheduler._get_scaling_factor()
        unlabel_speaker_mse = cons_func(
            s_est_speech_unlabel, t_est_speech_unlabel.detach()
        )
        unlabel_noise_mse = cons_func(
            s_est_noise_unlabel, permuted_t_est_noise.detach()
        )
        unlabel_mse_loss = unlabel_speaker_mse + unlabel_noise_mse

        unlabel_speaker_sisdr = torch.mean(
            torch.clamp(
                pairwise_neg_sisdr(
                    t_est_speech_unlabel.detach(), s_est_speech_unlabel
                ),
                min=-30.0,
                max=30.0,
            )
        )
        unlabel_noise_sisdr = torch.mean(
            torch.clamp(
                pairwise_neg_sisdr(
                    permuted_t_est_noise.detach(), s_est_noise_unlabel
                ),
                min=-30.0,
                max=30.0,
            )
        )
        unlabel_sisdr_loss = (
            0.5 * unlabel_speaker_sisdr + 0.5 * unlabel_noise_sisdr
        )
        l1 = unlabel_mse_loss + unlabel_sisdr_loss

        # Supervised loss between student estimates and clean references.
        speaker_mse = cons_func(s_est_speech_label, label_speaker_input_mix)
        noise_mse = cons_func(s_est_noise_label, label_noise)
        mse_loss = speaker_mse + noise_mse

        speaker_sisdr = torch.mean(
            torch.clamp(
                pairwise_neg_sisdr(
                    label_speaker_input_mix, s_est_speech_label
                ),
                min=-30.0,
                max=30.0,
            )
        )
        noise_sisdr = torch.mean(
            torch.clamp(
                pairwise_neg_sisdr(label_noise, s_est_noise_label),
                min=-30.0,
                max=30.0,
            )
        )
        sisdr_loss = 0.5 * speaker_sisdr + 0.5 * noise_sisdr
        l2 = mse_loss + sisdr_loss

        l = w_cons * l1 + l2
        l.backward()

        if hparams["clip_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                student.parameters(), hparams["clip_grad_norm"]
            )

        opt.step()
        scheduler.step()

        teacher = update_ema(
            student,
            teacher,
            scheduler.step_num,
            hparams["teacher_momentum"],
        )
