"""Core domain-adversarial (DA) components used in SS-SENet.

This file consolidates the DA-related code from ``Discriminator.py`` and
``main_MT+DA(1).py``. Evaluation, logging, checkpointing, MT losses, GGCA,
and DSAM are intentionally omitted.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils import spectral_norm

import models.improved_sudormrf as improved_sudormrf


class DA(nn.Module):
    """Domain classifier operating on bottleneck features."""

    def __init__(self, in_channels=256, out_channels=2, bottleneck_dim=128):
        super(DA, self).__init__()

        self.features = nn.Sequential(
            spectral_norm(
                nn.Conv1d(
                    in_channels,
                    bottleneck_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            ),
            nn.GroupNorm(8, bottleneck_dim),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv1d(
                    bottleneck_dim,
                    in_channels,
                    kernel_size=1,
                    stride=1,
                )
            ),
            nn.GroupNorm(8, in_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(in_channels, 128)),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(128, out_channels)),
        )

    def forward(self, x):
        x = self.features(x)
        x_gap = torch.mean(x, dim=-1)
        out = self.fc(x_gap)
        return out, x_gap


class GradientReversal(Function):
    """Identity in the forward pass and gradient reversal in backpropagation."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class SuDORMRF_DA(improved_sudormrf.SuDORMRF):
    """SuDoRM-RF with a training-only domain-adversarial branch.

    Only the student contains the domain classifier. The teacher and the
    inference path retain the original enhancement architecture.
    """

    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        enc_num_basis=512,
        num_sources=2,
        model="student",
    ):
        super().__init__(
            out_channels=out_channels,
            in_channels=in_channels,
            num_blocks=num_blocks,
            upsampling_depth=upsampling_depth,
            enc_kernel_size=enc_kernel_size,
            enc_num_basis=enc_num_basis,
            num_sources=num_sources,
        )

        self.model_type = model

        if model == "student":
            self.classifier = DA(
                in_channels=out_channels,
                out_channels=2,
            )
        elif model != "teacher":
            raise ValueError(
                f"model must be 'student' or 'teacher', got {model}"
            )

    def forward(
        self,
        input_wav,
        alpha=0.0,
        mode="eval",
        domain_type="label",
    ):
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        encoded_features = x.clone()

        x = self.ln(x)
        x = self.bottleneck(x)

        domain_loss = None
        domain_features = None

        if mode == "train" and hasattr(self, "classifier"):
            reversed_features = GradientReversal.apply(x, alpha)
            domain_logits, domain_features = self.classifier(reversed_features)

            batch_size = x.shape[0]

            if domain_type == "label":
                domain_targets = torch.zeros(
                    batch_size,
                    dtype=torch.long,
                    device=x.device,
                )
            elif domain_type == "unlabel":
                domain_targets = torch.ones(
                    batch_size,
                    dtype=torch.long,
                    device=x.device,
                )
            else:
                raise ValueError(f"Unsupported domain_type: {domain_type}")

            domain_loss = F.cross_entropy(domain_logits, domain_targets)

        x = self.sm(x)
        x = self.mask_net(x)

        x = x.view(
            x.shape[0],
            self.num_sources,
            self.enc_num_basis,
            -1,
        )

        x = self.mask_nl_class(x)
        x = x * encoded_features.unsqueeze(1)

        estimated_waveforms = self.decoder(
            x.view(x.shape[0], -1, x.shape[-1])
        )
        estimated_waveforms = self.remove_trailing_zeros(
            estimated_waveforms,
            input_wav,
        )

        return estimated_waveforms, domain_loss, domain_features


def get_new_student(hparams, depth_growth, model):
    """Build the student or teacher using the original model parameters."""

    student = SuDORMRF_DA(
        out_channels=hparams["out_channels"],
        in_channels=hparams["in_channels"],
        num_blocks=int(depth_growth * hparams["num_blocks"]),
        upsampling_depth=hparams["upsampling_depth"],
        enc_kernel_size=hparams["enc_kernel_size"],
        enc_num_basis=hparams["enc_num_basis"],
        num_sources=2,
        model=model,
    )
    return student


def get_alpha_DANN(p):
    """Compute the GRL coefficient from normalized training progress."""

    return 2.0 / (1.0 + np.exp(-10 * p)) - 1.0


def domain_adversarial_forward(
    student,
    label_input_mix,
    bootstrapped_mix,
    epoch,
    batch_index,
    len_dataloader,
):
    """Run the two student paths and compute the averaged domain loss."""

    if epoch < 100:
        p = float(batch_index + epoch * len_dataloader) / float(
            100 * len_dataloader
        )
        alpha = get_alpha_DANN(p)
    else:
        alpha = 1.0

    student_estimates_label, domain_loss_label, _ = student(
        input_wav=label_input_mix,
        alpha=alpha,
        mode="train",
        domain_type="label",
    )

    student_estimates_unlabel, domain_loss_unlabel, _ = student(
        input_wav=bootstrapped_mix,
        alpha=alpha,
        mode="train",
        domain_type="unlabel",
    )

    domain_loss = torch.mean(
        0.5 * (domain_loss_label + domain_loss_unlabel)
    )

    return (
        student_estimates_label,
        student_estimates_unlabel,
        domain_loss,
        alpha,
    )


def add_domain_loss(mt_loss, domain_loss):
    """Add the DA term to the existing Mean Teacher objective."""

    return mt_loss + 0.05 * domain_loss
