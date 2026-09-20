Speech enhancement (SE) models trained on synthetic noisy
clean pairs often generalize poorly to real recordings without
clean references. We propose SS-SENet, a semi-supervised
monaural SE framework that jointly exploits synthetic la
beled and real-world unlabeled data. SS-SENet employs
the Mean Teacher (MT) framework, in which the student
model learns from corresponding clean reference signals and
teacher-generated pseudo-targets, while the teacher model
is updated by an exponential moving average of the stu
dent model parameters. We introduce the domain-adversarial
training strategy with a gradient reversal layer to reduce
the feature discrepancies between labeled and remixed un
labeled mixtures. Experiments on CHiME-5, Reverberant
LibriCHiME-5, and LibriMix demonstrate that SS-SENet can
effectively leverage unlabeled recordings from real-world en
vironments to improve SE performance.
