"""AudioSet evaluation preprocessing and ordered label loading."""
import csv
from importlib.resources import files
import torch
import torchaudio
from torch.nn import functional as F

SAMPLE_RATE = 16000
NUM_MEL_BINS = 128
TARGET_FRAMES = 1024
MEAN = -4.2677393
STD = 4.5689974


def waveform_to_fbank(waveform, sample_rate):
    """Average channels, remove DC before resampling, then apply the training recipe."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or min(waveform.shape) < 1:
        raise ValueError("Expected nonempty waveform [channels, samples].")
    if sample_rate <= 0 or not torch.isfinite(waveform).all():
        raise ValueError("Audio must have a positive sample rate and finite samples.")
    waveform = waveform.detach().to(device="cpu", dtype=torch.float32).mean(dim=0, keepdim=True)
    waveform = waveform - waveform.mean()
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)
    if waveform.shape[-1] < 400:
        waveform = F.pad(waveform, (0, 400 - waveform.shape[-1]))
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=SAMPLE_RATE, use_energy=False,
        window_type="hanning", num_mel_bins=NUM_MEL_BINS, dither=0.0,
        frame_length=25.0, frame_shift=10.0,
    )
    fbank = fbank[:TARGET_FRAMES]
    fbank = F.pad(fbank, (0, 0, 0, TARGET_FRAMES - fbank.shape[0]))
    return (fbank - MEAN) / (STD * 2)


def load_audio(path):
    waveform, sample_rate = torchaudio.load(str(path))
    return waveform_to_fbank(waveform, sample_rate)


def load_labels():
    resource = files("ad_mamba").joinpath("class_labels_indices.csv")
    with resource.open("r", encoding="utf-8", newline="") as stream:
        labels = list(csv.DictReader(stream))
    if len(labels) != 527 or [int(row["index"]) for row in labels] != list(range(527)):
        raise ValueError("AudioSet label indices must be exactly 0..526.")
    return labels
