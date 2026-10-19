# AD-Mamba

Pretrained AD-Mamba models for **527 AudioSet sound events**. Run inference on your own audio with an NVIDIA GPU.

## Pretrained models

Small and Medium differ in model size. Both were trained on GPUs and use the same installation steps.

| Model | Approx. parameters | AudioSet mAP | Checkpoint |
|---|---:|---:|---|
| Small (default) | 30M | 50.3% | `ad_mamba_small.pt` |
| Medium | 49M | 50.6% | `ad_mamba_medium.pt` |

Place the checkpoint files in `checkpoints/`. Weights will be distributed as GitHub Release assets, separately from the Python package. The mAP values above are historical single-model evaluation results; full AudioSet evaluation has not been rerun for this release.

## Installation

Use **Python 3.10+** and an NVIDIA GPU. Install CUDA-enabled PyTorch and matching TorchAudio using the [official instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

If you already have compatible PyTorch and TorchAudio installed, keep your existing environment. The examples below use the included PyTorch implementation and do not require compiling extra CUDA extensions.

## Inference

Replace `example.wav` with the path to your own audio file:

```bash
python inference.py --audio example.wav --model small --device cuda
```

This selects the Small model on GPU, preprocesses the audio, and prints the **top 5 labels and scores**.

Select Medium with the same GPU settings, classify multiple files, or save the results as JSON:

```bash
python inference.py --audio example.wav --model medium --device cuda
python inference.py --audio a.wav b.wav --batch-size 2 --device cuda
python inference.py --audio example.wav --output predictions.json --device cuda
```

Use `--top-k 10` to show more labels, or `--checkpoint /path/to/model.pt` if your weights are stored elsewhere. Run `python inference.py --help` for all options.

### Use in your own Python code

Run from the repository directory, or install the local package with `python -m pip install .`:

```python
from ad_mamba import AudioClassifier

classifier = AudioClassifier(checkpoint="checkpoints/ad_mamba_small.pt", device="cuda")
for item in classifier.predict("example.wav"):
    print(item["label"], item["score"])
```

The model is loaded once and can be reused. For Medium, set `model="medium"` and use its checkpoint.

**Notes:** This is multi-label sound classification, not speech transcription. Multiple labels may apply, and scores do not need to sum to one. Inputs are processed as roughly 10-second clips; longer audio is truncated to the first 1024 frames, without sliding-window inference. The supported labels are listed in `ad_mamba/class_labels_indices.csv`.

## Training and evaluation

The basic workflow is: **load labeled audio, initialize the student from a checkpoint, fine-tune with classification loss and optional distillation, then validate and save the best model**. This is a simplified fine-tuning reference, not a full reproduction of the paper's three-stage training pipeline.

Install the additional dependencies with `python -m pip install -r requirements-train.txt`. Use JSON data lists such as `{"data":[{"wav":"audio/example.wav","labels":"/m/09x0r"}]}`. Relative audio paths are resolved from the JSON file's directory; labels are AudioSet MIDs.

```bash
python train.py --checkpoint checkpoints/ad_mamba_small.pt --train-data train.json --val-data eval.json --output-dir runs/finetune --device cuda
python evaluate.py --data eval.json --output runs/metrics.json --device cuda
```

Training saves `best.pt` and `last.pt`; evaluation reports mAP. Optional distillation uses `--teacher teacher.ts.pt`: provide a trusted TorchScript teacher accepting normalized `[B,1024,128]` inputs and returning `[B,527]` logits in the same label order. See `--help` for other options.
