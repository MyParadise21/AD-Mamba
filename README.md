# AD-Mamba

Classify audio with pretrained AD-Mamba models covering **527 AudioSet sound events**. No training is needed for inference. This project is intended for NVIDIA GPU use; CPU execution is not a validated deployment target.

## Pretrained models

Small and Medium are two model sizes with different network depths. Both were trained on GPUs and use the same GPU environment; they are not CPU and GPU editions.

| Model | Approx. parameters | AudioSet mAP | Checkpoint |
|---|---:|---:|---|
| Small (default) | 30M | 50.3% | `ad_mamba_small.pt` |
| Medium | 49M | 50.6% | `ad_mamba_medium.pt` |

Place the checkpoint files in `checkpoints/`. Weights will be distributed as GitHub Release assets, separately from the Python package. The mAP values above are historical single-model evaluation results; full AudioSet evaluation has not been rerun for this release.

## Installation

For the VMamba-derived CUDA path, use **Linux, Python 3.10, an NVIDIA GPU and a compatible NVIDIA driver**. The pinned PyTorch 2.0 stack below uses CUDA 11.8 wheels. This is an installation recipe, not a newly tested environment.

```bash
python -m pip install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

The `oflex` backend requires VMamba's `selective_scan_cuda_oflex` extension. Following the [VMamba installation guide](https://github.com/MzeroMiko/VMamba#installation), install a compatible CUDA Toolkit (`nvcc`), a C++ compiler and the build dependencies, then build the extension in the same Python environment. A CUDA-enabled PyTorch wheel alone does not provide the compiler toolchain.

```bash
python -m pip install packaging ninja setuptools wheel
git clone https://github.com/MzeroMiko/VMamba.git
python -m pip install --no-build-isolation ./VMamba/kernels/selective_scan
```

The Toolkit must be compatible with the installed PyTorch CUDA build; the upstream build also queries the available GPU. Installing `mamba-ssm` alone does not provide the `selective_scan_cuda_oflex` module used here. This repository does not import VMamba's full training stack, so unrelated VMamba dependencies are not required.

**Hardware compatibility:** the pinned CUDA 11.8 / PyTorch 2.0 environment is not suitable for RTX 50-series (Blackwell) GPUs. Those require a newer, mutually compatible PyTorch/TorchAudio CUDA build, updated dependency pins and a rebuilt extension. Native Windows builds of the VMamba extension are not validated here.

**Backends:** the examples below explicitly select `cuda` and `oflex`. The existing code defaults remain `cpu` and `torch`; do not omit these options when following the GPU instructions. The `torch` backend is a pure-PyTorch reference implementation and can be selected with `--scan-backend torch` without compiling the extension. Its presence does not establish CPU support or numerical/performance equivalence to the GPU results; neither backend has been revalidated in this documentation update.

## Inference

Replace `example.wav` with the path to your own audio file:

```bash
python inference.py --audio example.wav --model small --device cuda --scan-backend oflex
```

This selects the Small model on GPU, preprocesses the audio, and prints the **top 5 labels and scores**.

Select Medium with the same GPU settings, classify multiple files, or save the results as JSON:

```bash
python inference.py --audio example.wav --model medium --device cuda --scan-backend oflex
python inference.py --audio a.wav b.wav --batch-size 2 --device cuda --scan-backend oflex
python inference.py --audio example.wav --output predictions.json --device cuda --scan-backend oflex
```

Use `--top-k 10` to show more labels, or `--checkpoint /path/to/model.pt` if your weights are stored elsewhere. Run `python inference.py --help` for all options.

### Use in your own Python code

Run from the repository directory, or install the local package with `python -m pip install .`:

```python
from ad_mamba import AudioClassifier

classifier = AudioClassifier(
    checkpoint="checkpoints/ad_mamba_small.pt", device="cuda", scan_backend="oflex"
)
for item in classifier.predict("example.wav"):
    print(item["label"], item["score"])
```

The model is loaded once and can be reused for multiple `predict` calls. For Medium, set `model="medium"` and provide its checkpoint; keep the same device and backend settings.

**Notes:** This is multi-label sound classification, not speech transcription. Multiple labels may apply, and scores do not need to sum to one. Inputs are processed as roughly 10-second clips; longer audio is truncated to the first 1024 frames, without sliding-window inference. The supported labels are listed in `ad_mamba/class_labels_indices.csv`.

## Training and evaluation

The basic workflow is: **load labeled audio, initialize the student from a checkpoint, fine-tune with classification loss and optional distillation, then validate and save the best model**. This is a simplified fine-tuning reference, not a full reproduction of the paper's three-stage training pipeline.

Install the additional dependencies with `python -m pip install -r requirements-train.txt`. Use JSON data lists such as `{"data":[{"wav":"audio/example.wav","labels":"/m/09x0r"}]}`. Relative audio paths are resolved from the JSON file's directory; labels are AudioSet MIDs.

```bash
python train.py --checkpoint checkpoints/ad_mamba_small.pt --train-data train.json --val-data eval.json --output-dir runs/finetune --device cuda --scan-backend oflex
python evaluate.py --data eval.json --output runs/metrics.json --device cuda --scan-backend oflex
```

Training saves `best.pt` and `last.pt`; evaluation reports mAP. Optional distillation uses `--teacher teacher.ts.pt`: provide a trusted TorchScript teacher accepting normalized `[B,1024,128]` inputs and returning `[B,527]` logits in the same label order. See `--help` for other options.
