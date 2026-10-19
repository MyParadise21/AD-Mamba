# AD-Mamba

Pretrained AD-Mamba models for **527 AudioSet sound events**. Run inference on your own audio with an NVIDIA GPU.

## Pretrained models

| Model | Approx. parameters | AudioSet mAP | Download |
|---|---:|---:|---|
| Small (default) | 30M | 50.3% | [Checkpoint](https://github.com/MyParadise21/AD-Mamba/releases/download/pretrained/ad_mamba_small.pt) |
| Medium | 49M | 50.6% | [Checkpoint](https://github.com/MyParadise21/AD-Mamba/releases/download/pretrained/ad_mamba_medium.pt) |

## Installation

Recommended: **Python 3.11** and an NVIDIA GPU.

```bash
git clone https://github.com/MyParadise21/AD-Mamba.git
cd AD-Mamba
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Use a driver compatible with the selected CUDA wheels. No custom CUDA extensions are required. Existing environments need matching PyTorch/TorchAudio versions satisfying `requirements.txt` (both `<2.9`).

## Inference

From the repository directory, download Pretrained models and run inference. Replace `example.wav` with your own audio file.

```bash
mkdir -p checkpoints
curl -fL https://github.com/MyParadise21/AD-Mamba/releases/download/pretrained/ad_mamba_small.pt -o checkpoints/ad_mamba_small.pt
python inference.py --audio example.wav --model small --device cuda
```

This prints the **top 5 labels and sigmoid scores**. For Medium, download its checkpoint above to `checkpoints/ad_mamba_medium.pt` and use `--model medium`.

```bash
python inference.py --audio a.wav b.wav --batch-size 2 --device cuda --output predictions.json
```

Use `--top-k 10` for more labels or `--checkpoint /absolute/path/to/model.pt` for weights elsewhere. See `python inference.py --help` for all options.

### Use in your own Python code

Run from the repository directory, or install with `python -m pip install .`. Use an absolute checkpoint path when calling the installed package from another directory; weights are not included in the package.

```python
from ad_mamba import AudioClassifier

classifier = AudioClassifier(checkpoint="/absolute/path/to/ad_mamba_small.pt", device="cuda")
for item in classifier.predict("example.wav"):
    print(item["label"], item["score"])
```

Reuse the classifier across files. For Medium, set `model="medium"` and supply its checkpoint.

**Notes:** This is multi-label sound classification, not transcription. Scores need not sum to one; top-k results are rankings, not thresholded detections. Audio is padded or truncated to 1024 frames (roughly 10 seconds), without sliding windows. Labels are in `ad_mamba/class_labels_indices.csv`.

## Training and evaluation

Install `requirements-train.txt`. Provide JSON lists such as `{"data":[{"wav":"audio/example.wav","labels":"/m/09x0r"}]}`; audio paths are relative to the JSON file and labels are AudioSet MIDs.

```bash
python -m pip install -r requirements-train.txt
python train.py --checkpoint checkpoints/ad_mamba_small.pt --train-data train.json --val-data eval.json --output-dir runs/finetune --device cuda
python evaluate.py --checkpoint runs/finetune/best.pt --data eval.json --output runs/metrics.json --device cuda
```

Training saves `best.pt` and `last.pt`; the command above evaluates the fine-tuned checkpoint and reports mAP. Optional `--teacher teacher.ts.pt` accepts a trusted TorchScript teacher mapping normalized `[B,1024,128]` inputs to `[B,527]` logits in the same label order.
