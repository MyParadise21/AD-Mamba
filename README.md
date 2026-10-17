# AD-Mamba

Classify audio with pretrained AD-Mamba models covering **527 AudioSet sound events**. No training is needed for inference. Both CPU and NVIDIA GPU are supported.

## Pretrained models

| Model | Parameters | AudioSet mAP | Checkpoint | Size |
|---|---:|---:|---|---:|
| Small (default) | 30M | 50.3% | [ad_mamba_small.pt](https://github.com/MyParadise21/AD-Mamba/releases/download/v0.3.0/ad_mamba_small.pt) | 114.54 MiB |
| Medium | 49M | 50.6% | [ad_mamba_medium.pt](https://github.com/MyParadise21/AD-Mamba/releases/download/v0.3.0/ad_mamba_medium.pt) | 186.58 MiB |

Download the desired checkpoint from [Release v0.3.0](https://github.com/MyParadise21/AD-Mamba/releases/tag/v0.3.0), create `checkpoints/` if needed, and place it there. Weights are distributed as GitHub Release assets, separately from the Git source tree and Python package. Use the matching `v0.3.0` source tag for this release.

The mAP values above are historical single-model evaluation results; full AudioSet evaluation has not been rerun for this release.

## Installation

Use **Python 3.10** and run these commands from the repository directory:

```bash
python -m pip install torch==2.0.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For NVIDIA GPUs, replace `/cpu` with `/cu118` in the first command. If these dependencies are already installed, you can skip this step. No custom Mamba extension needs to be compiled.

## Inference

Replace `example.wav` with the path to your own audio file:

```bash
python inference.py --audio example.wav
```

This loads the Small model on CPU, preprocesses the audio automatically, and prints the **top 5 labels and scores**.

Use the Medium model on GPU, classify multiple files, or save the results as JSON:

```bash
python inference.py --audio example.wav --model medium --device cuda
python inference.py --audio a.wav b.wav --batch-size 2
python inference.py --audio example.wav --output predictions.json
```

Use `--top-k 10` to show more labels, or `--checkpoint /path/to/model.pt` if your weights are stored elsewhere. Run `python inference.py --help` for all options.

### Use in your own Python code

Run from the repository directory, or install the local package with `python -m pip install .`:

```python
from ad_mamba import AudioClassifier

classifier = AudioClassifier(checkpoint="checkpoints/ad_mamba_small.pt")
for item in classifier.predict("example.wav"):
    print(item["label"], item["score"])
```

The model is loaded once and can be reused for multiple `predict` calls. For Medium, set `model="medium"` and provide its checkpoint. Add `device="cuda"` to use a GPU.

**Notes:** This is multi-label sound classification, not speech transcription. Multiple labels may apply, and scores do not need to sum to one. Inputs are processed as roughly 10-second clips; longer audio is truncated to the first 1024 frames, without sliding-window inference. The supported labels are listed in `ad_mamba/class_labels_indices.csv`.

## Training and evaluation

The basic workflow is: **load labeled audio, initialize the student from a checkpoint, fine-tune with classification loss and optional distillation, then validate and save the best model**. This is a simplified fine-tuning reference, not a full reproduction of the paper's three-stage training pipeline.

Install the additional dependencies with `python -m pip install -r requirements-train.txt`. Use JSON data lists such as `{"data":[{"wav":"audio/example.wav","labels":"/m/09x0r"}]}`. Relative audio paths are resolved from the JSON file's directory; labels are AudioSet MIDs.

```bash
python train.py --checkpoint checkpoints/ad_mamba_small.pt --train-data train.json --val-data eval.json --output-dir runs/finetune
python evaluate.py --data eval.json --output runs/metrics.json
```

Training saves `best.pt` and `last.pt`; evaluation reports mAP. Optional distillation uses `--teacher teacher.ts.pt`: provide a trusted TorchScript teacher accepting normalized `[B,1024,128]` inputs and returning `[B,527]` logits in the same label order. See `--help` for other options.

## Paper and license

The paper is not yet available on arXiv; its link will be added once it is posted.

AD-Mamba-specific code and the released Small and Medium model weights are licensed under the **MIT License**. All existing third-party copyright notices and license terms are retained in [LICENSE](LICENSE) and continue to apply to the relevant third-party components. Third-party datasets and audio recordings are not covered by this license.
