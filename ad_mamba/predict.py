"""Load an AD-Mamba checkpoint once and classify audio files."""
from pathlib import Path

import torch

from .audio import load_audio, load_labels
from .model import load_model


class AudioClassifier:
    """AudioSet tagging with automatic audio preprocessing and label decoding."""

    def __init__(self, model="small", checkpoint=None, device="cpu", scan_backend="torch", threads=4):
        if threads < 1:
            raise ValueError("threads must be positive.")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA devices are supported.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use device='cpu'.")
        if scan_backend == "oflex" and self.device.type != "cuda":
            raise ValueError("The oflex backend requires CUDA.")
        if model not in ("small", "medium"):
            raise ValueError("model must be small or medium.")
        self.checkpoint = Path(checkpoint) if checkpoint is not None else (
            Path(__file__).resolve().parents[1] / "checkpoints" / f"ad_mamba_{model}.pt"
        )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint}. "
                f"Place ad_mamba_{model}.pt in checkpoints/ or provide checkpoint= explicitly."
            )
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model_size = model
        self.scan_backend = scan_backend
        self.network = load_model(self.checkpoint, model, self.device, scan_backend)
        self.labels = load_labels()

    def predict(self, audio, top_k=5):
        """Return the top labels as [{label, score, index, mid}, ...] for one file."""
        return self.predict_many([audio], top_k=top_k)[0]["top_k"]

    @torch.inference_mode()
    def predict_many(self, audio_files, top_k=5, batch_size=1, all_scores=False):
        """Return one result per file, optionally including all 527 sigmoid scores."""
        if not 1 <= top_k <= 527 or batch_size < 1:
            raise ValueError("top_k must be 1..527 and batch_size must be positive.")
        if isinstance(audio_files, (str, Path)):
            raise ValueError("Use predict(path) for one file or predict_many([path, ...]).")
        paths = [Path(path) for path in audio_files]
        if not paths:
            raise ValueError("Provide at least one audio file.")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError("Audio not found: " + ", ".join(missing))
        predictions = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            inputs = torch.stack([load_audio(path) for path in batch_paths]).to(self.device)
            logits = self.network(inputs).float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite model output.")
            scores = logits.sigmoid().cpu()
            for path, values in zip(batch_paths, scores):
                top_scores, indices = values.topk(top_k)
                top = [dict(label=self.labels[i]["display_name"], score=s, index=i, mid=self.labels[i]["mid"])
                       for s, i in zip(top_scores.tolist(), indices.tolist())]
                result = dict(audio=str(path), top_k=top)
                if all_scores:
                    result["scores"] = values.tolist()
                predictions.append(result)
        return predictions
