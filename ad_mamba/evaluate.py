"""Single-checkpoint AudioSet mAP evaluation; no prediction ensembling."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .audio import load_labels
from .data import AudioSetDataset
from .model import load_model


def classification_metrics(targets, scores):
    targets, scores = np.asarray(targets), np.asarray(scores)
    if targets.ndim != 2 or targets.shape != scores.shape or targets.shape[0] < 1 or targets.shape[1] != 527:
        raise ValueError("Expected nonempty targets and scores with shape [N, 527].")
    if not np.isfinite(scores).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("Scores must be finite and evaluation targets must be binary.")
    positives = targets.sum(axis=0)
    ap = np.zeros(527, dtype=np.float64)
    for index in np.flatnonzero(positives):
        ap[index] = average_precision_score(targets[:, index], scores[:, index])
    present = positives > 0
    return {
        "samples": len(targets), "classes": 527, "mAP": float(ap.mean()),
        "mAP_percent": float(ap.mean() * 100),
        "classes_with_positives": int(present.sum()),
        "mAP_present_classes": float(ap[present].mean()) if present.any() else None,
        "absent_class_policy": "AP=0; mAP always averages all 527 classes",
        "per_class": [dict(index=i, mid=label["mid"], label=label["display_name"],
                           positives=int(positives[i]), AP=float(ap[i]))
                      for i, label in enumerate(load_labels())],
    }


@torch.inference_mode()
def evaluate(model, loader, device="cpu"):
    was_training = model.training
    model.eval()
    targets, scores = [], []
    loss_sum = 0.0
    try:
        for features, labels in loader:
            labels = labels.to(device)
            logits = model(features.to(device)).float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite evaluation logits.")
            loss_sum += F.binary_cross_entropy_with_logits(logits, labels, reduction="sum").item()
            targets.append(labels.cpu().numpy())
            scores.append(logits.sigmoid().cpu().numpy())
    finally:
        model.train(was_training)
    if not targets:
        raise ValueError("Empty evaluation loader.")
    result = classification_metrics(np.concatenate(targets), np.concatenate(scores))
    result["bce"] = loss_sum / (result["samples"] * 527)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="AudioSet evaluation JSON")
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--model", choices=("small", "medium"), default="small")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scan-backend", choices=("torch", "oflex"), default="torch")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Smoke check only: evaluate first N records")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) < 1 or args.workers < 0:
        parser.error("batch-size/threads must be positive; workers must be nonnegative.")
    if args.output.exists():
        parser.error("Output already exists; choose a new output path.")
    try:
        device = torch.device(args.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA devices are supported.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu.")
        if args.scan_backend == "oflex" and device.type != "cuda":
            raise ValueError("--scan-backend oflex requires CUDA.")
        torch.set_num_threads(args.threads)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dataset = AudioSetDataset(args.data, args.audio_root, args.limit)
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
        checkpoint = args.checkpoint or Path(__file__).resolve().parents[1] / "checkpoints" / f"ad_mamba_{args.model}.pt"
        model = load_model(checkpoint, args.model, args.device, args.scan_backend)
        result = evaluate(model, loader, args.device)
        result.update(model=args.model, checkpoint=str(checkpoint), data=str(args.data),
                      subset=args.limit is not None, scan_backend=args.scan_backend,
                      metric_source="current evaluation, not historical CSV",
                      official_split_verified=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "per_class"}, indent=2))
        if args.limit is not None or result["classes_with_positives"] < 527:
            print("WARNING: subset/incomplete class coverage; not a reproduction of the paper mAP.", file=sys.stderr)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"AD-Mamba evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
