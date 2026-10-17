"""Simple full-student fine-tuning / online logit distillation, not the full paper recipe."""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchaudio

from .audio import MEAN, STD
from .data import AudioSetDataset
from .evaluate import evaluate
from .model import load_model


def distillation_loss(student, labels, teacher=None, weight=0.5, temperature=2.0):
    """Historical KL: soften only teacher logits; no temperature-squared multiplier."""
    if not 0 <= weight <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid distillation weight or temperature.")
    bce = F.binary_cross_entropy_with_logits(student.float(), labels.float())
    if teacher is None:
        return bce
    if not isinstance(teacher, torch.Tensor) or teacher.shape != student.shape or not torch.isfinite(teacher).all():
        raise ValueError("Teacher must return finite logits with the same [B, 527] shape as the student.")
    kl = F.kl_div(F.log_softmax(student.float(), dim=1),
                  F.log_softmax(teacher.detach().float() / temperature, dim=1),
                  reduction="batchmean", log_target=True)
    return (1 - weight) * bce + weight * kl


def augment_batch(features, labels, frequency_mask=48, time_mask=192, mixup_probability=0.5):
    """Lightweight feature-space augmentation; teacher receives the exact same tensor."""
    if features.shape[0] > 1 and random.random() < mixup_probability:
        ratio = float(np.random.beta(10.0, 10.0))
        permutation = torch.randperm(features.shape[0], device=features.device)
        features = ratio * features + (1 - ratio) * features[permutation]
        labels = ratio * labels + (1 - ratio) * labels[permutation]
    raw = (features * (STD * 2) + MEAN).transpose(1, 2)
    if frequency_mask:
        raw = torchaudio.transforms.FrequencyMasking(frequency_mask, iid_masks=True)(raw)
    if time_mask:
        raw = torchaudio.transforms.TimeMasking(time_mask, iid_masks=True)(raw)
    return ((raw - MEAN) / (STD * 2)).transpose(1, 2), labels


def train_one_epoch(model, loader, optimizer, device="cpu", teacher=None,
                    accumulation_steps=1, kd_weight=0.5, temperature=2.0,
                    augment=True, frequency_mask=48, time_mask=192,
                    mixup_probability=0.5, grad_clip=5.0):
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive.")
    model.train()
    if teacher is not None:
        teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    total_loss, total_examples, pending_examples, updates = 0.0, 0, 0, 0
    for step, (features, labels) in enumerate(loader, 1):
        features, labels = features.to(device), labels.to(device)
        if augment:
            features, labels = augment_batch(features, labels, frequency_mask, time_mask, mixup_probability)
        with torch.no_grad():
            teacher_logits = teacher(features) if teacher is not None else None
        logits = model(features)
        loss = distillation_loss(logits, labels, teacher_logits, kd_weight, temperature)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss.")
        count = len(labels)
        (loss * count).backward()
        total_loss += loss.item() * count
        total_examples += count
        pending_examples += count
        if step % accumulation_steps == 0 or step == len(loader):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending_examples)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending_examples = 0
            updates += 1
    if not total_examples:
        raise ValueError("Empty training loader.")
    return dict(loss=total_loss / total_examples, samples=total_examples, optimizer_steps=updates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--val-data", required=True, type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Student initialization, NOT optimizer resume")
    parser.add_argument("--model", choices=("small", "medium"), default="small")
    parser.add_argument("--teacher", type=Path, help="Trusted TorchScript: normalized [B,1024,128] -> raw [B,527] logits")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--scan-backend", choices=("torch", "oflex"), default="torch")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-7)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-step", type=int, default=20)
    parser.add_argument("--lr-gamma", type=float, default=0.5)
    parser.add_argument("--kd-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--frequency-mask", type=int, default=48)
    parser.add_argument("--time-mask", type=int, default=192)
    parser.add_argument("--mixup-probability", type=float, default=0.5)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--drop-path", type=float)
    parser.add_argument("--samples-per-epoch", type=int, help="Replacement sampling; default traverses all training rows")
    parser.add_argument("--sample-weights", type=Path, help="One nonnegative sampling weight per training row (text)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.accumulation_steps, args.threads, args.lr_step) < 1 or args.workers < 0 or args.warmup_epochs < 0:
        parser.error("Invalid epoch, batch, accumulation, worker, thread or scheduler settings.")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("lr must be positive and weight-decay nonnegative, both finite.")
    if not 0 <= args.kd_weight <= 1 or not math.isfinite(args.temperature) or args.temperature <= 0 or not 0 < args.lr_gamma <= 1:
        parser.error("Invalid distillation or scheduler parameters.")
    if not 0 <= args.frequency_mask <= 128 or not 0 <= args.time_mask <= 1024 or not 0 <= args.mixup_probability <= 1:
        parser.error("Invalid augmentation settings.")
    if args.samples_per_epoch is not None and args.samples_per_epoch < 1:
        parser.error("samples-per-epoch must be positive.")
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("output-dir must be new or empty; existing experiments are never overwritten.")
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
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        train_data = AudioSetDataset(args.train_data, args.audio_root)
        val_data = AudioSetDataset(args.val_data, args.audio_root)
        sampler = None
        if args.samples_per_epoch is not None or args.sample_weights is not None:
            weights = np.loadtxt(args.sample_weights).reshape(-1) if args.sample_weights else np.ones(len(train_data))
            if len(weights) != len(train_data) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
                raise ValueError("Invalid sample weights; expected one finite nonnegative value per training row and a positive sum.")
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                            args.samples_per_epoch or len(train_data), replacement=True)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=sampler is None,
                                  sampler=sampler, num_workers=args.workers)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, num_workers=args.workers)
        model = load_model(args.checkpoint, args.model, args.device, args.scan_backend)
        model.configure_finetuning(args.drop_path, not args.no_gradient_checkpointing)
        teacher = torch.jit.load(str(args.teacher), map_location=args.device).eval() if args.teacher else None
        if teacher is not None:
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.95, 0.999))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(training_scope="simplified full-student fine-tuning; not the paper's three-stage reproduction",
                      initialization_only=True, optimizer_resume=False, precision="float32")
        (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        history, best_map = [], -1.0
        for epoch in range(1, args.epochs + 1):
            warmup = 0.01 ** max(0.0, 1 - epoch / args.warmup_epochs) if args.warmup_epochs else 1.0
            lr = args.lr * warmup * args.lr_gamma ** ((epoch - 1) // args.lr_step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            training = train_one_epoch(model, train_loader, optimizer, args.device, teacher,
                                       args.accumulation_steps, args.kd_weight, args.temperature,
                                       not args.no_augment, args.frequency_mask, args.time_mask, args.mixup_probability)
            validation = evaluate(model, val_loader, args.device)
            row = dict(epoch=epoch, lr=lr, train=training,
                       validation={key: value for key, value in validation.items() if key != "per_class"})
            history.append(row)
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(state, args.output_dir / "last.pt")
            if validation["mAP"] > best_map:
                best_map = validation["mAP"]
                torch.save(state, args.output_dir / "best.pt")
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"AD-Mamba training failed: {exc}\n")


if __name__ == "__main__":
    main()
