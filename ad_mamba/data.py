"""AudioSet JSON manifests; relative audio paths resolve against the manifest."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .audio import load_audio, load_labels


class AudioSetDataset(Dataset):
    def __init__(self, manifest, audio_root=None, limit=None):
        manifest = Path(manifest).resolve()
        with manifest.open(encoding="utf-8") as stream:
            document = json.load(stream)
        rows = document.get("data") if isinstance(document, dict) else document
        if not isinstance(rows, list) or not rows:
            raise ValueError("Manifest must be a nonempty list or {\"data\": [...]}.")
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive.")
            rows = rows[:limit]
        root = Path(audio_root).resolve() if audio_root else manifest.parent
        indices = {row["mid"]: int(row["index"]) for row in load_labels()}
        self.items = []
        for number, row in enumerate(rows, 1):
            if not isinstance(row, dict) or not isinstance(row.get("wav"), str) or not row["wav"]:
                raise ValueError(f"Row {number}: expected a nonempty wav path.")
            labels = row.get("labels")
            if isinstance(labels, str):
                labels = [label.strip() for label in labels.split(",") if label.strip()]
            if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
                raise ValueError(f"Row {number}: labels must be comma-separated MIDs or a list of MIDs.")
            unknown = set(labels) - indices.keys()
            if unknown:
                raise ValueError(f"Row {number}: unknown AudioSet MIDs: {sorted(unknown)}")
            path = Path(row["wav"])
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise FileNotFoundError(f"Row {number}: missing audio: {path}")
            self.items.append((path, [indices[label] for label in labels]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, indices = self.items[index]
        target = torch.zeros(527)
        target[indices] = 1.0
        try:
            features = load_audio(path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Cannot decode {path}: {exc}") from exc
        return features, target
