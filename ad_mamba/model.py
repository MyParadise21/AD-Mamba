"""Checkpoint-compatible AD-Mamba architecture.

Parameter names match the original student after removing the module. prefix.
See LICENSE for upstream copyright notices and licenses.
"""
from collections import OrderedDict
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from .scan import selective_scan


class Linear2d(nn.Linear):
    def forward(self, x):
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 5, padding=2, bias=False)

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1)
        weights = self.conv(pooled.squeeze(-1).transpose(1, 2))
        return weights.transpose(1, 2).unsqueeze(-1).sigmoid()


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.SpatialGate = nn.Module()
        self.SpatialGate.spatial = nn.Module()
        self.SpatialGate.spatial.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)

    def forward(self, x):
        return self.SpatialGate.spatial.conv(x.mean(1, keepdim=True)).sigmoid()


class SS2D(nn.Module):
    def __init__(self, dim, scan_backend):
        super().__init__()
        rank = math.ceil(dim / 16)
        self.scan_backend = scan_backend
        self.x_proj_weight = nn.Parameter(torch.zeros(4, rank + 2, dim))
        self.dt_projs_weight = nn.Parameter(torch.zeros(4, dim, rank))
        self.dt_projs_bias = nn.Parameter(torch.zeros(4, dim))
        self.A_logs = nn.Parameter(torch.zeros(4 * dim, 1))
        self.Ds = nn.Parameter(torch.ones(4 * dim))
        self.out_norm = LayerNorm2d(dim)
        self.in_proj = Linear2d(dim, dim, bias=False)
        self.conv2d = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.out_proj = Linear2d(dim, dim, bias=False)

    def forward(self, x):
        x = F.silu(self.conv2d(self.in_proj(x)))
        batch, dim, height, width = x.shape
        length = height * width
        rank = self.dt_projs_weight.shape[-1]
        forward = torch.stack((x.flatten(2), x.transpose(2, 3).flatten(2)), dim=1)
        xs = torch.cat((forward, forward.flip(-1)), dim=1)
        projected = F.conv1d(xs.reshape(batch, -1, length), self.x_proj_weight.reshape(-1, dim, 1), groups=4)
        dt, B, C = projected.reshape(batch, 4, -1, length).split((rank, 1, 1), dim=2)
        dt = F.conv1d(dt.reshape(batch, -1, length), self.dt_projs_weight.reshape(4 * dim, rank, 1), groups=4)
        ys = selective_scan(
            xs.reshape(batch, -1, length), dt, -self.A_logs.float().exp(), B, C,
            self.Ds.float(), self.dt_projs_bias.flatten().float(), self.scan_backend,
        ).reshape(batch, 4, dim, length)
        merged = ys[:, :2] + ys[:, 2:].flip(-1)
        y = merged[:, 0] + merged[:, 1].reshape(batch, dim, width, height).transpose(2, 3).reshape(batch, dim, length)
        y = self.out_norm(y.reshape(batch, dim, height, width)).to(x.dtype)
        return self.out_proj(y)


class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = Linear2d(dim, 4 * dim)
        self.fc2 = Linear2d(4 * dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class VSSBlock(nn.Module):
    def __init__(self, dim, scan_backend):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.op = SS2D(dim, scan_backend)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp(dim)
        self.channel_attn = ChannelAttention()
        self.spatial_attn = SpatialAttention()
        self.lpu1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.lpu2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.drop_path = DropPath()

    def forward(self, x):
        x = x + self.lpu1(x)
        normalized = self.norm(x)
        x = x + self.drop_path(self.op(normalized) * self.channel_attn(normalized) * self.spatial_attn(normalized))
        x = x + self.lpu2(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class VSSM(nn.Module):
    def __init__(self, depths, scan_backend):
        super().__init__()
        dims = (96, 192, 384, 768)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(1, 48, 3, stride=2, padding=1), nn.Identity(), LayerNorm2d(48),
            nn.Identity(), nn.GELU(), nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.Identity(), LayerNorm2d(96),
        )
        self.layers = nn.ModuleList()
        for stage, (dim, depth) in enumerate(zip(dims, depths)):
            downsample = nn.Identity() if stage == 3 else nn.Sequential(
                nn.Identity(), nn.Conv2d(dim, dims[stage + 1], 3, stride=2, padding=1),
                nn.Identity(), LayerNorm2d(dims[stage + 1]),
            )
            self.layers.append(nn.Sequential(OrderedDict(
                blocks=nn.Sequential(*(VSSBlock(dim, scan_backend) for _ in range(depth))),
                downsample=downsample,
            )))
        self.classifier = nn.Sequential(OrderedDict(
            norm=LayerNorm2d(768), permute=nn.Identity(), avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1), head=nn.Linear(768, 527),
        ))

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            if self.training and getattr(self, "gradient_checkpointing", False):
                from torch.utils.checkpoint import checkpoint
                for block in layer.blocks:
                    x = checkpoint(block, x, use_reentrant=False)
                x = layer.downsample(x)
            else:
                x = layer(x)
        return self.classifier(x)


class ADMamba(nn.Module):
    """Checkpoint-compatible student; input [batch, 1024, 128], output 527 logits."""
    DEPTHS = {"small": (2, 2, 8, 2), "medium": (2, 2, 20, 2)}

    def __init__(self, model_size="small", scan_backend="torch"):
        super().__init__()
        if model_size not in self.DEPTHS:
            raise ValueError("model_size must be small or medium.")
        if scan_backend not in ("torch", "oflex"):
            raise ValueError("scan_backend must be torch or oflex.")
        self.model_size = model_size
        self.scan_backend = scan_backend
        self.v = VSSM(self.DEPTHS[model_size], scan_backend)

    def configure_finetuning(self, drop_path_rate=None, gradient_checkpointing=True):
        if drop_path_rate is None:
            drop_path_rate = 0.2 if self.model_size == "small" else 0.3
        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0, 1).")
        blocks = [block for layer in self.v.layers for block in layer.blocks]
        for block, rate in zip(blocks, torch.linspace(0, drop_path_rate, len(blocks)).tolist()):
            block.drop_path.probability = rate
        self.v.gradient_checkpointing = gradient_checkpointing
        return self

    def forward(self, fbank):
        if fbank.ndim != 3 or tuple(fbank.shape[1:]) != (1024, 128) or fbank.shape[0] < 1:
            raise ValueError("Expected nonempty fbank with shape [batch, 1024, 128].")
        return self.v(fbank.unsqueeze(1).transpose(2, 3))


def load_model(checkpoint, model_size="small", device="cpu", scan_backend="torch"):
    """Strictly load tensor-only state_dict; no downloads or teacher dependencies."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a nonempty, tensor-only state_dict.")
    normalized = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key in normalized:
            raise ValueError(f"Duplicate checkpoint key: {key}")
        normalized[key] = value
    model = ADMamba(model_size, scan_backend)
    model.load_state_dict(normalized, strict=True)
    return model.to(device).eval()
