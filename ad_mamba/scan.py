"""Differentiable selective scan for AD-Mamba (d_state=1)."""

import torch
from torch.nn import functional as F


def _affine_prefix(a, b):
    """Inclusive composition of h[t] = a[t] * h[t-1] + b[t]."""
    offset = 1
    while offset < a.shape[-1]:
        next_b = torch.cat((b[..., :offset], b[..., offset:] + a[..., offset:] * b[..., :-offset]), -1)
        a = torch.cat((a[..., :offset], a[..., offset:] * a[..., :-offset]), -1)
        b = next_b
        offset *= 2
    return a, b


def selective_scan_torch(u, delta, A, B, C, D, delta_bias, chunk_size=128):
    """Stable float32 scan without CUDA extensions or exp/cumsum division."""
    batch, channels, length = u.shape
    groups = B.shape[1]
    if A.shape != (channels, 1) or B.shape[2] != 1 or C.shape != B.shape:
        raise ValueError("This scan supports only d_state=1.")
    if channels % groups or length < 1 or chunk_size < 1:
        raise ValueError("Invalid selective scan dimensions.")
    u = u.float().reshape(batch, groups, channels // groups, length)
    delta = F.softplus(delta.float().reshape_as(u) + delta_bias.float().reshape(1, groups, -1, 1))
    a = torch.exp(delta * A.float().reshape(1, groups, -1, 1))
    b = delta * B.float() * u
    padding = (-length) % chunk_size
    a = F.pad(a, (0, padding), value=1).reshape(batch, groups, channels // groups, -1, chunk_size)
    b = F.pad(b, (0, padding)).reshape_as(a)
    a, b = _affine_prefix(a, b)
    _, ends = _affine_prefix(a[..., -1], b[..., -1])
    previous = F.pad(ends[..., :-1], (1, 0))
    states = (b + a * previous.unsqueeze(-1)).flatten(-2)[..., :length]
    y = states * C.float() + D.float().reshape(1, groups, -1, 1) * u
    return y.reshape(batch, channels, length)


class _SelectiveScanOflex(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias):
        import selective_scan_cuda_oflex
        tensors = tuple(value.float().contiguous() for value in (u, delta, A, B, C, D, delta_bias))
        output, states, *_ = selective_scan_cuda_oflex.fwd(*tensors, True, 1, True)
        ctx.save_for_backward(*tensors, states)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        import selective_scan_cuda_oflex
        u, delta, A, B, C, D, delta_bias, states = ctx.saved_tensors
        gradients = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, grad_output.float().contiguous(), states, True, 1,
        )
        return tuple(gradients[:7])


def selective_scan(u, delta, A, B, C, D, delta_bias, backend):
    if backend == "torch":
        return selective_scan_torch(u, delta, A, B, C, D, delta_bias)
    if backend != "oflex":
        raise ValueError("scan_backend must be torch or oflex.")
    if not u.is_cuda:
        raise ValueError("The optional oflex backend requires CUDA tensors.")
    try:
        import selective_scan_cuda_oflex
    except ImportError as exc:
        raise RuntimeError("Optional VMamba oflex extension unavailable; use scan_backend=torch.") from exc
    if torch.is_grad_enabled():
        return _SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias)
    return selective_scan_cuda_oflex.fwd(
        u.contiguous(), delta.contiguous(), A, B.contiguous(), C.contiguous(),
        D, delta_bias, True, 1, True,
    )[0]
