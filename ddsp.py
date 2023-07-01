import torch
import torch.nn.functional as F

@torch.jit.script
def comb(x: torch.Tensor, b: float = 1.0, M: int = 2000, a: float = 0.9) -> torch.Tensor:
    y = torch.zeros(x.shape[-1] + M, dtype=x.dtype, device=x.device)
    feedback = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    for i in range(y.shape[-1]):
        if i < x.shape[-1]:
            y[i] = b * x[i]
        if i >= M:
            y[i] += feedback
            feedback = -a * y[i - M]
    return y

@torch.jit.script
def lbcf(x: torch.Tensor, b: float = 1.0, M: int = 2000, a: float = 0.9, d: float = 0.5) -> torch.Tensor:
    y = torch.zeros(x.shape[-1] + M, dtype=x.dtype, device=x.device)
    feedback = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    for i in range(y.shape[-1]):
        if i < x.shape[-1]:
            y[i] = b * x[0][i]
        if i >= M:
            y[i] += feedback
            if i - M >= y.shape[-1]:
                feedback += (1 - d) * ((a * y[-1]) - feedback)
            else:
                feedback += (1 - d) * ((a * y[i - M]) - feedback)
    return y

@torch.jit.script
def allpass(x: torch.Tensor, M: int = 2000, a: float = 0.5) -> torch.Tensor:
    y = torch.zeros(x.shape[-1] + M, dtype=x.dtype, device=x.device)
    feedback = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    for i in range(y.shape[-1]):
        if i < x.shape[-1]:
            y[i] = x[i] - feedback
            feedback *= a
            if i >= M:
                feedback += x[i]
        else:
            y[i] -= feedback
            feedback *= a
    return y

@torch.jit.script
def freeverb(
        x: torch.Tensor,
        cb: torch.Tensor = torch.tensor([1.0] * 8),
        cM: torch.Tensor = torch.tensor([1557, 1617, 1491, 1422, 1277, 1356, 1188, 1116]),
        ca: torch.Tensor = torch.tensor([0.84] * 8),
        cd: torch.Tensor = torch.tensor([0.2] * 8),
        aM: torch.Tensor = torch.tensor([225, 556, 441, 341]),
        aa: torch.Tensor = torch.tensor([0.5] * 4)
) -> torch.Tensor:

    y = torch.zeros(x.shape[-1], dtype=x.dtype, device=x.device)

    # Comb filters
    for b, M, a, d in zip(cb, cM, ca, cd):
        y_ = lbcf(x=x, b=b, M=M, a=a, d=d)
        shape = y.shape[-1]
        shape_ = y_.shape[-1]
        pad_length = abs(shape-shape_)
        if shape < shape_:
            y = F.pad(y, (0, pad_length))
        elif shape_ < shape:
            y_ = F.pad(y_, (0, pad_length))
        y.add_(y_)

    # Allpass filters
    for M, a in zip(aM, aa):
        y = allpass(y, M, a)

    max_abs_value = torch.max(torch.abs(y))
    epsilon = 1e-12
    y = y / (max_abs_value + epsilon)

    return y
