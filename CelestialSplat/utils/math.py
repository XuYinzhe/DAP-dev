import torch
from torch import autograd
from typing import Tuple, Union, Optional

SoftClampRange = Tuple[Union[torch.Tensor, float], Union[torch.Tensor, float]]


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Compute inverse sigmoid."""
    return torch.log(x / (1.0 - x))


def inverse_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute inverse softplus.

    Numerically stable implementation matching ml-sharp's math.py.
    """
    x = x.clamp_min(eps)
    sigmoid = torch.sigmoid(-x)
    exp = sigmoid / (1.0 - sigmoid)
    return x + torch.log(-exp + 1.0)


def softclamp(
    tensor: torch.Tensor,
    min: Optional[SoftClampRange] = None,
    max: Optional[SoftClampRange] = None,
) -> torch.Tensor:
    """Clamp tensor to min/max in a differentiable way using tanh smoothing.

    Unlike torch.clamp which has zero gradient beyond the threshold,
    softclamp provides a smooth transition and non-vanishing gradients.

    Args:
        tensor: The tensor to clamp.
        min: Pair of (threshold_to_start_clamping, clamp_target_value).
            threshold should be > target. E.g. (0.1, 0.01) starts smoothing
            from 0.1 down to 0.01.
        max: Pair of (threshold_to_start_clamping, clamp_target_value).
            threshold should be < target. E.g. (100.0, 1000.0) starts smoothing
            from 100 up to 1000.

    Returns:
        The smoothed-clamped tensor.
    """

    def normalize(clamp_range: SoftClampRange) -> torch.Tensor:
        value0, value1 = clamp_range
        return value0 + (value1 - value0) * torch.tanh((tensor - value0) / (value1 - value0))

    tensor_clamped = tensor
    if min is not None:
        tensor_clamped = torch.maximum(tensor_clamped, normalize(min))
    if max is not None:
        tensor_clamped = torch.minimum(tensor_clamped, normalize(max))

    return tensor_clamped


class _ClampWithPushback(autograd.Function):
    """Implementation of clamp_with_pushback function.

    Equivalent to adding a regularizer:
        pushback * sum_i (relu(min - preactivation_i) + relu(preactivation_i - max))
    to the loss, which actively pushes clamped values back into range.
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        min: Optional[float],
        max: Optional[float],
        pushback: float,
    ) -> torch.Tensor:
        if min is not None and max is not None and min >= max:
            raise ValueError("Only min < max is supported.")

        ctx.save_for_backward(tensor)
        ctx.min = min
        ctx.max = max
        ctx.pushback = pushback
        return torch.clamp(tensor, min=min, max=max)

    @staticmethod
    def backward(ctx, grad_in: torch.Tensor):
        grad_out = grad_in.clone()
        (tensor,) = ctx.saved_tensors

        if ctx.min is not None:
            mask_min = tensor < ctx.min
            grad_out[mask_min] = -ctx.pushback

        if ctx.max is not None:
            mask_max = tensor > ctx.max
            grad_out[mask_max] = ctx.pushback

        return grad_out, None, None, None


def clamp_with_pushback(
    tensor: torch.Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
    pushback: float = 1e-2,
) -> torch.Tensor:
    """Variant of clamp that avoids vanishing gradients.

    For values outside the clamp range, instead of zero gradient,
    returns a constant pushback gradient that drives the preactivation
    back into the valid range.

    Args:
        tensor: Input tensor.
        min: Minimum clamp value.
        max: Maximum clamp value.
        pushback: Gradient magnitude for clamped values. Should be > 0
            for minimization problems.

    Returns:
        Clamped tensor with pushback gradients.
    """
    output = _ClampWithPushback.apply(tensor, min, max, pushback)
    assert isinstance(output, torch.Tensor)
    return output
