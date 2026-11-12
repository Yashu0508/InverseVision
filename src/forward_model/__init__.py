"""Gaussian image-degradation forward model."""

from forward_model.degradation import DegradationResult, degrade_image
from forward_model.gaussian import (
    gaussian_blur_matrix_1d,
    gaussian_blur_operator_2d,
    gaussian_kernel_1d,
)
from forward_model.noise import add_gaussian_noise
from forward_model.operators import (
    apply_separable_gaussian_blur,
    gaussian_blur_linear_operator,
)

__all__ = [
    "DegradationResult",
    "add_gaussian_noise",
    "apply_separable_gaussian_blur",
    "degrade_image",
    "gaussian_blur_linear_operator",
    "gaussian_blur_matrix_1d",
    "gaussian_blur_operator_2d",
    "gaussian_kernel_1d",
]
