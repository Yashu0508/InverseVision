"""Shared reproducible deblurring problem setup for controlled experiments."""

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import sparse
from skimage import data

from analysis.svd_analysis import SVDAnalysis, analyze_svd
from forward_model.degradation import DegradationResult, degrade_image
from forward_model.gaussian import gaussian_blur_operator_2d
from preprocessing.images import normalize_image


@dataclass(frozen=True)
class DeblurringExperimentProblem:
    """One original, operator, degradation, observation, and aligned SVD."""

    original: np.ndarray
    operator: sparse.csr_matrix
    degradation: DegradationResult
    observation: np.ndarray
    svd_analysis: SVDAnalysis


def create_camera_deblurring_problem(
    image_shape: tuple[int, int],
    sigma: float,
    kernel_size: int,
    noise_standard_deviation: float,
    random_seed: int,
    rank_relative_tolerance: float,
) -> DeblurringExperimentProblem:
    """Create one reproducible camera-image problem using existing APIs."""
    camera = Image.fromarray(data.camera())
    resized = camera.resize(
        (image_shape[1], image_shape[0]),
        resample=Image.Resampling.LANCZOS,
    )
    original = normalize_image(np.asarray(resized))
    degradation = degrade_image(
        original,
        sigma=sigma,
        kernel_size=kernel_size,
        noise_standard_deviation=noise_standard_deviation,
        seed=random_seed,
    )
    operator = gaussian_blur_operator_2d(
        image_shape,
        sigma=sigma,
        kernel_size=kernel_size,
    )
    observation = degradation.noisy_observation.ravel(order="C")
    svd = analyze_svd(
        operator,
        relative_tolerance=rank_relative_tolerance,
    )
    return DeblurringExperimentProblem(
        original=original,
        operator=operator,
        degradation=degradation,
        observation=observation,
        svd_analysis=svd,
    )
