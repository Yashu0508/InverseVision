"""Tests for reconstruction-quality and residual metrics."""

import numpy as np
import pytest

from evaluation.reconstruction_metrics import (
    evaluate_reconstruction,
    mean_squared_error,
    peak_signal_to_noise_ratio,
    relative_reconstruction_error,
    residual_metrics,
    structural_similarity_index,
)


def test_relative_reconstruction_error_matches_definition() -> None:
    reference = np.array([3.0, 4.0])
    reconstruction = np.array([0.0, 4.0])
    assert relative_reconstruction_error(reference, reconstruction) == pytest.approx(
        3.0 / 5.0
    )


def test_mse_and_psnr_match_normalized_definitions() -> None:
    reference = np.zeros((2, 2))
    reconstruction = np.array([[0.0, 0.0], [0.0, 1.0]])
    assert mean_squared_error(reference, reconstruction) == pytest.approx(0.25)
    assert peak_signal_to_noise_ratio(reference, reconstruction) == pytest.approx(
        10.0 * np.log10(4.0)
    )


def test_perfect_reconstruction_metric_edges() -> None:
    image = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    assert relative_reconstruction_error(image, image) == 0.0
    assert mean_squared_error(image, image) == 0.0
    assert np.isinf(peak_signal_to_noise_ratio(image, image))
    assert structural_similarity_index(image, image) == pytest.approx(1.0)


def test_relative_reconstruction_zero_denominator() -> None:
    zeros = np.zeros((3, 3))
    assert relative_reconstruction_error(zeros, zeros) == 0.0
    assert np.isinf(relative_reconstruction_error(zeros, np.ones((3, 3))))


def test_ssim_uses_raw_unclipped_reconstruction() -> None:
    reference = np.linspace(0.0, 1.0, 64).reshape(8, 8)
    outside_range = reference + 1.5
    value = structural_similarity_index(reference, outside_range)
    assert np.isfinite(value)
    assert value < 1.0


def test_residual_norm_and_relative_residual() -> None:
    matrix = np.array([[2.0, 0.0], [0.0, 3.0]])
    reconstruction = np.array([1.0, 1.0])
    observation = np.array([2.0, 4.0])
    norm, relative = residual_metrics(matrix, reconstruction, observation)
    assert norm == pytest.approx(1.0)
    assert relative == pytest.approx(1.0 / np.sqrt(20.0))


def test_relative_residual_zero_denominator() -> None:
    matrix = np.eye(2)
    norm, relative = residual_metrics(matrix, np.zeros(2), np.zeros(2))
    assert norm == 0.0
    assert relative == 0.0

    norm, relative = residual_metrics(matrix, np.ones(2), np.zeros(2))
    assert norm == pytest.approx(np.sqrt(2.0))
    assert np.isinf(relative)


def test_complete_evaluation_keeps_error_and_residual_distinct() -> None:
    reference = np.zeros((3, 3))
    reconstruction = np.ones((3, 3))
    matrix = np.eye(9)
    observation = np.ones(9)

    metrics = evaluate_reconstruction(
        reference,
        reconstruction,
        matrix,
        observation,
    )

    assert np.isinf(metrics.relative_reconstruction_error)
    assert metrics.residual_norm == 0.0
    assert metrics.relative_residual == 0.0
