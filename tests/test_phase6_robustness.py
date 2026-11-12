import inspect

import numpy as np
import pytest

from evaluation.robustness import aggregate_records, classify_unstable_solution
from regularization.parameter_selection import select_parameter
from utils.experiment_setup import create_camera_deblurring_problem
from utils.phase6_protocol import (
    PHASE6_IMAGE_SHAPES,
    PHASE6_NOISE_SEEDS,
    Phase6Case,
    build_phase6_cases,
)


def test_phase6_cases_are_controlled_and_respect_dense_limit():
    cases = build_phase6_cases()
    assert len(cases) == 11
    noise_cases = [case for case in cases if case.sensitivity_axis == "noise"]
    assert tuple(case.random_seed for case in noise_cases) == PHASE6_NOISE_SEEDS
    assert len(set(case.random_seed for case in noise_cases)) == len(PHASE6_NOISE_SEEDS)
    assert {case.image_shape for case in cases if case.sensitivity_axis == "resolution"} == set(PHASE6_IMAGE_SHAPES)
    assert all(case.image_shape[0] * case.image_shape[1] <= 256 for case in cases)


def test_phase6_case_rejects_dense_limit_and_invalid_configuration():
    with pytest.raises(ValueError, match="dense SVD limit"):
        Phase6Case(
            case_id="too_large", sensitivity_axis="resolution",
            condition_value="17x17", image_shape=(17, 17), sigma=1.5,
            noise_standard_deviation=0.01, random_seed=1,
        )
    with pytest.raises(ValueError, match="zero-boundary"):
        Phase6Case(
            case_id="wrong_boundary", sensitivity_axis="blur",
            condition_value="test", image_shape=(6, 6), sigma=1.5,
            noise_standard_deviation=0.01, random_seed=1, boundary="reflect",
        )
    with pytest.raises(ValueError, match="finite and positive"):
        Phase6Case(
            case_id="bad_sigma", sensitivity_axis="blur",
            condition_value="test", image_shape=(6, 6), sigma=float("nan"),
            noise_standard_deviation=0.01, random_seed=1,
        )


def test_noise_realization_and_forward_operator_are_reproducible():
    case = next(
        case for case in build_phase6_cases()
        if case.case_id == "noise_seed_42"
    )
    arguments = {
        "image_shape": case.image_shape,
        "sigma": case.sigma,
        "kernel_size": case.kernel_size,
        "noise_standard_deviation": case.noise_standard_deviation,
        "random_seed": case.random_seed,
        "rank_relative_tolerance": 1e-12,
    }
    first = create_camera_deblurring_problem(**arguments)
    second = create_camera_deblurring_problem(**arguments)
    np.testing.assert_array_equal(first.observation, second.observation)
    np.testing.assert_array_equal(first.operator.toarray(), second.operator.toarray())
    forward = first.operator @ first.original.ravel(order="C")
    np.testing.assert_allclose(
        forward, first.degradation.blurred.ravel(order="C"), atol=1e-14
    )
    np.testing.assert_array_equal(
        first.observation,
        first.degradation.noisy_observation.ravel(order="C"),
    )
    other_seed = create_camera_deblurring_problem(
        image_shape=case.image_shape,
        sigma=case.sigma,
        kernel_size=case.kernel_size,
        noise_standard_deviation=case.noise_standard_deviation,
        random_seed=21,
        rank_relative_tolerance=1e-12,
    )
    assert not np.array_equal(first.degradation.noise, other_seed.degradation.noise)


def test_aggregation_uses_descriptive_statistics_and_excludes_failures():
    records = [
        {
            "sensitivity_axis": "noise", "aggregation_condition": "all_noise_seeds",
            "method": "TSVD", "criterion": "gcv", "status": "success",
            "selected_parameter": 2.0, "relative_reconstruction_error": 1.0,
            "unstable_solution": False,
        },
        {
            "sensitivity_axis": "noise", "aggregation_condition": "all_noise_seeds",
            "method": "TSVD", "criterion": "gcv", "status": "success",
            "selected_parameter": 4.0, "relative_reconstruction_error": 3.0,
            "unstable_solution": True,
        },
        {
            "sensitivity_axis": "noise", "aggregation_condition": "all_noise_seeds",
            "method": "TSVD", "criterion": "gcv", "status": "failed",
            "selected_parameter": float("nan"),
            "relative_reconstruction_error": float("nan"),
            "unstable_solution": True,
        },
    ]
    summaries = aggregate_records(records)
    selected = next(
        row for row in summaries if row["metric"] == "selected_parameter"
    )
    assert selected["n"] == 2
    assert selected["mean"] == pytest.approx(3.0)
    assert selected["standard_deviation"] == pytest.approx(np.sqrt(2.0))
    assert selected["median"] == pytest.approx(3.0)
    error = next(
        row for row in summaries if row["metric"] == "relative_reconstruction_error"
    )
    assert error["minimum"] == 1.0 and error["maximum"] == 3.0
    unstable = next(
        row for row in summaries if row["metric"] == "unstable_solution"
    )
    assert unstable["mean"] == pytest.approx(0.5)


def test_unstable_flag_has_explicit_threshold_and_nonfinite_handling():
    assert not classify_unstable_solution(2.0, 2.0, 1.0)
    assert classify_unstable_solution(10.0, 1.0, 1.0)
    assert classify_unstable_solution(1.0, 10.0, 1.0)
    assert classify_unstable_solution(float("inf"), 1.0, 1.0)


def test_parameter_selection_api_remains_ground_truth_free():
    assert "reference_image" not in inspect.signature(select_parameter).parameters
