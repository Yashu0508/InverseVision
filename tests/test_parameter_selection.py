import numpy as np
import pytest

from analysis.svd_analysis import analyze_svd
from regularization.tsvd import solve_tsvd
from regularization.parameter_selection import (
    discrete_lcurve_curvature,
    select_parameter,
    tikhonov_candidate_lambdas,
    tsvd_candidate_ranks,
)
from utils.phase4_protocol import create_phase4_problem


@pytest.fixture
def small_system():
    matrix = np.array([[1.0, 0.0], [0.0, 0.2]])
    observation = np.array([1.0, 0.25])
    return matrix, observation, analyze_svd(matrix, relative_tolerance=1e-12)


def test_lcurve_curvature_handles_endpoints_duplicates_and_degeneracy():
    curvature = discrete_lcurve_curvature(
        np.array([3.0, 2.0, 2.0, 1.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
        parameter_coordinates=np.array([0.0, 1.0, 1.0, 3.0]),
    )
    assert np.isnan(curvature[0]) and np.isnan(curvature[-1])
    assert np.all(np.isnan(curvature))
    assert np.all(np.isnan(discrete_lcurve_curvature([1.0, 1.0], [1.0, 1.0])))
    duplicate_points = discrete_lcurve_curvature(
        [4.0, 2.0, 2.0, 1.0], [1.0, 2.0, 2.0, 4.0]
    )
    assert np.isnan(duplicate_points[1])
    assert np.isnan(duplicate_points[2])


def test_gcv_tsvd_uses_trace_equal_to_retained_rank(small_system):
    matrix, observation, svd = small_system
    result = select_parameter(
        "TSVD", "gcv", matrix, observation, svd, tsvd_candidate_ranks(svd)
    )
    np.testing.assert_allclose(result.effective_degrees_of_freedom, [1, 2])
    assert np.isfinite(result.criterion_values[0])
    assert result.criterion_values[0] == pytest.approx(0.125)
    assert np.isinf(result.criterion_values[1])


def test_tikhonov_gcv_matches_explicit_hat_matrix_trace(small_system):
    matrix, observation, svd = small_system
    lambdas = np.array([0.1, 0.5, 2.0])
    result = select_parameter("Tikhonov", "gcv", matrix, observation, svd, lambdas)
    expected = []
    for lam in lambdas:
        hat = matrix @ np.linalg.solve(matrix.T @ matrix + lam**2 * np.eye(2), matrix.T)
        residual = np.linalg.norm(hat @ observation - observation)
        dof = np.trace(hat)
        expected.append((residual**2 / 2) / (1 - dof / 2) ** 2)
    np.testing.assert_allclose(result.effective_degrees_of_freedom, [
        np.trace(matrix @ np.linalg.solve(matrix.T @ matrix + lam**2 * np.eye(2), matrix.T))
        for lam in lambdas
    ])
    np.testing.assert_allclose(result.criterion_values, expected)


@pytest.mark.parametrize("method", ["TSVD", "Tikhonov"])
def test_discrepancy_selection_records_target_and_out_of_range_behavior(small_system, method):
    matrix, observation, svd = small_system
    parameters = np.array([1.0, 2.0]) if method == "TSVD" else np.array([0.1, 1.0])
    result = select_parameter(
        method, "discrepancy", matrix, observation, svd, parameters,
        noise_standard_deviation=100.0, discrepancy_factor=1.2,
    )
    assert "outside candidate residual range" in result.diagnostic
    expected = np.argmax(result.residual_norms)  # target exceeds every candidate
    assert result.selected_index == int(expected)


@pytest.mark.parametrize("method", ["TSVD", "Tikhonov"])
def test_lcurve_selects_interior_candidate(small_system, method):
    matrix = np.diag([1.0, 0.3, 0.1, 0.03])
    observation = np.array([1.0, 0.2, 0.15, 0.05])
    svd = analyze_svd(matrix, relative_tolerance=1e-12)
    parameters = tsvd_candidate_ranks(svd) if method == "TSVD" else np.geomspace(0.001, 10, 41)
    result = select_parameter(method, "l_curve", matrix, observation, svd, parameters)
    assert 0 < result.selected_index < parameters.size - 1
    assert np.isfinite(result.criterion_values[result.selected_index])


def test_candidate_grids_are_data_independent_and_documented(small_system):
    _, _, svd = small_system
    assert np.array_equal(tsvd_candidate_ranks(svd), [1.0, 2.0])
    lambdas = tikhonov_candidate_lambdas(svd, count=5)
    assert lambdas.size == 5
    assert np.all(np.diff(lambdas) > 0)
    assert lambdas[0] == pytest.approx(svd.singular_values[0] * 1e-6)


def test_tsvd_gcv_matches_explicit_smoother_and_solver_residual(small_system):
    matrix, observation, svd = small_system
    ranks = tsvd_candidate_ranks(svd)
    result = select_parameter("TSVD", "gcv", matrix, observation, svd, ranks)
    for index, rank_value in enumerate(ranks.astype(int)):
        rank = int(rank_value)
        u = svd.left_singular_vectors[:, :rank]
        v = svd.right_singular_vectors_transpose[:rank, :].T
        sigma = svd.singular_values[:rank]
        hat = matrix @ v @ np.diag(1.0 / sigma) @ u.T
        explicit_residual = np.linalg.norm(hat @ observation - observation)
        explicit_dof = np.trace(hat)
        explicit_num = explicit_residual**2 / observation.size
        explicit_den = 1.0 - explicit_dof / observation.size
        expected_gcv = explicit_num / explicit_den**2 if explicit_den else np.inf
        solved = solve_tsvd(matrix, observation, (1, 2), rank, svd_analysis=svd)
        actual_residual = np.linalg.norm(matrix @ solved.reconstruction_vector - observation)
        assert result.residual_norms[index] == pytest.approx(actual_residual, abs=1e-14)
        assert result.effective_degrees_of_freedom[index] == pytest.approx(explicit_dof)
        assert result.criterion_values[index] == pytest.approx(expected_gcv)


def test_numerical_rank_excludes_unusable_tsvd_directions():
    matrix = np.diag([1.0, 1e-14])
    observation = np.array([1.0, 1e-14])
    svd = analyze_svd(matrix, relative_tolerance=1e-12)
    assert svd.numerical_rank == 1
    assert np.array_equal(tsvd_candidate_ranks(svd), [1.0])
    with pytest.raises(ValueError, match="exceeds numerically usable rank"):
        solve_tsvd(matrix, observation, (1, 2), 2, svd_analysis=svd)


def test_discrepancy_selects_candidate_closest_to_bracketed_target(small_system):
    matrix, observation, svd = small_system
    parameters = tsvd_candidate_ranks(svd)
    target = 0.1
    result = select_parameter(
        "TSVD", "discrepancy", matrix, observation, svd, parameters,
        noise_standard_deviation=target / np.sqrt(observation.size),
        discrepancy_factor=1.0,
    )
    assert "outside candidate residual range" not in result.diagnostic
    assert result.selected_index == int(np.argmin(np.abs(result.residual_norms - target)))


def test_selection_api_does_not_accept_ground_truth(small_system):
    import inspect

    assert "reference_image" not in inspect.signature(select_parameter).parameters


def test_phase5_protocol_reproduces_rank142_gcv_and_forward_residual():
    problem = create_phase4_problem()
    matrix = problem.operator
    observation = problem.observation
    svd = problem.svd_analysis
    assert svd.numerical_rank == 144
    assert np.all(np.diff(svd.singular_values) <= 0.0)
    result = select_parameter(
        "TSVD", "gcv", matrix, observation, svd, tsvd_candidate_ranks(svd)
    )
    assert result.selected_parameter == 142.0
    rank_index = 141
    assert result.effective_degrees_of_freedom[rank_index] == 142.0
    assert result.residual_norms[rank_index] == pytest.approx(0.0013785170149, abs=1e-10)
    solution = solve_tsvd(
        matrix, observation, problem.original.shape, 142, svd_analysis=svd
    )
    forward_residual = np.linalg.norm(matrix @ solution.reconstruction_vector - observation)
    assert forward_residual == pytest.approx(result.residual_norms[rank_index], abs=1e-10)
    assert np.linalg.norm(
        matrix @ problem.original.ravel(order="C") - problem.degradation.blurred.ravel(order="C")
    ) < 1e-14
    repeat = create_phase4_problem()
    np.testing.assert_array_equal(observation, repeat.observation)
