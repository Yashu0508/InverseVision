import numpy as np
import pytest
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, aslinearoperator

from forward_model.gaussian import gaussian_blur_operator_2d
from forward_model.operators import gaussian_blur_linear_operator
from numerical_methods.iterative_solvers import (
    solve_cgls,
    solve_lsqr,
    solve_lsqr_checkpoints,
)
from regularization.tikhonov import solve_tikhonov_svd
from utils.experiment_setup import create_camera_deblurring_problem
from utils.phase7_protocol import (
    create_matrix_free_problem,
    structured_operator_condition_number,
)


def test_lsqr_matches_small_dense_least_squares():
    matrix = np.array([[1.0, 0.2], [0.1, 2.0], [0.4, -0.3]])
    observation = np.array([1.0, -0.5, 0.7])
    result = solve_lsqr(
        aslinearoperator(matrix), observation,
        iteration_limit=20, relative_tolerance=1e-13,
        absolute_tolerance=1e-13,
    )
    reference = linalg.lstsq(matrix, observation)[0]
    np.testing.assert_allclose(result.reconstruction_vector, reference, atol=1e-11)
    assert result.residual_norm == pytest.approx(
        np.linalg.norm(matrix @ reference - observation), abs=1e-12
    )
    assert result.normal_equation_residual_norm < 1e-10
    expected_normal = np.linalg.norm(matrix.T @ (matrix @ result.reconstruction_vector - observation))
    assert result.normal_equation_residual_norm == pytest.approx(expected_normal, abs=1e-12)


def test_cgls_matches_small_dense_least_squares_and_tracks_iterates():
    matrix = np.array([[1.0, 0.2], [0.1, 2.0], [0.4, -0.3]])
    observation = np.array([1.0, -0.5, 0.7])
    result = solve_cgls(
        aslinearoperator(matrix), observation,
        iteration_limit=20, relative_tolerance=1e-12,
    )
    reference = linalg.lstsq(matrix, observation)[0]
    np.testing.assert_allclose(result.reconstruction_vector, reference, atol=1e-10)
    assert result.iteration_count <= 2
    assert len(result.history) == len(result.reconstruction_history)
    assert result.history[-1].residual_norm >= 0.0
    assert result.normal_equation_residual_norm < 1e-9
    expected_normal = np.linalg.norm(
        matrix.T @ (matrix @ result.reconstruction_vector - observation)
    )
    assert result.normal_equation_residual_norm == pytest.approx(expected_normal)


def test_damped_lsqr_matches_existing_tikhonov_lambda_convention():
    matrix = np.array([[1.0, 0.2], [0.1, 2.0], [0.4, -0.3]])
    observation = np.array([1.0, -0.5, 0.7])
    damping = 0.15
    result = solve_lsqr(
        aslinearoperator(matrix), observation,
        iteration_limit=20, relative_tolerance=1e-13,
        absolute_tolerance=1e-13, damping_parameter=damping,
    )
    reference = np.linalg.solve(
        matrix.T @ matrix + damping**2 * np.eye(matrix.shape[1]),
        matrix.T @ observation,
    )
    np.testing.assert_allclose(result.reconstruction_vector, reference, atol=1e-11)
    assert result.damping_parameter == damping
    assert result.method == "damped_LSQR"


def test_damped_lsqr_matches_existing_tikhonov_svd_reference():
    problem = create_camera_deblurring_problem(
        image_shape=(6, 6), sigma=0.8, kernel_size=5,
        noise_standard_deviation=0.01, random_seed=42,
        rank_relative_tolerance=1e-12,
    )
    matrix = problem.operator
    operator = gaussian_blur_linear_operator((6, 6), sigma=0.8, kernel_size=5)
    damping = 0.05
    iterative = solve_lsqr(
        operator, problem.observation, iteration_limit=100,
        relative_tolerance=1e-12, absolute_tolerance=1e-12,
        damping_parameter=damping, condition_limit=1e14,
    )
    reference = solve_tikhonov_svd(
        matrix, problem.observation, (6, 6), damping,
        relative_tolerance=1e-12, svd_analysis=problem.svd_analysis,
    )
    np.testing.assert_allclose(
        iterative.reconstruction_vector,
        reference.reconstruction_vector,
        rtol=2e-7,
        atol=2e-8,
    )


@pytest.mark.parametrize("shape", [(5, 7), (16, 16)])
def test_gaussian_linear_operator_forward_adjoint_identity(shape):
    operator = gaussian_blur_linear_operator(shape, sigma=1.1, kernel_size=7)
    rng = np.random.default_rng(314)
    x = rng.normal(size=shape[0] * shape[1])
    y = rng.normal(size=shape[0] * shape[1])
    lhs = float(np.dot(operator.matvec(x), y))
    rhs = float(np.dot(x, operator.rmatvec(y)))
    assert lhs == pytest.approx(rhs, rel=2e-13, abs=2e-13)


def test_iterative_solver_requires_matrix_free_linear_operator():
    with pytest.raises(TypeError, match="LinearOperator"):
        solve_lsqr(np.eye(2), np.ones(2))
    with pytest.raises(ValueError, match="iteration_limit"):
        solve_cgls(aslinearoperator(np.eye(2)), np.ones(2), iteration_limit=0)


def test_cgls_stopping_limits_and_lsqr_checkpoints_are_explicit():
    matrix = np.diag([1.0, 0.3, 0.05])
    observation = np.array([1.0, 0.2, 0.15])
    short = solve_cgls(
        aslinearoperator(matrix), observation,
        iteration_limit=1, relative_tolerance=0.0,
    )
    long = solve_cgls(
        aslinearoperator(matrix), observation,
        iteration_limit=3, relative_tolerance=0.0,
    )
    assert short.iteration_count == 1
    assert long.iteration_count <= 3
    checkpoints = solve_lsqr_checkpoints(
        aslinearoperator(matrix), observation, (1, 2, 3),
        relative_tolerance=1e-14, absolute_tolerance=1e-14,
    )
    assert [item.iteration_limit for item in checkpoints] == [1, 2, 3]
    assert all(item.iteration_count <= item.iteration_limit for item in checkpoints)
    with pytest.raises(ValueError, match="strictly increasing"):
        solve_lsqr_checkpoints(
            aslinearoperator(matrix), observation, (4, 2)
        )


def test_large_operator_path_never_materializes_full_image_matrix(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("dense SVD must not be called by this test")

    monkeypatch.setattr("analysis.svd_analysis.analyze_svd", forbidden)
    operator = gaussian_blur_linear_operator((32, 32), sigma=1.5, kernel_size=9)
    assert operator.shape == (1024, 1024)
    result = solve_lsqr(
        operator, np.ones(1024), iteration_limit=2,
        relative_tolerance=0.0, absolute_tolerance=0.0,
    )
    assert result.reconstruction_vector.shape == (1024,)
    assert not hasattr(operator, "toarray")


def test_matrix_free_problem_is_reproducible_and_uses_operator_observation():
    arguments = {
        "image_shape": (8, 8),
        "sigma": 1.2,
        "kernel_size": 7,
        "noise_standard_deviation": 0.01,
        "random_seed": 42,
    }
    first = create_matrix_free_problem(**arguments)
    second = create_matrix_free_problem(**arguments)
    np.testing.assert_array_equal(first.noisy_observation, second.noisy_observation)
    np.testing.assert_array_equal(first.noise, second.noise)
    expected_blur = first.operator.matvec(first.original.ravel(order="C"))
    np.testing.assert_allclose(expected_blur, first.blurred.ravel(order="C"), atol=0.0)
    np.testing.assert_array_equal(
        first.observation, first.noisy_observation.ravel(order="C")
    )


def test_structured_condition_number_matches_small_explicit_operator():
    shape = (5, 7)
    sigma = 1.1
    kernel_size = 7
    expected = np.linalg.cond(
        gaussian_blur_operator_2d(
            shape, sigma=sigma, kernel_size=kernel_size
        ).toarray()
    )
    actual = structured_operator_condition_number(
        shape, sigma=sigma, kernel_size=kernel_size
    )
    assert actual == pytest.approx(expected, rel=3e-12)


def test_lsqr_rejects_mismatched_observation_and_invalid_damping():
    operator = aslinearoperator(np.eye(3))
    with pytest.raises(ValueError, match="matching operator rows"):
        solve_lsqr(operator, np.ones(2))
    with pytest.raises(ValueError, match="damping_parameter"):
        solve_lsqr(operator, np.ones(3), damping_parameter=-0.1)
    with pytest.raises(TypeError, match="complex observations"):
        solve_lsqr(operator, np.ones(3, dtype=complex))


def test_iterative_solvers_require_a_usable_adjoint():
    operator = LinearOperator((2, 2), matvec=lambda vector: vector)
    with pytest.raises(ValueError, match="adjoint rmatvec"):
        solve_lsqr(operator, np.ones(2))
