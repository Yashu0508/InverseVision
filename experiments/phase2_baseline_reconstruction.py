"""Compare Phase 2 unregularized dense reconstruction baselines."""

import csv
from pathlib import Path
import sys
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG  # noqa: E402
from evaluation.reconstruction_metrics import evaluate_reconstruction  # noqa: E402
from forward_model.degradation import degrade_image  # noqa: E402
from forward_model.gaussian import gaussian_blur_operator_2d  # noqa: E402
from numerical_methods.baseline_solvers import (  # noqa: E402
    MAX_DENSE_PIXELS,
    solve_direct_inverse,
    solve_normal_equations,
    solve_qr_least_squares,
)
from preprocessing.images import normalize_image  # noqa: E402
from visualization.reconstruction import (  # noqa: E402
    plot_metric_comparison,
    plot_reconstruction_comparison,
)


EXPERIMENT_SHAPE = (12, 12)
CSV_FIELDS = (
    "solver",
    "status",
    "diagnostic",
    "execution_time_seconds",
    "relative_reconstruction_error",
    "mean_squared_error",
    "peak_signal_to_noise_ratio",
    "structural_similarity_index",
    "residual_norm",
    "relative_residual",
)


def _experiment_image() -> np.ndarray:
    camera = Image.fromarray(data.camera())
    rows, columns = EXPERIMENT_SHAPE
    resized = camera.resize((columns, rows), resample=Image.Resampling.LANCZOS)
    return normalize_image(np.asarray(resized))


def main() -> None:
    """Run all baseline solvers on one shared degradation experiment."""
    if int(np.prod(EXPERIMENT_SHAPE)) > MAX_DENSE_PIXELS:
        raise ValueError("experiment exceeds the documented dense-solver limit")

    original = _experiment_image()
    degradation = degrade_image(
        original,
        sigma=DEFAULT_CONFIG.gaussian_blur_sigma,
        kernel_size=DEFAULT_CONFIG.gaussian_kernel_size,
        noise_standard_deviation=DEFAULT_CONFIG.noise_standard_deviation,
        seed=DEFAULT_CONFIG.random_seed,
    )
    operator = gaussian_blur_operator_2d(
        EXPERIMENT_SHAPE,
        sigma=DEFAULT_CONFIG.gaussian_blur_sigma,
        kernel_size=DEFAULT_CONFIG.gaussian_kernel_size,
    )
    observation = degradation.noisy_observation.ravel(order="C")

    methods = (
        ("Direct inverse", solve_direct_inverse),
        ("QR least squares", solve_qr_least_squares),
        ("Normal equations", solve_normal_equations),
    )
    records: list[dict[str, object]] = []
    reconstructions: dict[str, np.ndarray | None] = {}
    failures: dict[str, str] = {}

    for display_name, solver in methods:
        attempt_start = perf_counter()
        try:
            result = solver(operator, observation, EXPERIMENT_SHAPE)
            metrics = evaluate_reconstruction(
                original,
                result.reconstruction_image,
                operator,
                observation,
            )
            record: dict[str, object] = {
                "solver": display_name,
                "status": result.status,
                "diagnostic": result.diagnostic,
                "execution_time_seconds": result.execution_time_seconds,
                "relative_reconstruction_error": (
                    metrics.relative_reconstruction_error
                ),
                "mean_squared_error": metrics.mean_squared_error,
                "peak_signal_to_noise_ratio": (
                    metrics.peak_signal_to_noise_ratio
                ),
                "structural_similarity_index": (
                    metrics.structural_similarity_index
                ),
                "residual_norm": metrics.residual_norm,
                "relative_residual": metrics.relative_residual,
            }
            reconstructions[display_name] = result.reconstruction_image
            raw_path = (
                DEFAULT_CONFIG.output_directories["images"]
                / f"phase2_{result.solver_name}_raw.npy"
            )
            np.save(raw_path, result.reconstruction_image)
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            message = str(error)
            record = {
                "solver": display_name,
                "status": "failed",
                "diagnostic": message,
                "execution_time_seconds": perf_counter() - attempt_start,
                "relative_reconstruction_error": "",
                "mean_squared_error": "",
                "peak_signal_to_noise_ratio": "",
                "structural_similarity_index": "",
                "residual_norm": "",
                "relative_residual": "",
            }
            reconstructions[display_name] = None
            failures[display_name] = message
        records.append(record)

    metrics_path = (
        DEFAULT_CONFIG.output_directories["metrics"]
        / "phase2_baseline_comparison.csv"
    )
    with metrics_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    reconstruction_figure, _ = plot_reconstruction_comparison(
        original,
        degradation.noisy_observation,
        reconstructions,
        failures=failures,
        show=False,
    )
    reconstruction_path = (
        DEFAULT_CONFIG.output_directories["plots"]
        / "phase2_reconstructions.png"
    )
    reconstruction_figure.savefig(reconstruction_path, dpi=150)
    plt.close(reconstruction_figure)

    metric_figure, _ = plot_metric_comparison(records, show=False)
    metric_path = (
        DEFAULT_CONFIG.output_directories["plots"] / "phase2_metrics.png"
    )
    metric_figure.savefig(metric_path, dpi=150)
    plt.close(metric_figure)

    print(f"Experiment shape: {EXPERIMENT_SHAPE}")
    print(f"Dense solver limit: {MAX_DENSE_PIXELS} pixels")
    print(f"Metrics: {metrics_path}")
    print(f"Reconstructions: {reconstruction_path}")
    print(f"Metric plot: {metric_path}")
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
