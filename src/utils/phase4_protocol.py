"""Fixed, explicitly selected Phase 4 experimental protocol."""

from utils.experiment_setup import (
    DeblurringExperimentProblem,
    create_camera_deblurring_problem,
)


PHASE4_IMAGE_SHAPE = (12, 12)
PHASE4_SIGMA = 1.5
PHASE4_KERNEL_SIZE = 9
PHASE4_NOISE_STANDARD_DEVIATION = 0.01
PHASE4_RANDOM_SEED = 42
PHASE4_RANK_RELATIVE_TOLERANCE = 1e-12
PHASE4_TSVD_RANKS = (5, 10, 15, 20, 30, 40, 60, 80, 100, 120, 144)
PHASE4_TIKHONOV_PARAMETERS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def create_phase4_problem() -> DeblurringExperimentProblem:
    """Create the one shared observation used by every Phase 4 experiment."""
    return create_camera_deblurring_problem(
        image_shape=PHASE4_IMAGE_SHAPE,
        sigma=PHASE4_SIGMA,
        kernel_size=PHASE4_KERNEL_SIZE,
        noise_standard_deviation=PHASE4_NOISE_STANDARD_DEVIATION,
        random_seed=PHASE4_RANDOM_SEED,
        rank_relative_tolerance=PHASE4_RANK_RELATIVE_TOLERANCE,
    )
