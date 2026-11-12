"""Controlled, dense-safe experimental cases for Phase 6."""

from dataclasses import dataclass
import math

from analysis.svd_analysis import MAX_FULL_SVD_DIMENSION


PHASE6_KERNEL_SIZE = 9
PHASE6_NOISE_STANDARD_DEVIATION = 0.01
PHASE6_REFERENCE_SIGMA = 1.5
PHASE6_REFERENCE_SHAPE = (12, 12)
PHASE6_RANK_RELATIVE_TOLERANCE = 1e-12
PHASE6_DISCREPANCY_FACTOR = 1.05
PHASE6_NOISE_SEEDS = (7, 21, 42, 84, 126)
PHASE6_BLUR_STRENGTHS = (0.8, 1.5, 2.2)
PHASE6_IMAGE_SHAPES = ((6, 6), (12, 12), (16, 16))


@dataclass(frozen=True)
class Phase6Case:
    """One fully specified reproducible forward-problem realization."""

    case_id: str
    sensitivity_axis: str
    condition_value: str
    image_shape: tuple[int, int]
    sigma: float
    noise_standard_deviation: float
    random_seed: int
    kernel_size: int = PHASE6_KERNEL_SIZE
    boundary: str = "zero"

    def __post_init__(self) -> None:
        if not self.case_id or not self.sensitivity_axis or not self.condition_value:
            raise ValueError("case identifiers and labels must be non-empty")
        if (
            not isinstance(self.image_shape, tuple)
            or len(self.image_shape) != 2
            or any(not isinstance(size, int) or size <= 0 for size in self.image_shape)
        ):
            raise ValueError("image_shape must contain two positive integers")
        pixel_count = self.image_shape[0] * self.image_shape[1]
        if pixel_count > MAX_FULL_SVD_DIMENSION:
            raise ValueError(
                f"exact dense SVD limit is {MAX_FULL_SVD_DIMENSION}; "
                f"case has {pixel_count} pixels"
            )
        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and positive")
        if (
            not math.isfinite(self.noise_standard_deviation)
            or self.noise_standard_deviation < 0.0
        ):
            raise ValueError(
                "noise_standard_deviation must be finite and non-negative"
            )
        if not isinstance(self.random_seed, int) or isinstance(self.random_seed, bool):
            raise TypeError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if (
            not isinstance(self.kernel_size, int)
            or isinstance(self.kernel_size, bool)
            or self.kernel_size <= 0
            or self.kernel_size % 2 == 0
        ):
            raise ValueError("kernel_size must be a positive odd integer")
        if self.boundary != "zero":
            raise ValueError("Phase 6 preserves the zero-boundary convention")


def build_phase6_cases() -> tuple[Phase6Case, ...]:
    """Build noise, blur, and resolution groups with a shared reference case.

    The 12x12, sigma=1.5, seed=42 case is repeated in each group to make the
    controlled factor for each sensitivity axis explicit. All configurations
    remain inside the established exact dense SVD dimension limit.
    """
    cases: list[Phase6Case] = []
    for seed in PHASE6_NOISE_SEEDS:
        cases.append(Phase6Case(
            case_id=f"noise_seed_{seed}", sensitivity_axis="noise",
            condition_value="all_noise_seeds", image_shape=PHASE6_REFERENCE_SHAPE,
            sigma=PHASE6_REFERENCE_SIGMA,
            noise_standard_deviation=PHASE6_NOISE_STANDARD_DEVIATION,
            random_seed=seed,
        ))
    for sigma in PHASE6_BLUR_STRENGTHS:
        cases.append(Phase6Case(
            case_id=f"blur_sigma_{sigma:g}", sensitivity_axis="blur",
            condition_value=f"sigma_{sigma:g}", image_shape=PHASE6_REFERENCE_SHAPE,
            sigma=sigma, noise_standard_deviation=PHASE6_NOISE_STANDARD_DEVIATION,
            random_seed=42,
        ))
    for shape in PHASE6_IMAGE_SHAPES:
        cases.append(Phase6Case(
            case_id=f"resolution_{shape[0]}x{shape[1]}",
            sensitivity_axis="resolution",
            condition_value=f"resolution_{shape[0]}x{shape[1]}",
            image_shape=shape, sigma=PHASE6_REFERENCE_SIGMA,
            noise_standard_deviation=PHASE6_NOISE_STANDARD_DEVIATION,
            random_seed=42,
        ))
    return tuple(cases)
