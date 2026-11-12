"""Centralized paths and numerical defaults for the deblurring project."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable default settings shared by future project modules."""

    image_shape: tuple[int, int] = (64, 64)
    gaussian_blur_sigma: float = 1.5
    gaussian_kernel_size: int = 9
    noise_standard_deviation: float = 0.01
    random_seed: int = 42
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-6

    project_root: Path = PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        """Return the root directory for input data."""
        return self.project_root / "data"

    @property
    def original_data_dir(self) -> Path:
        """Return the directory for original images."""
        return self.data_dir / "original"

    @property
    def blurred_data_dir(self) -> Path:
        """Return the directory for blurred images."""
        return self.data_dir / "blurred"

    @property
    def noisy_data_dir(self) -> Path:
        """Return the directory for noisy images."""
        return self.data_dir / "noisy"

    @property
    def results_dir(self) -> Path:
        """Return the root directory for generated results."""
        return self.project_root / "results"

    @property
    def output_directories(self) -> dict[str, Path]:
        """Return named output directories for generated artifacts."""
        return {
            "images": self.results_dir / "images",
            "plots": self.results_dir / "plots",
            "metrics": self.results_dir / "metrics",
            "logs": self.results_dir / "logs",
        }


DEFAULT_CONFIG = ProjectConfig()
