"""Validation tests for the Phase 0 project setup."""

from pathlib import Path

from config import DEFAULT_CONFIG, ProjectConfig


def test_configuration_imports_and_has_expected_defaults() -> None:
    """The central configuration imports and exposes the baseline settings."""
    assert isinstance(DEFAULT_CONFIG, ProjectConfig)
    assert DEFAULT_CONFIG.image_shape == (64, 64)


def test_configured_directories_resolve_to_project_paths() -> None:
    """All configured data and output directories resolve correctly."""
    configured_paths = [
        DEFAULT_CONFIG.data_dir,
        DEFAULT_CONFIG.original_data_dir,
        DEFAULT_CONFIG.blurred_data_dir,
        DEFAULT_CONFIG.noisy_data_dir,
        *DEFAULT_CONFIG.output_directories.values(),
    ]

    assert all(
        isinstance(path, Path) and path.is_absolute()
        for path in configured_paths
    )
    assert all(path.exists() for path in configured_paths)


def test_expected_phase_zero_structure_exists() -> None:
    """The required top-level directories are available in the repository."""
    required_directories = [
        "data",
        "src",
        "experiments",
        "results",
        "tests",
        "report",
        "docs",
    ]

    assert all((DEFAULT_CONFIG.project_root / name).is_dir()
               for name in required_directories)
