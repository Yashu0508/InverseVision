"""Run the Phase 1 degradation pipeline on scikit-image's camera image."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DEFAULT_CONFIG  # noqa: E402
from forward_model.degradation import degrade_image  # noqa: E402
from preprocessing.images import normalize_image  # noqa: E402
from visualization.degradation import plot_degradation  # noqa: E402


def _display_image(image: np.ndarray) -> Image.Image:
    """Create an 8-bit display copy without changing the numerical array."""
    clipped_copy = np.clip(image, 0.0, 1.0)
    pixels = np.rint(clipped_copy * 255.0).astype(np.uint8)
    return Image.fromarray(pixels, mode="L")


def main() -> None:
    """Generate and save a reproducible 64-by-64 degradation example."""
    camera = Image.fromarray(data.camera())
    target_height, target_width = DEFAULT_CONFIG.image_shape
    resized = camera.resize(
        (target_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )
    original = normalize_image(np.asarray(resized))
    result = degrade_image(original)

    original_path = DEFAULT_CONFIG.original_data_dir / "camera_64.png"
    blurred_path = DEFAULT_CONFIG.blurred_data_dir / "camera_64_blurred.png"
    noisy_path = DEFAULT_CONFIG.noisy_data_dir / "camera_64_noisy_display.png"
    noisy_raw_path = DEFAULT_CONFIG.noisy_data_dir / "camera_64_noisy_raw.npy"
    plot_path = DEFAULT_CONFIG.output_directories["plots"] / "phase1_degradation.png"

    _display_image(result.original).save(original_path)
    _display_image(result.blurred).save(blurred_path)
    _display_image(result.noisy_observation).save(noisy_path)
    np.save(noisy_raw_path, result.noisy_observation)

    figure, _ = plot_degradation(
        result.original,
        result.blurred,
        result.noisy_observation,
        show=False,
    )
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)

    print(f"Original: {original_path}")
    print(f"Blurred: {blurred_path}")
    print(f"Noisy display copy: {noisy_path}")
    print(f"Raw noisy array: {noisy_raw_path}")
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
