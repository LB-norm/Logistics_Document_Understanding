"""Shared, project-wide image-resolution presets.

The public interface intentionally exposes named categories rather than arbitrary
pixel budgets. Model-specific integrations can still pass the resolved maximum
pixel count to their underlying processors.
"""

from __future__ import annotations

from typing import Final, Literal, cast


ImageResolution = Literal["low", "medium", "high", "native"]

DEFAULT_IMAGE_RESOLUTION: Final[ImageResolution] = "medium"
IMAGE_RESOLUTION_PIXELS: Final[dict[ImageResolution, int]] = {
    "low": 1_400_000,
    "medium": 2_800_000,
    "high": 4_200_000,
    "native": 5_600_000,
}
IMAGE_RESOLUTION_CHOICES: Final[tuple[ImageResolution, ...]] = tuple(
    IMAGE_RESOLUTION_PIXELS
)


def max_pixels_for_resolution(resolution: str) -> int:
    """Resolve one supported resolution category to its maximum pixel budget."""
    if resolution not in IMAGE_RESOLUTION_PIXELS:
        choices = ", ".join(IMAGE_RESOLUTION_CHOICES)
        raise ValueError(
            f"Unsupported resolution {resolution!r}; choose one of: {choices}."
        )
    return IMAGE_RESOLUTION_PIXELS[cast(ImageResolution, resolution)]
