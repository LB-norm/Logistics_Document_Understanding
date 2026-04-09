from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Any

import matplotlib.pyplot as plt


def _as_result_list(results: Any) -> List[Any]:
    """
    Normalize a PP-StructureV3 prediction output into a list of Result objects.
    Accepts:
      - a single Result object
      - a list/tuple of Result objects
      - a generator already converted by list(...)
    """
    if results is None:
        return []

    if isinstance(results, (list, tuple)):
        return list(results)

    # single result object
    return [results]


def save_ppstructure_visualizations(
    results: Any,
    output_dir: str | Path,
) -> None:
    """
    Save PaddleOCR's built-in visualization images for each result object.

    Parameters
    ----------
    results:
        Output of PPStructureV3.predict(...), or one single result object.
    output_dir:
        Directory where PaddleOCR should save the visualization PNGs.

    Notes
    -----
    This uses the built-in `res.save_to_img(...)` method of PaddleOCR.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_list = _as_result_list(results)
    if not result_list:
        raise ValueError("No results were provided.")

    for res in result_list:
        res.save_to_img(save_path=str(output_dir))

