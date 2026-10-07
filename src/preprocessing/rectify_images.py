"""Save rectified document images from a single file or a folder."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

import cv2

from src.preprocessing.document_rectification import (
    OrientationDetector,
    RectificationConfig,
    RectificationResult,
    rectify_document,
    tesseract_orientation,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _sidecar(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".json")


def _check_output(source: Path, output: Path, overwrite: bool) -> None:
    if source.resolve() in (output.resolve(), _sidecar(output).resolve()):
        raise ValueError("Use a separate output path to preserve the source image")
    for path in (output, _sidecar(output)):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {path}; use overwrite=True or --overwrite")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _metadata(source: Path, output: Path, input_size: tuple[int, int],
              result: RectificationResult) -> dict[str, Any]:
    return {
        "source_path": str(source.resolve()), "output_path": str(output.resolve()),
        "input_size": list(input_size), "output_size": list(result.image.shape[1::-1]),
        "transform": result.transform.tolist(),
        "page_quad": result.page_quad.tolist() if result.page_quad is not None else None,
        "deskew_degrees": result.deskew_degrees,
        "clockwise_rotation": result.clockwise_rotation,
        "warnings": list(result.warnings),
    }


def rectify_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    config: RectificationConfig | None = None,
    orientation_detector: OrientationDetector | None = None,
    clockwise_rotation: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Save one image and its transformation sidecar, returning the metadata."""
    source, output = Path(input_path), Path(output_path)
    _check_output(source, output, overwrite)
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"Cannot read image: {source}")
    result = rectify_document(image, config=config, orientation_detector=orientation_detector,
                              clockwise_rotation=clockwise_rotation)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result.image):
        raise OSError(f"Cannot write image: {output}")
    metadata = _metadata(source, output, image.shape[1::-1], result)
    _write_json(_sidecar(output), metadata)
    return metadata


def rectify_folder(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    recursive: bool = False,
    config: RectificationConfig | None = None,
    orientation_detector: OrientationDetector | None = None,
    clockwise_rotation: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Rectify a folder to lossless PNGs using the same in-memory pipeline.

    Preserve relative folders and image stems. Reject filename collisions and
    existing outputs before writing. Continue after individual image failures
    and write manifest.json with successes, warnings, and failures. Directory
    trees must be separate so repeated discovery never includes prior outputs.
    """
    source_root, output_root = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source_root.is_dir():
        raise NotADirectoryError(source_root)
    if source_root.is_relative_to(output_root) or output_root.is_relative_to(source_root):
        raise ValueError("Input and output folders must be separate, non-overlapping directory trees")
    iterator = source_root.rglob("*") if recursive else source_root.iterdir()
    sources = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not sources:
        raise ValueError(f"No supported images found in {source_root}")
    targets: dict[Path, Path] = {}
    for source in sources:
        target = (output_root / source.relative_to(source_root)).with_suffix(".png")
        if target in targets:
            raise ValueError(f"Output filename collision: {targets[target]} and {source} both map to {target}")
        targets[target] = source
        _check_output(source, target, overwrite)
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {manifest_path}; use overwrite=True or --overwrite")
    records = []
    for target, source in targets.items():
        try:
            metadata = rectify_file(source, target, config=config,
                                    orientation_detector=orientation_detector,
                                    clockwise_rotation=clockwise_rotation, overwrite=overwrite)
            records.append({"status": "ok", **metadata})
        except Exception as exc:
            records.append({"status": "error", "source_path": str(source),
                            "output_path": str(target), "error": f"{type(exc).__name__}: {exc}"})
    succeeded = sum(record["status"] == "ok" for record in records)
    manifest = {
        "format_version": 1, "input_dir": str(source_root), "output_dir": str(output_root),
        "recursive": recursive, "config": asdict(config or RectificationConfig()),
        "clockwise_rotation": clockwise_rotation,
        "total": len(records), "succeeded": succeeded, "failed": len(records) - succeeded,
        "records": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="An image file or a folder of images")
    parser.add_argument("output", type=Path, help="Output image path, or output folder for folder input")
    parser.add_argument("--recursive", action="store_true", help="Include subfolders and preserve their structure")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs and metadata")
    parser.add_argument("--orientation", choices=("auto", "skip", "0", "90", "180", "270"), default="auto",
                        help="Clockwise correction, or optional Tesseract OSD (auto)")
    parser.add_argument("--no-page-detection", action="store_true")
    parser.add_argument("--no-deskew", action="store_true")
    args = parser.parse_args(argv)
    settings = {
        "config": RectificationConfig(detect_page=not args.no_page_detection, deskew=not args.no_deskew),
        "orientation_detector": tesseract_orientation if args.orientation == "auto" else None,
        "clockwise_rotation": int(args.orientation) if args.orientation.isdigit() else None,
        "overwrite": args.overwrite,
    }
    try:
        if args.input.is_dir():
            manifest = rectify_folder(args.input, args.output, recursive=args.recursive, **settings)
            for record in manifest["records"]:
                print(f"{record['status']}: {record['source_path']} -> {record['output_path']}")
                if record["status"] == "error":
                    print(f"  {record['error']}")
            print(f"Saved {manifest['succeeded']}/{manifest['total']} images; "
                  f"{manifest['failed']} failed. Manifest: {args.output / 'manifest.json'}")
            return int(manifest["failed"] > 0)
        if args.recursive:
            parser.error("--recursive requires a folder input")
        metadata = rectify_file(args.input, args.output, **settings)
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
