from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = REPO_ROOT / "data" / "raw_data"
DEFAULT_DATASETS_DIR = REPO_ROOT / "data" / "datasets"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASETS_DIR
DEFAULT_SPLIT_RATIOS = (0.8, 0.1, 0.1)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DPI_SUFFIX_RE = re.compile(r"_\d+dpi$", re.IGNORECASE)


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    dataset_name: str
    image_path: Path
    annotation_path: Path


@dataclass(frozen=True)
class SplitResult:
    train: int
    val: int
    test: int
    total: int
    skipped_images_without_annotations: int
    skipped_annotations_without_images: int
    output_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create train/val/test splits from raw image and JSON annotation pairs. "
            "The default paths read from data/raw_data and write to a dated dataset folder in data/datasets."
        )
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Root directory containing raw dataset folders with images and JSON annotations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Parent directory where the dated dataset folder will be written.",
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[0])
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT_RATIOS[2])
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed used for deterministic shuffling."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty output directory.",
    )
    return parser.parse_args()


def validate_split_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float
) -> None:
    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    for name, ratio in ratios.items():
        if ratio < 0:
            raise ValueError(f"{name} must be >= 0. Received {ratio}.")

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("At least one split ratio must be greater than 0.")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Received train={train_ratio}, val={val_ratio}, test={test_ratio}, total={total}."
        )


def candidate_annotation_stems(image_path: Path) -> list[str]:
    stems = [image_path.stem]
    stem_without_dpi = DPI_SUFFIX_RE.sub("", image_path.stem)
    if stem_without_dpi != image_path.stem:
        stems.append(stem_without_dpi)
    return stems


def find_annotation_for_image(image_path: Path, raw_data_dir: Path | None = None) -> Path | None:
    exact_match = image_path.with_suffix(".json")
    if exact_match.exists():
        return exact_match

    zero_suffix_match = image_path.with_name(f"{image_path.stem}_0.json")
    if zero_suffix_match.exists():
        return zero_suffix_match

    wildcard_matches = sorted(image_path.parent.glob(f"{image_path.stem}_*.json"))
    if wildcard_matches:
        return wildcard_matches[0]

    ground_truth_dirs: list[Path] = []
    if raw_data_dir is not None:
        ground_truth_dirs.append(raw_data_dir / "ground_truths")
    if image_path.parent.name == "images":
        ground_truth_dirs.append(image_path.parent.parent / "ground_truths")

    checked_dirs: set[Path] = set()
    for ground_truth_dir in ground_truth_dirs:
        if ground_truth_dir in checked_dirs or not ground_truth_dir.is_dir():
            continue
        checked_dirs.add(ground_truth_dir)
        for stem in candidate_annotation_stems(image_path):
            for annotation_name in (f"{stem}.json", f"gt_{stem}.json"):
                annotation_path = ground_truth_dir / annotation_name
                if annotation_path.exists():
                    return annotation_path

    return None


def find_image_for_annotation(
    annotation_path: Path, raw_data_dir: Path | None = None
) -> Path | None:
    candidates = [
        annotation_path.with_suffix(extension) for extension in IMAGE_EXTENSIONS
    ]
    if annotation_path.stem.endswith("_0"):
        base_stem = annotation_path.stem[:-2]
        candidates.extend(
            annotation_path.with_name(f"{base_stem}{extension}")
            for extension in IMAGE_EXTENSIONS
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    image_dirs: list[Path] = []
    if raw_data_dir is not None:
        image_dirs.append(raw_data_dir / "images")
    if annotation_path.parent.name == "ground_truths":
        image_dirs.append(annotation_path.parent.parent / "images")

    base_stem = annotation_path.stem.removeprefix("gt_")
    checked_dirs: set[Path] = set()
    for image_dir in image_dirs:
        if image_dir in checked_dirs or not image_dir.is_dir():
            continue
        checked_dirs.add(image_dir)
        for extension in IMAGE_EXTENSIONS:
            for candidate in (
                image_dir / f"{base_stem}{extension}",
                image_dir / f"{base_stem}_360dpi{extension}",
            ):
                if candidate.exists():
                    return candidate
    return None


def make_sample_id(raw_data_dir: Path, image_path: Path) -> str:
    relative = image_path.relative_to(raw_data_dir).with_suffix("")
    return "__".join(relative.parts)


def collect_raw_samples(
    raw_data_dir: Path,
) -> tuple[list[DatasetSample], list[Path], list[Path]]:
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    samples: list[DatasetSample] = []
    images_without_annotations: list[Path] = []
    paired_annotation_paths: set[Path] = set()

    image_paths = sorted(
        path
        for path in raw_data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in image_paths:
        annotation_path = find_annotation_for_image(image_path, raw_data_dir)
        if annotation_path is None:
            images_without_annotations.append(image_path)
            continue

        dataset_name = image_path.relative_to(raw_data_dir).parts[0]
        samples.append(
            DatasetSample(
                sample_id=make_sample_id(raw_data_dir, image_path),
                dataset_name=dataset_name,
                image_path=image_path,
                annotation_path=annotation_path,
            )
        )
        paired_annotation_paths.add(annotation_path)

    annotations_without_images = [
        annotation_path
        for annotation_path in sorted(raw_data_dir.rglob("*.json"))
        if annotation_path not in paired_annotation_paths
        and find_image_for_annotation(annotation_path, raw_data_dir) is None
    ]

    return samples, images_without_annotations, annotations_without_images


def split_samples(
    samples: list[DatasetSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[DatasetSample]]:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[
            train_count + val_count : train_count + val_count + test_count
        ],
    }


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Pass overwrite=True or use --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def infer_dataset_output_name(raw_data_dir: Path) -> str:
    if raw_data_dir.name != DEFAULT_RAW_DATA_DIR.name or not raw_data_dir.is_dir():
        return raw_data_dir.name

    dataset_dirs = sorted(path for path in raw_data_dir.iterdir() if path.is_dir())
    if len(dataset_dirs) == 1:
        return dataset_dirs[0].name

    return raw_data_dir.name


def make_dataset_output_dir(
    raw_data_dir: Path, datasets_dir: Path, created_at: datetime | None = None
) -> Path:
    creation_date = (created_at or datetime.now()).strftime("%Y%m%d")
    dataset_name = f"{infer_dataset_output_name(raw_data_dir)}_{creation_date}"
    return datasets_dir / dataset_name


def copy_sample(
    sample: DatasetSample, split_name: str, raw_data_dir: Path, output_dir: Path
) -> dict[str, Any]:
    image_relative = sample.image_path.relative_to(raw_data_dir)
    annotation_relative = sample.annotation_path.relative_to(raw_data_dir)

    output_image = output_dir / split_name / "images" / image_relative
    output_annotation = output_dir / split_name / "annotations" / annotation_relative
    output_image.parent.mkdir(parents=True, exist_ok=True)
    output_annotation.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(sample.image_path, output_image)
    shutil.copy2(sample.annotation_path, output_annotation)

    return {
        "id": sample.sample_id,
        "dataset": sample.dataset_name,
        "image": str(output_image.relative_to(output_dir).as_posix()),
        "annotation": str(output_annotation.relative_to(output_dir).as_posix()),
        "source_image": str(sample.image_path.relative_to(raw_data_dir).as_posix()),
        "source_annotation": str(
            sample.annotation_path.relative_to(raw_data_dir).as_posix()
        ),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def convert_pdf_to_pngs(
    pdf_path: Path | str,
    output_dir: Path | str | None = None,
    dpi: int = 240,
    overwrite: bool = False,
    backend: str = "auto",
    pdftoppm_executable: str = "pdftoppm",
) -> list[Path]:
    """Render a PDF to PNG files at the requested DPI."""
    if dpi <= 0:
        raise ValueError(f"dpi must be a positive integer. Received {dpi}.")
    if backend not in {"auto", "pymupdf", "pdftoppm"}:
        raise ValueError(f"Unsupported PDF render backend: {backend}")

    pdf = Path(pdf_path)
    if not pdf.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf}")
    if pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, received: {pdf}")

    destination_dir = Path(output_dir) if output_dir is not None else pdf.parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    output_prefix = destination_dir / f"{pdf.stem}_{dpi}dpi"
    existing_outputs = sorted(destination_dir.glob(f"{output_prefix.name}*.png"))
    if existing_outputs:
        if not overwrite:
            raise FileExistsError(
                f"PNG output already exists for {pdf}: {existing_outputs[0]}. "
                "Pass overwrite=True to replace existing files."
            )
        for existing_output in existing_outputs:
            existing_output.unlink()

    if backend in {"auto", "pymupdf"}:
        try:
            return _convert_pdf_to_pngs_with_pymupdf(pdf, destination_dir, dpi)
        except ImportError:
            if backend == "pymupdf":
                raise RuntimeError(
                    "PyMuPDF is not installed. Install it with `pip install PyMuPDF` "
                    "or use backend='pdftoppm'."
                )

    if backend in {"auto", "pdftoppm"}:
        return _convert_pdf_to_pngs_with_pdftoppm(
            pdf=pdf,
            destination_dir=destination_dir,
            dpi=dpi,
            pdftoppm_executable=pdftoppm_executable,
        )

    raise RuntimeError("No PDF rendering backend is available.")


def _pdf_png_output_path(pdf: Path, output_dir: Path, dpi: int, page_count: int, page_index: int) -> Path:
    if page_count == 1:
        return output_dir / f"{pdf.stem}_{dpi}dpi.png"
    page_number = page_index + 1
    return output_dir / f"{pdf.stem}_page_{page_number}_{dpi}dpi.png"


def _convert_pdf_to_pngs_with_pymupdf(pdf: Path, output_dir: Path, dpi: int) -> list[Path]:
    import fitz

    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)
    rendered_outputs: list[Path] = []
    with fitz.open(pdf) as document:
        for page_index in range(document.page_count):
            output_path = _pdf_png_output_path(
                pdf=pdf,
                output_dir=output_dir,
                dpi=dpi,
                page_count=document.page_count,
                page_index=page_index,
            )
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            pixmap.save(output_path)
            rendered_outputs.append(output_path)
    return rendered_outputs


def _convert_pdf_to_pngs_with_pdftoppm(
    pdf: Path,
    destination_dir: Path,
    dpi: int,
    pdftoppm_executable: str,
) -> list[Path]:
    output_prefix = destination_dir / f"{pdf.stem}_{dpi}dpi"
    command = [
        pdftoppm_executable,
        "-r",
        str(dpi),
        "-png",
        str(pdf),
        str(output_prefix),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "pdftoppm was not found. Install Poppler or pass pdftoppm_executable "
            "with the full path to pdftoppm."
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Failed to render PDF to PNG: {pdf}. {details}") from exc

    rendered_outputs = sorted(destination_dir.glob(f"{output_prefix.name}-*.png"))
    if not rendered_outputs:
        single_output = output_prefix.with_suffix(".png")
        if single_output.exists():
            return [single_output]
        raise RuntimeError(f"pdftoppm did not create any PNG files for {pdf}.")

    if len(rendered_outputs) == 1:
        single_output = output_prefix.with_suffix(".png")
        rendered_outputs[0].replace(single_output)
        return [single_output]

    renamed_outputs: list[Path] = []
    for rendered_output in rendered_outputs:
        page_match = re.search(r"-(\d+)$", rendered_output.stem)
        page_number = page_match.group(1) if page_match else str(len(renamed_outputs) + 1)
        renamed_output = destination_dir / f"{pdf.stem}_page_{page_number}_{dpi}dpi.png"
        rendered_output.replace(renamed_output)
        renamed_outputs.append(renamed_output)
    return renamed_outputs


def create_train_val_test_dataset(
    raw_data_dir: Path | str = DEFAULT_RAW_DATA_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    train_ratio: float = DEFAULT_SPLIT_RATIOS[0],
    val_ratio: float = DEFAULT_SPLIT_RATIOS[1],
    test_ratio: float = DEFAULT_SPLIT_RATIOS[2],
    seed: int = 42,
    overwrite: bool = False,
) -> SplitResult:
    """Create train/val/test splits from raw image and JSON annotation pairs."""
    raw_data_path = Path(raw_data_dir)
    datasets_path = Path(output_dir)
    output_path = make_dataset_output_dir(raw_data_path, datasets_path)

    samples, missing_annotations, missing_images = collect_raw_samples(raw_data_path)
    if not samples:
        raise ValueError(f"No complete image/annotation pairs found in {raw_data_path}")

    ensure_output_dir(output_path, overwrite=overwrite)
    splits = split_samples(samples, train_ratio, val_ratio, test_ratio, seed)

    split_counts: dict[str, int] = {}
    for split_name, split_samples_list in splits.items():
        records = [
            copy_sample(sample, split_name, raw_data_path, output_path)
            for sample in split_samples_list
        ]
        write_jsonl(output_path / split_name / "metadata.jsonl", records)
        split_counts[split_name] = len(records)

    summary = {
        "raw_data_dir": str(raw_data_path),
        "output_dir": str(output_path),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "counts": {
            "total_complete_pairs": len(samples),
            "train": split_counts["train"],
            "val": split_counts["val"],
            "test": split_counts["test"],
            "skipped_images_without_annotations": len(missing_annotations),
            "skipped_annotations_without_images": len(missing_images),
        },
        "skipped": {
            "images_without_annotations": [
                str(path.relative_to(raw_data_path).as_posix())
                for path in missing_annotations
            ],
            "annotations_without_images": [
                str(path.relative_to(raw_data_path).as_posix())
                for path in missing_images
            ],
        },
    }
    with (output_path / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return SplitResult(
        train=split_counts["train"],
        val=split_counts["val"],
        test=split_counts["test"],
        total=len(samples),
        skipped_images_without_annotations=len(missing_annotations),
        skipped_annotations_without_images=len(missing_images),
        output_dir=str(output_path),
    )


def main() -> int:
    args = parse_args()
    result = create_train_val_test_dataset(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
