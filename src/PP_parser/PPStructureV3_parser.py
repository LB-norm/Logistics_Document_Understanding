from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = (
    REPO_ROOT
    / "data"
    / "datasets"
    / "raw_data_20260527"
    / "train"
    / "images"
    / "cmr_dachser"
    / "3f3fdb18-c151-43dd-b54a-da34249241f6_CMR_page_1.jpg"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "paddleocr_vl"
DEFAULT_VIS_DIR = DEFAULT_OUTPUT_DIR / "vis"


@dataclass(frozen=True)
class InferenceSummary:
    image_path: str
    output_dir: str
    visualization_dir: str | None
    result_count: int
    manifest_path: str
    device: str
    pipeline_version: str | None
    use_doc_preprocessor: bool
    use_layout_detection: bool
    merge_layout_blocks: bool
    min_pixels: int | None
    max_pixels: int | None
    max_new_tokens: int | None
    layout_detection_model_dir: str | None
    vl_rec_model_dir: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PaddleOCR-VL parser inference on a logistics document image and "
            "save JSON, Markdown, and visualization outputs."
        )
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to the input document image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PaddleOCR-VL JSON and Markdown results are saved.",
    )
    parser.add_argument(
        "--vis-dir",
        type=Path,
        default=DEFAULT_VIS_DIR,
        help="Directory where visualization images are saved.",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        help="PaddleOCR device string, for example 'gpu', 'cpu', or 'gpu:0'.",
    )
    parser.add_argument(
        "--pipeline-version",
        default="v1.5",
        help="PaddleOCR-VL pipeline version. Pass an empty value to use the package default.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Optional CPU thread count passed to PaddleOCR-VL.",
    )
    parser.add_argument(
        "--layout-detection-model-name",
        default=None,
        help="Optional Paddle layout detection model name.",
    )
    parser.add_argument(
        "--layout-detection-model-dir",
        type=Path,
        default=None,
        help="Optional local layout detection model directory for offline inference.",
    )
    parser.add_argument(
        "--vl-rec-model-name",
        default=None,
        help="Optional PaddleOCR-VL recognition model name.",
    )
    parser.add_argument(
        "--vl-rec-model-dir",
        type=Path,
        default=None,
        help="Optional local PaddleOCR-VL recognition model directory for offline inference.",
    )
    parser.add_argument(
        "--vl-rec-backend",
        default=None,
        help="Optional PaddleOCR-VL recognition backend.",
    )
    parser.add_argument(
        "--doc-orientation-classify-model-dir",
        type=Path,
        default=None,
        help="Optional local document orientation classifier model directory.",
    )
    parser.add_argument(
        "--doc-unwarping-model-dir",
        type=Path,
        default=None,
        help="Optional local document unwarping model directory.",
    )
    parser.add_argument(
        "--use-doc-orientation-classify",
        action="store_true",
        help="Enable document orientation classification.",
    )
    parser.add_argument(
        "--use-doc-unwarping",
        action="store_true",
        help="Enable document unwarping.",
    )
    parser.add_argument(
        "--no-layout-detection",
        action="store_true",
        help="Disable PaddleOCR-VL layout detection.",
    )
    parser.add_argument(
        "--merge-layout-blocks",
        action="store_true",
        help="Merge detected layout blocks before parsing.",
    )
    parser.add_argument(
        "--use-queues",
        action="store_true",
        help="Enable PaddleOCR internal queues.",
    )
    parser.add_argument(
        "--no-doc-preprocessor",
        action="store_true",
        help="Disable document preprocessing during prediction.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional maximum image pixel budget for generation.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Optional minimum image pixel budget for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional maximum generated token count.",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Do not call PaddleOCR's save_to_json method.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Do not call PaddleOCR's save_to_markdown method.",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Do not save visualization images.",
    )
    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Print PaddleOCR result objects to stdout.",
    )
    return parser.parse_args()


def load_paddleocr_vl() -> Any:
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: paddleocr. Install PaddleOCR with PaddleOCR-VL "
            "support before running this script."
        ) from exc
    return PaddleOCRVL


def normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def resolve_vis_dir(args: argparse.Namespace) -> Path | None:
    if args.no_visualization:
        return None
    return Path(args.vis_dir)


def build_pipeline_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "device": args.device,
        "use_layout_detection": not args.no_layout_detection,
        "merge_layout_blocks": args.merge_layout_blocks,
        "use_queues": args.use_queues,
    }

    pipeline_version = normalize_optional_string(args.pipeline_version)
    if pipeline_version is not None:
        kwargs["pipeline_version"] = pipeline_version
    if args.cpu_threads is not None:
        kwargs["cpu_threads"] = args.cpu_threads
    optional_values = {
        "layout_detection_model_name": args.layout_detection_model_name,
        "layout_detection_model_dir": args.layout_detection_model_dir,
        "vl_rec_model_name": args.vl_rec_model_name,
        "vl_rec_model_dir": args.vl_rec_model_dir,
        "vl_rec_backend": args.vl_rec_backend,
        "doc_orientation_classify_model_dir": args.doc_orientation_classify_model_dir,
        "doc_unwarping_model_dir": args.doc_unwarping_model_dir,
    }
    for key, value in optional_values.items():
        if value is not None:
            kwargs[key] = str(value) if isinstance(value, Path) else value

    if args.use_doc_orientation_classify:
        kwargs["use_doc_orientation_classify"] = True
    if args.use_doc_unwarping:
        kwargs["use_doc_unwarping"] = True

    return kwargs


def build_predict_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "use_doc_preprocessor": not args.no_doc_preprocessor,
    }
    if args.min_pixels is not None:
        kwargs["min_pixels"] = args.min_pixels
    if args.max_pixels is not None:
        kwargs["max_pixels"] = args.max_pixels
    if args.max_new_tokens is not None:
        kwargs["max_new_tokens"] = args.max_new_tokens
    return kwargs


def as_result_list(results: Any) -> list[Any]:
    if results is None:
        return []
    if isinstance(results, list):
        return results
    if isinstance(results, tuple):
        return list(results)
    try:
        return list(results)
    except TypeError:
        return [results]


def result_to_jsonable(result: Any) -> Any:
    result_json = getattr(result, "json", None)
    if callable(result_json):
        result_json = result_json()
    if result_json is not None:
        return result_json
    if isinstance(result, dict):
        return result
    return repr(result)


def save_outputs(
    results: list[Any],
    output_dir: Path,
    vis_dir: Path | None,
    save_json: bool,
    save_markdown: bool,
    print_results: bool,
) -> list[Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_results: list[Any] = []

    if vis_dir is not None:
        from PP_vis import save_ppstructure_visualizations

        save_ppstructure_visualizations(results, vis_dir)

    for result in results:
        if print_results:
            result.print()
        if save_json:
            result.save_to_json(save_path=str(output_dir))
        if save_markdown:
            result.save_to_markdown(save_path=str(output_dir))
        combined_results.append(result_to_jsonable(result))

    combined_path = output_dir / "combined_results.json"
    with combined_path.open("w", encoding="utf-8") as handle:
        json.dump(combined_results, handle, ensure_ascii=False, indent=2)

    return combined_results


def write_manifest(
    args: argparse.Namespace,
    results: list[Any],
    output_dir: Path,
    vis_dir: Path | None,
) -> InferenceSummary:
    summary = InferenceSummary(
        image_path=str(args.image_path),
        output_dir=str(output_dir),
        visualization_dir=str(vis_dir) if vis_dir is not None else None,
        result_count=len(results),
        manifest_path=str(output_dir / "inference_summary.json"),
        device=args.device,
        pipeline_version=normalize_optional_string(args.pipeline_version),
        use_doc_preprocessor=not args.no_doc_preprocessor,
        use_layout_detection=not args.no_layout_detection,
        merge_layout_blocks=args.merge_layout_blocks,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
        layout_detection_model_dir=(
            str(args.layout_detection_model_dir)
            if args.layout_detection_model_dir is not None
            else None
        ),
        vl_rec_model_dir=(
            str(args.vl_rec_model_dir) if args.vl_rec_model_dir is not None else None
        ),
    )

    with (output_dir / "inference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, ensure_ascii=False, indent=2)
    return summary


def run_inference(args: argparse.Namespace) -> InferenceSummary:
    image_path = args.image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input path is not a file: {image_path}")

    output_dir = args.output_dir.resolve()
    vis_dir = resolve_vis_dir(args)
    if vis_dir is not None:
        vis_dir = vis_dir.resolve()

    PaddleOCRVL = load_paddleocr_vl()
    pipeline = PaddleOCRVL(**build_pipeline_kwargs(args))
    results = as_result_list(
        pipeline.predict(str(image_path), **build_predict_kwargs(args))
    )
    if not results:
        raise RuntimeError("PaddleOCR-VL did not return any results.")

    save_outputs(
        results=results,
        output_dir=output_dir,
        vis_dir=vis_dir,
        save_json=not args.no_json,
        save_markdown=not args.no_markdown,
        print_results=args.print_results,
    )
    return write_manifest(args, results, output_dir, vis_dir)


def main() -> int:
    args = parse_args()
    summary = run_inference(args)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
