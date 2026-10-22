"""Compose existing inference functions while inheriting their native defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from src.Qwen import run_inference as qwen
from src.Donut import run_donut_inference as donut
from src.PP_parser import OCRConfig, OCRExtractor
from src.fusion import FusionConfig, FusionPipeline
from src.eval_suite.schema import validate_json_schema
from src.utils.pipeline_tracking import PipelineTracker, attach_framework_peaks


@dataclass(frozen=True)
class PipelineConfig:
    mode: str
    model_type: str = "qwen"
    model_id: str | None = None
    adapter_path: Path | None = None
    fusion_mode: int | str | None = None
    yolo_model_dir: Path | None = None
    yolo_device: str | None = None
    yolo_threshold: float | None = None
    ocr_config: OCRConfig = field(default_factory=OCRConfig)
    min_overlap: float = 0.1
    model_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"end-to-end", "parser", "hybrid"}:
            raise ValueError("mode must be end-to-end, parser, or hybrid")
        if self.model_type not in {"qwen", "donut"}:
            raise ValueError("model_type must be qwen or donut")
        if not self.model_id and self.adapter_path is None:
            raise ValueError("Choose a model_id or a Qwen adapter_path")
        if self.adapter_path is not None:
            object.__setattr__(self, "adapter_path", Path(self.adapter_path))
        if self.yolo_model_dir is not None:
            object.__setattr__(self, "yolo_model_dir", Path(self.yolo_model_dir))
        if self.mode in {"parser", "hybrid"}:
            if self.model_type != "qwen":
                raise ValueError("Parser/hybrid mode currently supports Qwen")
            if self.mode == "hybrid" and self.fusion_mode is None:
                object.__setattr__(self, "fusion_mode", "hard")
            if self.fusion_mode is None or (self.mode == "parser" and self.yolo_model_dir is None):
                raise ValueError("Parser mode requires fusion_mode and yolo_model_dir")
            FusionConfig(self.fusion_mode, self.min_overlap)
            if self.mode == "hybrid" and FusionConfig(self.fusion_mode, self.min_overlap).mode != 2:
                raise ValueError("Hybrid mode requires hard fusion")
        elif self.fusion_mode is not None or self.yolo_model_dir is not None:
            raise ValueError("Fusion and YOLO options require parser or hybrid mode")
        if self.model_type == "donut" and self.adapter_path is not None:
            raise ValueError("Donut takes a complete model checkpoint, not a Qwen adapter")
        if self.yolo_threshold is not None and not 0 <= self.yolo_threshold <= 1:
            raise ValueError("yolo_threshold must be between 0 and 1")


def _native_args(config: PipelineConfig) -> Any:
    # Input/output selection belongs to this wrapper. Other settings go directly
    # through the selected model's existing parser, retaining its own defaults.
    reserved = {"--image-path", "--image-paths", "--ocr-text-paths", "--output-path",
                "--output-dir", "--diagnostics-path", "--model-id", "--adapter-path",
                "--hybrid", "--fusion-json-paths"}
    if any(token.split("=", 1)[0] in reserved for token in config.model_args):
        raise ValueError("Use PipelineConfig or wrapper flags for input, output, and model selection")
    argv = list(config.model_args)
    model_id = config.model_id
    adapter_path = config.adapter_path
    # A local adapter directory can be selected as the model itself.
    if config.model_type == "qwen" and model_id and adapter_path is None:
        if (Path(model_id) / "adapter_config.json").is_file():
            adapter_path, model_id = Path(model_id), None
    if model_id:
        argv.extend(["--model-id", str(model_id)])
    if adapter_path is not None:
        if not adapter_path.is_dir():
            raise ValueError(f"Adapter directory not found: {adapter_path}")
        argv.extend(["--adapter-path", str(adapter_path)])
    if config.model_type == "qwen":
        if config.mode == "parser":
            # Native Qwen defaults/model resolution should select the text path.
            # The actual representation is supplied in memory, never read here.
            argv.extend(["--ocr-text-paths", "<in-memory-fusion>"])
        elif config.mode == "hybrid":
            argv.append("--hybrid")
        return qwen.parse_args(argv)
    return donut.parse_args(argv)


class InferencePipeline:
    """Reuse loaded models across sequential per-image predictions.

    predict() returns the generated JSON in result['prediction'], plus diagnostics
    and total/stage measurements. Loading failures raise; per-image failures return
    status='inference_error' with elapsed time and any available memory readings.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.args = _native_args(config)
        self.backend = qwen if config.model_type == "qwen" else donut
        template_path = self.args.template_path if config.model_type == "qwen" else self.args.example_path
        contract = qwen.load_hybrid_contract(self.args) if config.mode == "hybrid" else None
        self.evidence_mode = ((contract or {}).get("evidence_mode", self.args.hybrid_evidence or "hard")
                              if config.mode == "hybrid" else None)
        self.needs_ocr = config.mode == "parser" or config.mode == "hybrid" and self.evidence_mode in {"ocr", "hard"}
        self.needs_layout = config.mode == "parser" or config.mode == "hybrid" and self.evidence_mode in {"layout", "hard"}
        if self.needs_layout and config.yolo_model_dir is None:
            raise ValueError("Layout/hard-fusion inference requires yolo_model_dir")
        if contract is not None:
            self.schema = self.target_schema = contract["schema"]
            self.template = contract["template"]
        else:
            self.schema = self.backend.load_json(self.args.schema_path)
            self.template = self.backend.extract_json_target(
                self.backend.load_json(template_path), self.args.annotation_target_key,
            )
            self.target_schema = self.backend.select_schema_node_for_target(self.schema, self.args.annotation_target_key)
        if config.model_type == "qwen":
            errors = validate_json_schema(self.template, self.target_schema)
            if errors:
                raise ValueError("Template does not satisfy the selected JSON Schema: " + "; ".join(errors))
        self.runtime: Any = None
        self.ocr: Any = None
        self.layout: Any = None
        self.fusion = (FusionPipeline(FusionConfig(config.fusion_mode, config.min_overlap))
                       if config.mode in {"parser", "hybrid"} else None)

    def load(self) -> "InferencePipeline":
        if self.runtime is None:
            self.runtime = self.backend.load_inference_runtime(self.args)
        if self.needs_ocr:
            if self.ocr is None:
                self.ocr = OCRExtractor(self.config.ocr_config)
            self.ocr.load()
        if self.needs_layout and self.layout is None:
            from src.yolo_v11s.inference import LayoutDetector

            self.layout = LayoutDetector(self.config.yolo_model_dir, device=self.config.yolo_device)
        return self

    @staticmethod
    def _stage(stages: dict[str, Any], name: str, backend: str | None, callback: Any) -> Any:
        started = perf_counter()
        stage: dict[str, Any] = {"backend": backend}
        stages[name] = stage
        try:
            result = callback()
            if isinstance(result, dict):
                vram = result.get("vram")
                native_seconds = result.get("inference_seconds")
            else:
                vram, native_seconds = getattr(result, "vram", None), getattr(result, "inference_seconds", None)
            if vram is not None:
                stage["vram"] = vram
            if native_seconds is not None:
                stage["native_inference_seconds"] = native_seconds
            return result
        except Exception as exc:
            stage["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stage["inference_seconds"] = perf_counter() - started

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".pdf":
            raise ValueError("The unified entry point takes page images; render PDFs to page images first")
        self.load()
        stages: dict[str, Any] = {}
        record: dict[str, Any] = {
            "image_path": str(path), "mode": self.config.mode, "model_type": self.config.model_type,
            "model_id": getattr(self.runtime, "model_id", self.args.model_id),
            "adapter_path": str(self.args.adapter_path) if self.config.model_type == "qwen" and self.args.adapter_path else None,
            "fusion_mode": self.fusion.config.mode if self.fusion else None,
            **({"evidence_mode": self.evidence_mode} if self.evidence_mode is not None else {}),
            "prediction": None, "stages": stages,
        }
        # CPU-only Donut never probes unrelated GPUs. Qwen's native device map
        # and optional Paddle/YOLO devices are checked by the stage trackers.
        with PipelineTracker(sample_gpu=self.config.model_type == "qwen") as tracker:
            try:
                representation = None
                if self.fusion is not None:
                    ocr = self._stage(stages, "ocr", "paddle", lambda: self.ocr.extract(path)) if self.needs_ocr else None
                    if self.config.mode == "parser" and not any(line["text"].strip() for page in ocr["pages"] for line in page["lines"]):
                        raise ValueError("OCR text is empty; refusing extraction without evidence")
                    kwargs = {} if self.config.yolo_threshold is None else {"threshold": self.config.yolo_threshold}
                    layout = self._stage(stages, "yolo", "torch", lambda: self.layout.predict(path, **kwargs)) if self.needs_layout else None
                    if ocr is None:
                        # Only geometry is needed to normalize layout; no OCR model runs.
                        ocr = {"source_path": str(path), "pages": [{"page_number": 1,
                            "width": layout["width"], "height": layout["height"],
                            "coordinate_space": "input_image_pixels", "lines": []}]}
                    if layout is None:
                        page = ocr["pages"][0]
                        layout = {"image": str(path), "width": page["width"], "height": page["height"], "detections": []}
                    representation = self._stage(stages, "fusion", None, lambda: self.fusion.fuse(ocr, layout))
                    record["fusion"] = representation
                if self.config.model_type == "qwen":
                    kwargs = {} if representation is None else {"fusion_document": representation}
                    result = self._stage(stages, "generation", "torch", lambda: qwen.generate_image_prediction(
                        self.runtime, path, self.template, self.target_schema, self.args, **kwargs,
                    ))
                    record["prediction"] = result.prediction
                    record["status"] = qwen.result_status(result, self.template)
                    diagnostics = qwen.build_diagnostic_record(
                        args=self.args, runtime=self.runtime, output_path=path.with_suffix(".json"),
                        result=result, template=self.template,
                    )
                    diagnostics.pop("prediction_path", None)
                    if representation is not None:
                        if self.config.mode != "hybrid":
                            diagnostics["input_mode"] = "fusion"
                        diagnostics.pop("ocr_text_path", None)
                        diagnostics["image_path"] = str(path)
                    record["generation"] = diagnostics
                else:
                    result = self._stage(stages, "generation", "torch", lambda: donut.generate_image_prediction(
                        self.runtime, path, self.template, self.schema, self.args,
                    ))
                    prediction = result["generated"]["guided_prediction"]
                    errors = validate_json_schema(prediction, self.target_schema)
                    raw = result["generated"]["raw_prediction"]
                    has_fields = (not isinstance(self.template, dict) or
                                  isinstance(raw, dict) and (not self.template or bool(set(raw).intersection(self.template))))
                    record.update(prediction=prediction, status="template_error" if not has_fields else "schema_error" if errors else "ok",
                                  generation={**result, "schema_errors": errors})
            except Exception as exc:
                record.update(status="inference_error", error=f"{type(exc).__name__}: {exc}")
        attach_framework_peaks(tracker.metrics, stages, parser=self.config.mode in {"parser", "hybrid"})
        record.update(tracker.metrics)
        return record


def run_inference(image_paths: Sequence[str | Path], config: PipelineConfig) -> list[dict[str, Any]]:
    """Load the chosen pipeline once and return one result per input page image."""
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("Supply at least one image")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".pdf":
            raise ValueError("Render PDFs to page images before running the unified pipeline")
    pipeline = InferencePipeline(config)
    return [pipeline.predict(path) for path in paths]
