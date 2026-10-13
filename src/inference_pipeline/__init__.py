"""One entry point for end-to-end and OCR/layout-based document extraction."""

from .pipeline import InferencePipeline, PipelineConfig, run_inference

__all__ = ["InferencePipeline", "PipelineConfig", "run_inference"]
