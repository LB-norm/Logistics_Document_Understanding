"""Model-independent OCR/layout representations for downstream LLMs."""

from .pipeline import FusionConfig, FusionPipeline, fuse_document, to_llm_text

__all__ = ["FusionConfig", "FusionPipeline", "fuse_document", "to_llm_text"]
