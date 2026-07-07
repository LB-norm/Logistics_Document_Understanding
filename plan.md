# High-Level Development Plan

## Summary

Build the project in layers: first define the benchmark target and evaluation contract, then make every pipeline produce comparable outputs, then fine-tune models, then run the final benchmark. The goal is to avoid training models before the project has a stable way to compare them.

## Work Order

### 1. Define The Evaluation Target

First, lock down what "correct output" means.

To-Do:

- Finalize the target JSON schema for evaluation.
- Decide whether evaluation uses the full annotation object or only `annotation["content"]`.
- Define required fields, optional fields, null handling, and allowed extra fields.
- Document the schema as the benchmark contract.

Default decision:

- Use `annotation["content"]` as the canonical target.
- Treat the existing `src/Donut/lieferschein.schema.json` as the schema source, but make clear which part is evaluated.

Outcome:

- Every pipeline is judged against the same final JSON target.

### 2. Build The Evaluation Module

Next, create the central evaluation module before doing serious training.

To-Do:

- Load dataset samples from the existing train/val/test split.
- Load predictions from all pipelines in one shared format.
- Normalize predictions before scoring.
- Compare normalized predictions against target annotations.
- Produce per-sample and aggregate reports.

Core metrics:

- JSON parse success
- schema validity
- field-level exact match
- normalized field match
- missing fields
- hallucinated fields
- null accuracy
- section-level scores
- item-list score

Outcome:

- You can evaluate saved predictions without loading any model.
- This becomes the backbone for comparing Donut, Qwen, and PaddleOCR-VL + LLM.

### 3. Define A Common Prediction Format

Before adapting individual pipelines, define the output format they all must write.

To-Do:

- Create one shared prediction record shape:
  - sample ID
  - pipeline name
  - model/checkpoint info
  - raw model output
  - parsed JSON
  - normalized JSON
  - status/errors
  - runtime metadata
- Require every pipeline to save predictions into the same run-folder structure.
- Include config files so every benchmark run is reproducible.

Outcome:

- The evaluator does not care whether a prediction came from Donut, Qwen, or PaddleOCR-VL + LLM.

### 4. Add Batch Inference For Each Pipeline

After the evaluation contract exists, make each pipeline produce benchmark-ready predictions.

To-Do:

- Donut:
  - Add or adapt batch inference over the test split.
  - Save outputs in the common prediction format.
- Qwen:
  - Add or adapt batch inference over the test split.
  - Support adapter paths for fine-tuned LoRA/QLoRA models.
- PaddleOCR-VL + local LLM:
  - Run PaddleOCR-VL parser over document images.
  - Feed parser text/Markdown/layout output into a local LLM.
  - Save final JSON predictions in the common format.
  - Store parser outputs as supporting artifacts.

Outcome:

- All pipelines can be run over the same test set and evaluated identically.

### 5. Establish Baselines Before Fine-Tuning

Run initial baselines to verify the benchmark works.

To-Do:

- Run Donut with an available base or smoke-test checkpoint.
- Run Qwen with the base model or a minimal local test if feasible.
- Run PaddleOCR-VL + local LLM with prompt-only extraction.
- Evaluate all baseline outputs.
- Inspect failure modes and confirm the metrics are meaningful.

Outcome:

- The benchmark pipeline is proven end-to-end before investing GPU time.

### 6. Fine-Tune All Approaches

Once evaluation and inference are stable, fine-tune models on the external machine.

To-Do:

- Fine-tune Donut on the train split.
- Fine-tune Qwen with LoRA/QLoRA on the train split.
- Prepare ERNIEKit SFT data and fine-tune PaddleOCR-VL VLM component.
- Optionally fine-tune the local LLM extractor later if prompt-only PaddleOCR-VL + LLM is weak.
- Save all training configs, commands, checkpoints, and run notes.

Outcome:

- Each approach has a trained version that can be plugged into the same benchmark.

### 7. Run Final Benchmark And Compare

After fine-tuning, run the full comparison.

To-Do:

- Run each trained pipeline on the same held-out test split.
- Evaluate all outputs with the same evaluation module.
- Generate aggregate reports.
- Compare:
  - overall JSON quality
  - section-level accuracy
  - item-list performance
  - schema validity
  - robustness to missing/null fields
  - runtime and practical complexity
- Identify representative success and failure examples.

Outcome:

- You have a defensible benchmark result for the thesis/project.

### 8. Improve Based On Results

Only optimize after the first full benchmark exists.

Likely follow-up work:

- Improve item-list matching if ordered comparison is too strict.
- Add parser-level evaluation if PaddleOCR-VL needs deeper analysis.
- Improve prompts for PaddleOCR-VL + local LLM.
- Add synthetic data only if it produces schema-valid labels.
- Expand annotations with OCR/layout evidence if parser evaluation becomes important.

## Recommended Priority

1. Final evaluation schema
2. Evaluation module
3. Common prediction format
4. Batch inference adapters
5. Baseline benchmark
6. Model fine-tuning
7. Final benchmark report
8. Targeted improvements

## Assumptions

- The first benchmark should evaluate final JSON only.
- Parser-level OCR/layout quality is useful but secondary.
- `annotation["content"]` is the target output.
- Fine-tuning happens later on another machine.
- The local codebase should focus first on making training results measurable and comparable.
