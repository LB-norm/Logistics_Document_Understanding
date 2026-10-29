---
title: "Implementation Overview — Report Notes"
chapter: "Implementation"
section: "Implementation Overview"
document_type: "report_notes"
language: "en"
evidence_checked_on: "2026-10-25"
status: "draft"
---

# Implementation Overview — Report Notes

## Purpose and scope

These notes support the opening section of the Implementation chapter. They
introduce the task, the three extraction approaches, the overall workflow, and
the experimental environment. Detailed algorithms, training settings, and metric
definitions belong in the later sections listed below.

Statements about supported functionality describe the repository implementation.
Statements about completed experiments require saved run artifacts; a default or
an experiment configuration alone does not establish that an experiment ran.

## 1. Task and implementation objectives

- The project implements information extraction from scanned logistics documents
  into a predefined, machine-readable JSON structure.
- The repository supports CMR documents and German delivery notes. In the report,
  distinguish this supported scope from the document types actually included in
  the reported experiments.
- Target information includes sender, consignee, carrier, goods, and
  signature-related details.
- Three approaches address the same extraction task: parser-based, end-to-end,
  and hybrid extraction.
- The comparison considers extraction quality and computational requirements.
- A shared output contract and evaluation implementation connect the approaches.

Supporting repository sources:

- [Project overview](../../readme.md)
- [Output template](../../json_schema/content.empty.json)
- [Output schema](../../json_schema/content.schema.json)

## 2. High-level architecture

| Approach | Main processing steps | Evidence supplied to the extraction model |
| --- | --- | --- |
| Parser-based | Page image → PP-OCRv6 and YOLO11s → intermediate representation → Qwen → JSON | Recognized text, coordinates, and detected layout regions; no document pixels |
| End-to-end | Page image → Qwen or Donut → JSON | Document image, with model-specific task instructions |
| Hybrid | Page image → selected OCR/layout stages → image and intermediate representation → Qwen → JSON | Full-page image plus OCR evidence, layout evidence, or both |

### 2.1 Parser-based approach

- OCR recognizes text and its location on the document.
- YOLO detects document regions and their labels.
- An intermediate representation exposes the OCR and layout evidence to a
  downstream language model.
- Qwen interprets this textual representation and generates the target JSON
  without receiving the document image.
- Name the separate, hard, and soft representations here; explain their
  construction and differences in the Parser-based Pipeline section.

### 2.2 End-to-end approach

- The extraction model generates structured information directly from the image.
- No separate OCR or layout detection stage supplies evidence to the model.
- The repository provides Qwen and Donut implementations for this approach.

### 2.3 Hybrid approach

- Qwen receives the full-page image together with explicit parser evidence.
- Implemented input variants provide OCR evidence, layout evidence, or both with
  hard memberships.
- These variants allow the contribution of the evidence sources to be
  investigated.
- All approaches ultimately target the same content structure, allowing shared
  evaluation.

Supporting repository sources:

- [Unified pipeline implementation](../../src/inference_pipeline/pipeline.py)
- [Unified inference documentation](../../src/inference_pipeline/README.md)
- [OCR/layout fusion](../../src/fusion/README.md)
- [Hybrid input comparison](../../experiments/qwen/input_comparison/README.md)

## 3. Overall workflow

### 3.1 Model preparation and experimentation

1. Prepare document page images and their annotations.
2. Establish training, validation, and test partitions.
3. Derive the inputs required by each approach: images, parser representations,
   or combined inputs.
4. Fine-tune extraction models and train the layout detector where applicable.
5. Use validation data to monitor training and select checkpoints.
6. Retain selected models, processors, configurations, and run metadata for
   inference.

### 3.2 Inference and evaluation

1. Supply a document page image to the selected pipeline.
2. Execute the required processing stages and generate the extraction.
3. Parse the generated output, align it with the target template, and record
   structural diagnostics.
4. Save the prediction with separate provenance and measurement records.
5. Compare predictions with ground-truth annotations and aggregate quality and
   computational measurements.

### 3.3 Notes for the workflow diagram

- Start with a shared document-input node and branch into the three approaches.
- Show OCR and layout detection as separate components receiving the same image.
  The current pipeline executes these stages sequentially; a shared input does
  not imply concurrent execution.
- In the hybrid branch, draw a direct image connection to Qwen alongside the
  parser-evidence connection.
- Merge the approaches at structured output and evaluation.
- Show ground truth feeding training and evaluation. Ground-truth answers are
  unavailable during inference.
- If rectification is included, label it as optional preparation. The repository
  provides geometric correction utilities, but the unified inference pipeline
  does not automatically apply them.
- The unified inference interface processes individual page images. Render PDFs
  into pages before this entry point.
- Models load once and are reused across sequential page predictions.

Supporting repository sources:

- [Unified inference workflow and artifacts](../../src/inference_pipeline/README.md)
- [Optional document rectification](../../src/preprocessing/README.md)
- [Shared evaluation implementation](../../src/eval_suite/README.md)

## 4. Software organization and experiment management

- Python modules and command-line interfaces implement dataset preparation,
  training, inference, and evaluation.
- PyTorch and Hugging Face Transformers provide the main extraction-model
  infrastructure.
- PEFT and bitsandbytes support adapter training and quantized model loading.
- PaddleOCR and PaddlePaddle provide OCR functionality.
- Ultralytics provides YOLO11s layout detection.
- Shared utilities handle output validation, run metadata, prediction inspection,
  and computational measurements.
- Version-controlled JSON files define Qwen experiments. A queue runner executes
  configurations sequentially in fresh Python processes.
- Saved configurations, checkpoint information, training histories, and inference
  diagnostics make individual runs traceable.

For the report, summarize these responsibilities in a short paragraph rather
than describing every repository directory.

Supporting repository sources:

- [Main dependencies](../../requirements.txt)
- [YOLO dependencies](../../src/yolo_v11s/requirements.txt)
- [Experiment configuration and queue management](../../experiments/qwen/README.md)

## 5. Experimental environment

### 5.1 Information established by repository evidence

| Item | Recorded information | Evidence and scope |
| --- | --- | --- |
| GPU model | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | Saved Qwen run artifacts, including completed hybrid input-comparison runs |
| GPU memory capacity | 94.96 GiB reported total capacity | Saved CUDA out-of-memory diagnostic; this is capacity, not measured model usage |
| Execution arrangement | One recorded CUDA device, `cuda:0`; sequential single-GPU experiment queue | Inspected Qwen artifacts and queue documentation; does not establish the total number of GPUs installed in the host |
| Python | Python 3.12 | Documented development version; not a complete historical runtime record |
| Dependency specification | Package version ranges | Requirements files; these do not establish exact installed versions for every experiment |

Supporting repository sources:

- [Recorded GPU identity](../../runs/qwen-input-comparison/qwen35-9b-hybrid-input-hard-nf4-r16-chunked/vram_usage.json)
- [Runtime diagnostic reporting GPU capacity](../../runs/qwen27b-memory-01/qwen35-27b-memory-04-int8-high/run_metadata.json)
- [Python development setup](../../readme.md#setup)
- [Dependency specification](../../requirements.txt)
- [Single-GPU experiment queue](../../experiments/qwen/README.md)

The linked run artifacts are present in the inspected workspace but are ignored
by Git. Preserve the relevant evidence separately if these notes need to remain
self-contained outside this workspace.

### 5.2 Environment details to complete before finalizing the report

Record these for the machine and environment used in the reported experiments:

- CPU model and relevant core/thread count.
- Installed system RAM.
- Operating system and version.
- NVIDIA driver version.
- CUDA runtime used by the frameworks, distinguishing it from any installed
  system toolkit when necessary.
- Exact versions of the main training and inference packages.
- CPU/GPU placement of individual stages during benchmarking.
- Any differences between training and benchmarking machines or environments.

These details are not fully established by the inspected artifacts. Do not assume
that the current local environment matches the environment of earlier runs.

### 5.3 Device placement and feasibility

- OCR supports CPU or GPU execution; record the actual placement used for the
  reported measurements.
- The current Donut inference implementation is CPU-only. Distinguish this
  implementation detail from Donut training and account for it when interpreting
  runtime comparisons.
- Quantization and memory-saving training techniques support the feasibility of
  larger configurations. Mention their role here, then explain the concrete
  methods in the relevant pipeline sections.

Supporting repository sources:

- [Inference device settings and Donut limitation](../../src/inference_pipeline/README.md)
- [OCR device configuration](../../src/PP_parser/README.md)
- [Qwen training options](../../src/Qwen/README.md)

## 6. Boundaries with the remaining Implementation sections

| Later section | Details to develop there |
| --- | --- |
| Dataset Preparation and Output Representation | Dataset composition, default/challenge rationale, split construction, preprocessing procedure, annotation conventions, and JSON schema details |
| Parser-based Pipeline | OCR implementation, layout detector training, representation variants, downstream fine-tuning, experiments, and inference |
| End-to-End VLM Pipeline | Image preparation, prompts, model-specific fine-tuning, experiments, and inference |
| Hybrid Pipeline | Evidence preparation and variants, fine-tuning, chunked loss, modality dropout, experiments, and inference |
| Benchmarking and Evaluation Implementation | Benchmark orchestration, metric definitions, normalization, latency boundaries, VRAM measurement methods, and aggregation |

Keep the overview focused on the system's purpose, component relationships,
workflow, and experimental setting. Put numerical performance results in the
appropriate results chapter.
