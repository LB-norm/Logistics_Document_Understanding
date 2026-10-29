---
title: "Parser-based Pipeline Implementation — Report Notes"
chapter: "Implementation"
section: "Parser-based Pipeline"
document_type: "report_notes"
language: "en"
evidence_checked_on: "2026-10-25"
status: "draft"
evidence_summary: "data/parser_pipeline_evidence_summary.json"
---

# Parser-based Pipeline Implementation — Report Notes

## Purpose and evidence

These notes cover OCR, layout detection, OCR/layout representations, downstream
fine-tuning, the experiment matrix, and inference. Dataset composition and the
target schema are described in the
[Dataset and Output Representation notes](dataset_output_representation_notes.md).

The [machine-readable evidence summary](data/parser_pipeline_evidence_summary.json)
preserves experiment definitions, preparation statistics, target consistency
checks, layout split membership checks, and saved test-output coverage.

The parser training directories and YOLO training-run directory are not present
in the inspected workspace. Distinguish current implementation defaults,
version-controlled experiment settings, and facts directly recorded by prepared
data or inference artifacts. The saved outputs establish that all 15 parser
configurations reached test inference; they do not recover every historical
training override or the detector's exact training history.

## 1. Architecture and division of responsibilities

The implemented path is:

```text
Document page image
  → PP-OCRv6: recognized text, confidence, and polygons
  → YOLO11s on the same page: region labels, confidence, and boxes
  → common coordinates and chosen OCR/layout representation
  → text-only Qwen extraction
  → parsed and template-filled content JSON, plus diagnostics
```

- OCR and layout detection are separate components. OCR recognizes text; YOLO
  detects semantic document regions. Both receive the same source page, and the
  unified implementation executes their stages sequentially.
- Fusion uses geometry to organize evidence. It does not itself extract the
  final semantic values or equate detector labels with ground-truth answers.
- The downstream model generates the shared content JSON. It receives text
  tokens only, including serialized evidence, schema, and instructions.
- The extraction model uses the Qwen3.5 multimodal checkpoint loader, so vision
  weights remain resident even though this path does not process page pixels.
- Training is staged: prepared OCR and layout predictions are reused to train
  Qwen. There is no joint optimization or gradient flow through OCR, YOLO, or
  the geometric fusion stage.
- The implemented OCR-only baseline supplies plain recognized text without
  layout. The 15-run campaign instead compares three OCR-plus-layout formats.
  Its `separate` format is not an OCR-only ablation.

Sources: [pipeline composition](../../src/inference_pipeline/pipeline.py),
[OCR module](../../src/PP_parser/ocr.py),
[fusion module](../../src/fusion/pipeline.py),
[text-only Qwen training](../../src/Qwen/README.md#text-only-ocrlayout-training).

## 2. OCR implementation

### 2.1 Model selection and configuration

- `OCRConfig` and `OCRExtractor` wrap PaddleOCR's detection/recognition pipeline.
- The selected medium pair is explicitly named `PP-OCRv6_medium_det` and
  `PP-OCRv6_medium_rec`; model selection does not depend on language defaults.
- The recorded dataset preparation uses `gpu:0`, a recognition confidence
  threshold of `0.0`, and a PDF rendering setting of 240 DPI.
- Document orientation classification, document unwarping, and text-line
  orientation classification are disabled. This avoids silently introducing
  different page frames before OCR/layout fusion.
- The project uses pretrained OCR inference. It does not implement or document
  task-specific PP-OCRv6 fine-tuning for this campaign.
- All 250 inspected training/validation OCR artifacts record PaddleOCR 3.7.0,
  PaddleX 3.7.2, and `paddlepaddle-gpu` 3.3.1.

### 2.2 Input handling and coordinate preservation

- The extractor accepts image/PDF paths, PIL images, or unsigned 8-bit BGR or
  grayscale arrays. PIL images are converted from RGB to the backend's BGR frame.
- The adapter does not resize raster pages; Paddle may resize internally for
  detection. Effective detection settings are retained in page metadata.
- OCR geometry refers to the supplied raster. For PDFs, explicit page rendering
  makes coordinates refer to rendered pixels rather than PDF points.
- Standalone OCR supports multiple PDF pages, while batch fusion training uses
  one raster page per example and the unified pipeline accepts page images.
- OCR weights initialize once and are reused sequentially. Extraction and saving
  are separate operations, allowing downstream use of the result in memory.

### 2.3 OCR output contract

Each recognized line contains:

| Field | Meaning |
| --- | --- |
| `id` | Page-local identifier such as `p0001-l000000` |
| `line_index` | Position in the backend's recognition order |
| `text` | Recognized string |
| `confidence` | Recognition confidence |
| `polygon` | Four boundary-ordered points in input-image pixels |
| `bbox` | Axis-aligned envelope in `[x1, y1, x2, y2]` form |

Pages additionally contain dimensions, page number, coordinate-space declaration,
joined text, and detector settings. The document records provenance, settings,
package versions, and measurements. JSON geometry and a plain text transcript are
saved as separate `.ocr.json` and `.ocr.txt` artifacts.

OCR order is preserved rather than replaced by a new semantic reading order.
Recognition order should not be described as a guarantee of correct table or
field reading order.

### 2.4 Preparing downstream training records

- Existing train/validation membership and sample order are retained.
- The input is generated from document pixels; only the assistant answer comes
  from the annotation's `content` object.
- Empty or failed OCR remains visible in preparation metadata but is excluded
  from the OCR-only training JSONL. The preparation summary records those cases.
- In the inspected preparation, all 200 training and 50 validation pages produced
  successful nonempty OCR, so no examples were lost at this stage.
- These OCR artifacts are reused for the three layout representations rather
  than rerunning OCR independently for every downstream experiment.

Sources: [OCR implementation](../../src/PP_parser/ocr.py),
[OCR training-record preparation](../../src/PP_parser/ocr_dataset.py),
[recorded preparation settings](../../data/datasets/250_CMRS_240dpi_20260707/ocr_v6_cmr/preparation_summary.json),
[recorded OCR versions](data/parser_pipeline_evidence_summary.json).

## 3. YOLO11s layout detection

### 3.1 Detector and annotation task

- The detector starts from COCO-pretrained Ultralytics `yolo11s.pt` weights and
  adapts them to document regions. Use the model name **YOLO11s** consistently;
  the project module is named `yolo_v11s`.
- Layout supervision consists of bounding boxes and region categories, not the
  final field-value JSON and not OCR transcription targets.
- The project also provides Conditional DETR and Faster R-CNN implementations.
  The composed parser path uses YOLO11s. Their presence alone does not establish
  a quantitative detector ranking or why YOLO was selected.
- The Label Studio export has 15 categories, largely corresponding to the
  content groups, with `carrierLicensePlate` as an additional separate region.

### 3.2 Available layout split

| Layout partition | Images | Bounding boxes | Categories |
| --- | ---: | ---: | ---: |
| Training | 30 | 444 | 15 |
| Validation | 6 | 89 | 15 |
| Combined export | 36 | 533 | 15 |

The available split manifests are `Layout_dataset/annotations/instances_train.json`
and `instances_val.json`, with their images in `Layout_dataset/train` and `val`.
This is a separate dataset partition from the 200/50 extraction-model split.

**Membership detail for reporting:** six extraction-test page IDs occur in the
available layout training manifest: two default and four challenge pages. A
seventh extraction-test page occurs in layout validation. If these manifests
were used for the reported YOLO checkpoint, those pages were not unseen by every
component of the complete parser pipeline. Layout boxes and final content answers
are different supervision, but the distinction must be stated when describing
test independence. The checkpoint's original run configuration is needed to
confirm its exact training inputs.

The companion evidence summary lists each matching page ID and test subset.

### 3.3 COCO validation and YOLO conversion

- The loader validates image dimensions, category identity, annotation/image
  links, and finite nonempty boxes within image bounds.
- Sparse COCO category IDs are mapped to contiguous model class indices.
  Validation categories must match the training category mapping.
- The loader uses stored-raster coordinates without automatic EXIF rotation and
  ignores crowd annotations as individual training targets.
- Training and validation cannot reference the same resolved image paths.
  This check does not identify copies of a source document at other paths.
- The exporter writes numbered RGB PNGs and normalized YOLO labels inside the
  run directory. Numbering avoids filename collisions and preserves the source.
- COCO `[x, y, width, height]` boxes become normalized center coordinates:

```text
x_center = (x + width / 2) / image_width
y_center = (y + height / 2) / image_height
box_width = width / image_width
box_height = height / image_height
```

- A data configuration and a content-hash manifest preserve the class mapping
  and source identity. Resume checks reject changed source data or incompatible
  training settings.

### 3.4 Implemented training recipe and provenance limits

The following are current launcher defaults, not recovered settings from the
missing YOLO run directory:

| Setting | Implemented default |
| --- | --- |
| Initialization | `yolo11s.pt` |
| Epochs | 30 |
| Physical batch size / target accumulation | 2 / 4 |
| Optimizer | AdamW |
| Initial learning rate / weight decay | `1e-4` / `1e-4` |
| Schedule | Cosine decay to 1% of initial learning rate; 5% epoch warmup |
| Image size | 1344, with aspect-preserving letterbox resizing |
| Precision | FP32; CUDA FP16/BF16 selectable |
| Backbone | Trainable unless explicitly frozen |
| Seed / data-loader workers | 42 / 0 |
| NMS IoU / maximum detections | 0.7 / 300 |

- Nominal batch size is physical batch size × target accumulation. Ultralytics
  ramps accumulation during warmup, so it is not constant throughout training.
- The implementation delegates optimization, detection losses, augmentation, and
  validation to the Ultralytics trainer. It does not define a custom YOLO loss.
- Native augmentation includes mosaic, translation/scaling, HSV variation, and
  horizontal flipping. Actual resolved settings should be reported from
  `args.yaml`; do not describe them as a custom document augmentation policy.
- The wrapper disables early stopping with `patience=0`, saves epoch checkpoints,
  and exports the selected detector with its category/configuration metadata.
- The documented supported Ultralytics version selects the best checkpoint by
  validation mAP50–95. Precision, recall, and detection mAP evaluate the detector,
  separately from final JSON extraction quality.
- The parser test launcher points by default to
  `runs/yolo_v11s/cmr_100epochs/best_model`. The path suggests a 100-epoch run,
  but a directory name does not establish its exact arguments, selected epoch,
  detector accuracy, or training split. Recover those from the original
  `training_config.json`, `args.yaml`, and validation results before reporting them.

### 3.5 Detector inference output

- `LayoutDetector` loads weights and the saved layout configuration once and
  checks that saved category names match the model's class names.
- Inference uses native preprocessing and class-aware NMS without test-time
  augmentation. The default confidence cutoff is `0.5`, with a pipeline override.
- Predictions contain labels, original category IDs, confidence, and clipped
  pixel-space boxes in both `xyxy` and `xywh` formats.
- Invalid/empty clipped boxes are omitted, and retained detections are sorted
  by descending confidence.
- The downstream training evidence contains 3,061 predicted regions across 200
  training pages and 768 across 50 validation pages. These are predictions on
  extraction pages, distinct from the 533 manually exported layout boxes.

Sources: [YOLO training](../../src/yolo_v11s/train.py),
[shared COCO loader](../../src/conditional_detr/dataset.py),
[YOLO export](../../src/yolo_v11s/dataset.py),
[YOLO inference](../../src/yolo_v11s/inference.py),
[YOLO documentation](../../src/yolo_v11s/README.md),
[layout split and prediction evidence](data/parser_pipeline_evidence_summary.json).

## 4. OCR/layout representation and geometric fusion

### 4.1 Common coordinate system and page pairing

- OCR polygons and layout rectangles are normalized to `[0, 1]` by dividing x
  coordinates by page width and y coordinates by page height.
- The origin is the upper-left corner; rectangles use `[x1, y1, x2, y2]`.
- OCR and YOLO must describe the same page, crop, orientation, and rectification.
  Dimension disagreement is rejected by default.
- Optional layout rescaling covers whole-page scale differences, not crop,
  padding, perspective, or rotation differences.
- Page pairing uses explicit page numbers or an unambiguous source filename,
  not list order. Multi-page fusion requires explicit page correspondence.
- The fusion implementation uses the Python standard library and saved or
  in-memory predictions. It has no trainable parameters.

### 4.2 Compared representations

| Mode | Evidence and assignment | Question addressed |
| --- | --- | --- |
| `separate` / 1 | Complete OCR and layout collections in shared coordinates; no memberships | Can Qwen relate recognized text to regions itself? |
| `hard` / 2 | At most one qualifying region per OCR line | Does explicit geometric assignment help extraction? |
| `soft` / 3 | All qualifying regions with relative overlap weights | Does retaining geometric ambiguity help interpretation? |

All three preserve OCR text and layout geometry. Separate means no explicit
fusion assignments; it still includes layout detection.

### 4.3 Assignment rule

For an OCR polygon `P_i` and layout rectangle `R_j`:

```text
overlap(i, j) = area(P_i intersect R_j) / area(P_i)
candidate(i, j) requires positive overlap and overlap >= 0.1
```

- This measures the fraction of the OCR polygon covered by a region. It is not
  region IoU: a short line fully inside a large region has overlap 1.
- The implementation clips the actual OCR quadrilateral against the rectangle,
  preserving rotated text geometry. Bbox-only OCR is treated as rectangular.
- Hard mode chooses the highest overlap. Ties prefer smaller region area, then
  higher detector confidence, then region ID. Its retained membership weight is 1.
- Soft mode retains every candidate and sets each weight to that candidate's
  overlap divided by the sum of retained overlaps for the line.
- Soft weights are geometric weights, not calibrated probabilities. OCR and
  detector confidence remain separate evidence and do not multiply the overlap.
- Unassigned lines remain available; there is no nearest-region fallback.
  Zero-area OCR is retained as unassigned, and simple edge contact is not a
  qualifying assignment.
- Repeated/nested regions remain distinct. Full soft exports may repeat a line
  in several derived field texts, linked through its stable OCR ID.
- Field text follows OCR order, without a new reading-order algorithm or
  splitting lines across fields. Final semantic decisions remain with Qwen.

### 4.4 Evidence retained in the prepared dataset

| Split | OCR lines | Predicted regions | Unassigned lines in hard/soft | Lines with multiple candidates in soft |
| --- | ---: | ---: | ---: | ---: |
| Training | 37,917 | 3,061 | 18,285 | 831 |
| Validation | 9,319 | 768 | 4,466 | 194 |

The substantial unassigned collection is part of the input, not discarded text.
The counts do not establish how much of that text is task-relevant.

Sources: [fusion algorithm](../../src/fusion/pipeline.py),
[fusion contracts](../../src/fusion/README.md),
[prepared evidence statistics](data/parser_pipeline_evidence_summary.json).

## 5. Compact serialization and training-data construction

### 5.1 Why a compact representation was introduced

- Full exports retain duplicate derived text, field copies, geometry envelopes,
  and execution metadata, making serialized inputs large.
- The compact representation retains each OCR line once, region evidence, and
  existing memberships, while removing redundant copies and execution metadata
  from model input.
- It preserves sample IDs, split membership, target objects, and assistant answer
  strings. Numeric input evidence is rounded to four decimal places; annotation
  values are not rounded.
- Memberships are calculated before rounding and are not recomputed afterward.
- Short integer IDs are assigned independently to lines and regions within each
  page. An empty candidate list still identifies unassigned text.

| Compact collection | Array columns |
| --- | --- |
| `lines` | `[line_id, text, polygon, ocr_confidence]` |
| `regions` | `[region_id, label, box_xyxy, layout_confidence]` |
| `memberships` | `[line_id, [[region_id, overlap, weight], ...]]` |

Separate mode omits `memberships`. The system prompt defines these array columns.
Compaction reduces repetition and numeric precision; it does not truncate the
OCR text or drop lines to meet a length budget.

### 5.2 Observed size change

Mean training conversation sizes, before applying the chat template:

| Representation | Full characters, rounded | Compact characters, rounded |
| --- | ---: | ---: |
| Separate | 97,893 | 35,373 |
| Hard | 121,199 | 38,209 |
| Soft | 121,768 | 38,293 |

These values include prompt and answer message content. They are character
counts, not token counts or measured VRAM savings. Actual token lengths are
checked separately with the selected model's tokenizer.

### 5.3 Alignment and validation

- The batch builder joins source examples, OCR metadata, OCR conversations, and
  predicted layouts by sample identity and unique image filename.
- It requires matching sample sets, consistent split labels, valid page geometry,
  and valid targets; missing/extra layouts or failed OCR cannot silently remove
  examples from a representation comparison.
- Targets are compared with both the source annotation and the OCR assistant
  answer and validated against the schema.
- Every mode uses the same 200 training and 50 validation examples, in the same
  order. Provenance remains outside compact model messages.
- During this inspection, all six compact split/mode files matched their current
  source targets; assistant answer strings were preserved from the full files.
- The builder stages outputs before publishing. Dataset preparation uses saved
  predictions and does not rerun the models for each fusion mode.

Sources: [fusion dataset builder](../../src/fusion/dataset.py),
[compact encoding](../../src/fusion/compact.py),
[compact conversion](../../src/fusion/compact_sft.py),
[size and target checks](data/parser_pipeline_evidence_summary.json).

## 6. Prompts and downstream supervised fine-tuning

### 6.1 Prompt contract

- Each example is a system/user/assistant conversation.
- The base system instruction identifies CMR information extraction and requests
  strict JSON. Additional instructions provide the JSON schema and skeleton,
  explain text-only evidence, and define representation semantics.
- The user message contains the task instruction followed by the serialized
  evidence. Only the assistant message contains the correct content answer.
- The task instruction assigns fields by meaning, distinguishes absent and
  unreadable information, and requests empty lists for absent array entries.
- Parser-specific instructions state that the model cannot see the image and
  must not invent OCR-missed text or infer visual illegibility from omissions.
- Region labels and memberships are hints rather than authoritative semantic
  assignments. All lines, including unassigned text, must remain available.
- Compact conversion replaces the full-format explanation with the compact
  column definitions while retaining the schema, skeleton, and task contract.
- OCR text is treated as document data, not as instructions to the model.

### 6.2 Tokenization and supervision

- `--input-mode text` consumes the prepared conversations directly; it does not
  rebuild them with the image extraction prompt.
- The model's chat template constructs both the complete training conversation
  and the generation prompt without the assistant answer.
- The generation prompt must be an exact token prefix of the complete training
  conversation, making the supervision boundary explicit.
- The collator sets prompt and padding labels to `-100`. Only assistant answer
  tokens, including the template's answer termination, contribute to the loss.
- Training uses teacher-forced causal next-token prediction. Autoregressive
  validation previews generate without receiving the target answer.
- All complete conversations are tokenized before model weights load. The code
  checks prompt plus answer against the smallest applicable model/tokenizer/user
  context limit and rejects oversized examples rather than truncating them.
- Cached token IDs and `text_sequence_lengths.json` support inspection of the
  actual sequence lengths.

### 6.3 Adaptation scope

- Every campaign run starts independently from `Qwen/Qwen3.5-9B`.
- The vision encoder and merger remain frozen in all parser configurations.
- Adapter runs modify language-side linear modules selected through explicit
  module paths. The resolver excludes vision modules and the output head from
  LoRA targeting; embeddings and other base weights remain frozen.
- Full-language runs enable all non-vision parameters, including embeddings and
  the language output head. They are not full visual-model fine-tuning.
- QLoRA loads frozen base weights in NF4 with double quantization and uses BF16
  computation. BF16 LoRA loads the base without NF4. Distinguish weight storage
  precision from computation precision.

Sources: [shared prompts](../../src/Qwen/prompts.py),
[compact prompt conversion](../../src/fusion/compact_sft.py),
[tokenization, collator, and parameter selection](../../src/Qwen/qwen_finetune_logic.py).

## 7. Chunked loss and training feasibility

- All 15 current parser experiment definitions use `loss_implementation: chunked`
  with at most 256 supervised positions per chunk.
- Masking prompt labels alone does not avoid the standard forward's full
  vocabulary projection for prompt positions. Long parser inputs make those
  unused logits expensive.
- The custom loss runs the native backbone over the complete sequence, then
  selects hidden states whose next-token labels are supervised.
- It retains the causal shift: hidden state at position `t` predicts label
  `t + 1`, including the final prompt position predicting the first answer token.
- Only selected states pass through the language-model head. Full-vocabulary
  cross-entropy is computed in FP32 for each chunk; chunk sums are divided by the
  total number of supervised positions, rather than averaging chunk means equally.
- Non-reentrant checkpointing recomputes projection/loss chunks during backward
  to avoid retaining all vocabulary-sized intermediate tensors.
- The intended next-token objective, evidence context, answer mask, and existing
  microbatch/gradient-accumulation normalization remain unchanged. Different
  operation shapes/order can introduce numerical differences.
- This reduces output-projection memory. It does not remove complete-sequence
  backbone activation memory or establish a specific VRAM saving by itself.
- Teacher-forced validation uses the same loss. Unlabeled generation uses the
  native forward; training-time caches are disabled and generation caches remain
  available.
- Existing tests compare standard/chunked losses and gradients, including the
  first answer token, padding, and unequal chunks. A diagnostic loss-comparison
  path also exists. Their existence is not a claim that every production run
  performed the diagnostic.

Describe this mechanism here once; the Hybrid Pipeline section can refer back to
it and explain how multimodal inputs pass through the same native backbone.
Modality dropout belongs in that later section. LoRA's `0.05` adapter dropout is
a separate setting.

Sources: [chunked loss implementation](../../src/Qwen/chunked_loss.py),
[training integration](../../src/Qwen/qwen_finetune_logic.py),
[loss and gradient tests](../../tests/test_qwen_chunked_loss.py).

## 8. Experiment design and observed execution

### 8.1 Controlled comparison matrix

Five training recipes are crossed with separate, hard, and soft inputs: 15 runs.

| Recipe | Base-weight loading | Language adaptation | Peak learning rate | Representations |
| --- | --- | --- | ---: | --- |
| Q1 | NF4, double quantization, BF16 compute | LoRA | `5e-5` | Separate, hard, soft |
| L1 | BF16 | LoRA | `5e-5` | Separate, hard, soft |
| L2 | BF16 | LoRA | `1e-5` | Separate, hard, soft |
| F1 | BF16 | Full language parameters | `1e-6` | Separate, hard, soft |
| F2 | BF16 | Full language parameters | `5e-6` | Separate, hard, soft |

- Within a recipe, the comparison varies representation while preserving sample
  identities and answers.
- Q1/L1 compare base-weight quantization at the same nominal adapter rate;
  L1/L2 compare adapter learning rates; F1/F2 compare full-language learning rates.
- All use `adamw_torch`, including Q1. Optimizer-state quantization is not an
  additional varying factor in this campaign.
- These are screening configurations at one seed, not proof of optimal settings
  or repeated-seed statistical stability.

### 8.2 Shared configured settings

| Setting | Value |
| --- | --- |
| Training / validation examples | 200 / 50 |
| Physical training / evaluation batch size | 1 / 1 |
| Gradient accumulation / effective training batch | 8 / 8 on one GPU |
| Epoch budget | 10 |
| Adapter rank / alpha / dropout | 16 / 32 / 0.05 |
| Optimizer | AdamW with 32-bit states |
| Schedule / warmup | Cosine / 5% |
| Weight decay / gradient clipping | 0.01 / norm 1.0 |
| Attention / gradient checkpointing | SDPA / enabled |
| Loss | Chunked answer-position loss; chunk size 256 |
| Sequence handling | Complete conversations; no truncation |
| Validation / checkpointing | Once per epoch |
| Generated previews | Two fixed validation pages every 50 optimizer steps |
| Preview generation budget | Up to 2048 tokens, capped by remaining context |
| Seed / data-loader workers | 42 / 0 |

With 200 examples and effective batch size 8, the configured schedule corresponds
to 25 optimizer steps per epoch and approximately 250 over ten epochs. These are
derived schedule counts, not recovered parser trainer logs.

### 8.3 Checkpoint selection and run management

- The Trainer selects the lowest teacher-forced validation loss and reloads that
  checkpoint after training. Generated preview metrics do not select it.
- Two fixed preview pages support inspection; their scores are not full-split
  validation results.
- This campaign retains only the selected `best_model/`, using model-only
  snapshots without optimizer/scheduler resume state. Failed runs can leave
  partial snapshots; completed model-only runs cannot resume a prior trajectory.
- The queue launches each experiment in a fresh process, attempting adapter
  recipes before full-language recipes. The shell launcher continues after
  failures and reports a nonzero result if any entry failed.
- Experiment definitions, preparation summaries, processor files, run metadata,
  sequence-length reports, and diagnostics support traceability.

### 8.4 What the local artifacts establish

- `results/New/Parser_based` contains all 15 expected model/representation
  combinations, each with 30 test predictions and 30 manifest records: 450
  per-page inference records in total.
- Each run covers the expected test page IDs. Parse/schema failures remain
  recorded; having a saved prediction is not equivalent to successful extraction.
- The locally saved `evaluation_table.csv` contains only 14 rows and omits
  soft/F2, although soft/F2 predictions exist. Treat that table as an incomplete
  snapshot when assembling a complete results comparison.
- A separate saved runtime audit records all 15 runs and 450 samples.
- Keep numerical extraction rankings and runtime findings in the results chapter.
  The implementation section should explain the comparisons and selection process.
- Plain OCR, full serialization, and alternative layout detectors are implemented
  capabilities. The inspected 15-run campaign does not itself establish completed
  controlled experiments for each of those alternatives.

Sources: [campaign design](../../experiments/qwen/fusion/README.md),
[experiment queue](../../experiments/qwen/fusion/queue.json),
[queue launcher](../../experiments/qwen/fusion/run_queue.sh),
[saved evaluation table](../../results/New/Parser_based/evaluation_table.csv),
[saved runtime audit](../../results/comparisons/parser-fusion-20261021/runtime_audit_20261022.json),
[verified output inventory](data/parser_pipeline_evidence_summary.json).

## 9. Inference implementation

### 9.1 Model loading and execution

- The unified interface selects parser mode, Qwen model/adapter, YOLO checkpoint,
  OCR configuration, fusion mode, and output contract.
- Base models, adapter directories, and complete fine-tuned checkpoints are
  supported. Adapter metadata resolves the corresponding base checkpoint and
  saved processor.
- Models load once per process and are reused across page predictions. OCR,
  YOLO, fusion, and Qwen generation execute sequentially per image.
- Empty OCR stops a parser prediction with an error rather than generating from
  layout metadata alone.
- Qwen uses the tokenizer path without image tensors. Image resolution presets
  do not resize or alter these text-only inputs.
- Generation is deterministic (`do_sample=False`) and uses caching, with a
  configured maximum of 2048 new tokens in the campaign launcher. The generated
  answer is decoded after removing the prompt-token prefix.

### 9.2 Matching compact training and inference

- The generic unified parser path builds the full fusion representation.
- The dedicated `experiments/qwen/fusion/run_testset.py` launcher reads the saved
  training configuration, compact dataset summary, and a prepared training
  conversation to recover the exact system prompt, task prefix, fusion mode,
  and rounding precision.
- Inside each worker, it replaces the fusion-message builder with one that
  compacts fresh OCR/layout evidence using the same serialization and inserts
  it after the saved compact-input marker.
- Consequently, a generic parser invocation alone does not guarantee the compact
  prompt contract for these adapters. The campaign launcher supplies that match.
- The launcher also preserves the configured NF4/BF16 loading choice rather than
  relying on the native inference default of 4-bit loading.
- It supplies the prepared schema and template, explicitly selects GPU OCR and
  local PP-OCRv6 medium weights, and processes exactly 30 test PNGs.
- Each model runs in a separate process. Failures in one model do not prevent
  later models from being attempted.

### 9.3 JSON parsing, normalization, and failure handling

- Decoding removes thinking blocks and extracts a JSON candidate for strict
  parsing. Generation is not constrained by a schema-aware decoding grammar.
- If strict parsing fails, the code can retain complete top-level target fields
  before the failure. It does not infer missing values or repair arbitrary syntax.
- The output is recursively filled from the target template: required structure
  is retained, missing scalar values become null, and missing lists become empty.
  Unexpected object keys are omitted by template traversal.
- The resulting prediction is schema-checked. Raw output, parse status, recovered
  fields, and schema diagnostics remain separate, so template filling does not
  make an invalid raw generation count as a successful parse.
- Per-page exceptions are recorded and batch processing continues.

### 9.4 Saved artifacts and measurement boundary

| Artifact | Purpose |
| --- | --- |
| `<image-stem>.json` | Final template-filled extraction |
| `<image-stem>.fusion.json` | Detailed OCR/layout representation for inspection |
| `inference_manifest.jsonl` | Model provenance, raw generation, status, artifact paths, stage times, and memory measurements |

- The detailed fusion export remains available even when the model receives
  compact evidence.
- Per-page pipeline time includes OCR, layout, fusion, prompt preparation,
  generation, decoding, and JSON postprocessing. Model loading and output-file
  writing are outside that per-page interval.
- Measurements account for both Paddle and PyTorch. Their exact memory semantics,
  synchronization, sampling limitations, and aggregation belong in the
  Benchmarking and Evaluation Implementation section.

Sources: [pipeline implementation](../../src/inference_pipeline/pipeline.py),
[native Qwen inference](../../src/Qwen/run_inference.py),
[compact campaign inference launcher](../../experiments/qwen/fusion/run_testset.py),
[unified output artifacts](../../src/inference_pipeline/README.md#artifacts-and-measurements).

## 10. Details to retain when writing the report

- Keep OCR recognition, layout detection, geometric organization, and final
  semantic extraction distinct.
- Explain the three representation variants before the training matrix.
- Distinguish manually annotated layout boxes from model-predicted evidence.
- State that compact input preserves all OCR lines and that its size statistics
  are characters, while training checks actual tokens.
- Report the parser's chunked loss here and refer back to it in the hybrid section.
- Recover the original detector run configuration before claiming its exact
  training recipe, accuracy, or test independence.
- Keep the complete 15-run output inventory separate from the older 14-row
  extraction-results table.
- Use the results chapter for numerical model rankings and interpretation of
  quality/runtime trade-offs.
