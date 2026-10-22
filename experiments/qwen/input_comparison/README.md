# Stage 1: image + OCR/layout input comparison

The current queue reruns **OCR + image** and **hard fusion + image** using
checkpointed chunked loss after the standard-loss OCR run exhausted GPU memory.
**Layout + image completed successfully and is excluded from the queue**; its
configuration and results are retained. The existing 9B image-only NF4 QLoRA run
remains the external baseline. Each queued run starts independently from
`Qwen/Qwen3.5-9B`, with a new `-chunked` run name; no old adapter initializes it.

| Run | Status | Evidence supplied with the full-page image |
| --- | --- | --- |
| `qwen35-9b-hybrid-input-ocr-nf4-r16-chunked` | Queued | OCR text, polygons and OCR confidence only |
| `qwen35-9b-hybrid-input-layout-nf4-r16` | Completed; excluded | Detected region labels, boxes and layout confidence only |
| `qwen35-9b-hybrid-input-hard-nf4-r16-chunked` | Queued | Both collections plus their hard memberships |

OCR input includes no YOLO regions or assignments. Layout input includes no OCR
text, OCR geometry, line IDs, or memberships. The full variant preserves unassigned
OCR lines. All three share the same schema, skeleton, system/task instructions,
source pages, sample order and assistant answers. Only the evidence block changes.
OCR coordinates are deliberately retained: this measures the value of adding the
layout detector and hard memberships beyond localized OCR, rather than testing
plain text transcription alone. Evidence is selected by a whitelist, so omitted
collections cannot leak through region text fields or execution metadata.

## Quantization and tuning choice

Use the image baseline's recipe consistently across all three runs. The reference
definition is [qwen35_9b_qlora_r16.json](../qwen35_9b_qlora_r16.json):

| Setting | Shared value |
| --- | --- |
| Base checkpoint | Qwen3.5-9B |
| Base-weight loading | 4-bit NF4, double quantization enabled |
| Compute dtype | BF16 |
| Language tuning | LoRA, all matching language-side linear modules |
| Vision encoder / merger | Both frozen |
| Adapter rank / alpha / dropout | 16 / 32 / 0.05 |
| Optimizer | `paged_adamw_8bit` (the image baseline's resolved `auto` choice) |
| Peak learning rate | `5e-5` |
| Image resolution | Medium, 2.8 MP maximum; full page, aspect ratio preserved |
| Physical / effective batch size | 1 / 8 on one GPU |
| Schedule | 10 epochs, cosine, 5% warmup |
| Weight decay / gradient clipping | 0.01 / 1.0 |
| Attention / gradient checkpointing | SDPA / enabled |
| Seed | 42 |
| Loss / sequence length | Queued OCR/hard runs: chunked answer-position loss, 256 positions per chunk; complete inputs without truncation |
| Parser dropout / image masking | Neither applied in stage one |

NF4 base-weight storage and BF16 computation are separate settings. This is QLoRA
on the language backbone, not BF16 base-weight LoRA and not visual LoRA. Using the
parser winner's BF16 recipe for these runs while retaining an NF4 image baseline
would vary input representation and optimization setup together. Precision and
visual adaptation can be revisited after selecting the input representation.

In the saved 2026-10-21 comparison, BF16 image L2 has the highest overall field F1
(0.8480), while the chosen 9B NF4 image reference reaches 0.8442. NF4 has higher
challenge F1 (0.7484 versus 0.7369) and schema validity (1.0000 versus 0.9333).
Thus NF4 is a strong chosen reference, although it is not the overall-F1 leader.
The parser leader uses BF16 LoRA and reaches 0.7958; hard-fusion NF4 reaches 0.7937.
These historical rankings do not establish an optimal hybrid precision.

The new runs match the version-controlled image-baseline training recipe. Their
hybrid prompts also include schema/skeleton and evidence instructions that differ
from the historical image-only prompt, so gains over that historical run are not
a perfectly isolated causal estimate of adding parser evidence. The three new
runs form the controlled input comparison; an additional matched image-only
rerun could strengthen the comparison later. It is deliberately absent from this
three-representation comparison as requested. Model-only checkpoint retention is
a storage policy change and does not alter optimizer updates. The current rerun
queue has two entries because layout + image is already complete.

## Chunked multimodal loss

The queued runs use the parser-to-LLM pipeline's existing chunked loss, extended
to pass images, image grids, modality IDs and positional inputs through Qwen's
native multimodal backbone. The complete image and parser sequence remain in
the context. Only positions predicting supervised answer tokens reach the LM head;
projection and FP32 cross-entropy run in checkpointed chunks of at most 256
positions. Teacher-forced validation uses the same loss; unlabeled generation and
saved-adapter inference retain Qwen's native forward.

For the recorded local training inputs, OCR averages 20,809 tokens (maximum
25,894), layout 7,485 (maximum 8,130), and hard fusion 23,610 (maximum 29,472).
Standard loss projects all those positions into the full vocabulary even though
prompt labels are masked. Chunking removes that vocabulary-sized memory cost for
prompt positions. Backbone activation memory still depends on the complete input;
peak VRAM must be measured on the training GPU.

The completed layout run keeps standard loss. The chunked implementation retains
the same full-vocabulary next-token objective, answer mask and microbatch mean;
changed matrix shapes and reduction order can introduce small numerical differences.
NF4, LoRA, resolution, data, schedule and effective batch size are unchanged.

## Prepare and validate

Run from the repository root using the project environment:

```bash
# Validate without writing:
python -m src.fusion.hybrid_sft \
  --evidence-modes ocr layout hard \
  --output-dir data/datasets/250_CMRS_240dpi_20260707/fusion_sft_input_comparison \
  --dry-run

# Prepare all three aligned inputs, without rerunning OCR/YOLO:
python -m src.fusion.hybrid_sft \
  --evidence-modes ocr layout hard \
  --output-dir data/datasets/250_CMRS_240dpi_20260707/fusion_sft_input_comparison

python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/input_comparison/queue.json --list

python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/input_comparison/queue.json --continue-on-error -- --dry-run
```

The default source is `fusion_sft_compact/` beside the new dataset; use
`--input-dir` to change it. The output contains `schema.json`, `template.json`,
`preparation_summary.json`, and `ocr/`, `layout/`, `hard/`, each with train/val JSONL.
Images are referenced relative to the output root rather than copied. Every mode
uses all 200 training and 50 validation pages in the current dataset. The builder
rejects existing output directories, preserves source data, and publishes only
after validating every requested mode and split. Use another output path for a
new build; no overwrite is implicit.

Check actual complete image + prompt + answer token lengths without model weights:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/input_comparison/queue.json --continue-on-error \
  -- --dry-run --check-multimodal-lengths
```

Unlike the previous four-run hybrid plan, these three representations have
different lengths; check each. Processor/model configuration files must be cached
or downloadable. Every actual training run automatically performs the same
preflight before loading weights, rejects oversized inputs, and records
`multimodal_sequence_lengths.json`. Chunked loss still requires a GPU memory check;
NF4 does not eliminate backbone activation memory. No dataset rebuild is needed
when switching to the chunked loss.

Optional loss/gradient comparison on the shortest complete multimodal example:

```bash
python src/Qwen/run_qwen_training.py \
  --config experiments/qwen/input_comparison/ocr_image_nf4_qlora_r16.json \
  --compare-loss --comparison-samples 1 \
  --output-dir runs/qwen-input-comparison-loss-check
```

This diagnostic also executes standard loss, so it can itself OOM; it is not part
of the training queue. It writes `loss_comparison.json` without optimizer updates.

## Training queue

```bash
bash experiments/qwen/input_comparison/run_queue.sh
```

Runs are written to `runs/qwen-input-comparison/<run-name>/`, with an appended
parent `queue.log`. Each experiment uses a fresh Python process. The launcher
attempts OCR + image and then hard fusion + image even after a failure, then
returns nonzero if either failed. Layout + image is not launched.
`QWEN_PYTHON` selects the interpreter and `QWEN_INPUT_RUNS_DIR` overrides the output
root. Additional arguments are common training overrides, for example:

```bash
bash experiments/qwen/input_comparison/run_queue.sh \
  --dataset-root /mnt/datasets/250_CMRS_240dpi_20260707/fusion_sft_input_comparison \
  --cache-dir /mnt/model-cache
```

To start at a later entry, use Python's queue runner with `--start-at` before `--`.
Each model retains the checkpoint with lowest epoch validation loss in
`best_model/`. Model-only snapshots do not contain resumable optimizer state.
Reruns use fresh `-chunked` directories under the same root, preserving the old
failed-run logs and completed layout adapter. They restart from the base model.
The retained adapter includes `hybrid_contract.json` recording its evidence mode,
rounding precision, prompts, schema and skeleton. Inference selects the evidence
mode from that contract and rejects contradictory overrides. Old hybrid contracts
without an evidence-mode field continue to mean hard fusion.

## Validation and test inference

```bash
python experiments/qwen/input_comparison/run_testset.py --split val
python experiments/qwen/input_comparison/run_testset.py --split test
```

The launcher discovers only completed runs under `runs/qwen-input-comparison`,
loads them in NF4, and writes predictions, diagnostics and audit artifacts to
`output/qwen-input-comparison/<split>/<run-name>/`. It runs models in separate
processes and continues after failures. Use `--run NAME` to select a run,
`--dry-run` to validate paths/contracts without loading models, and `--dataset-root`
to select the original image dataset (not the prepared training JSONL root).

OCR+image loads only OCR and Qwen; layout+image loads only YOLO and Qwen; the full
variant loads all three. Only used stages contribute to measured pipeline time.
Each image is supplied to Qwen in every mode. The existing `*.fusion.json` audit
export has empty placeholder collections for withheld sources; these are removed
from the actual model prompt. Manifest `evidence_mode` records the ablation.

Defaults use GPU PP-OCRv6 medium models under
`output/ocr_v6_model_cache/official_models/` and YOLO under
`runs/yolo_v11s/cmr_100epochs/best_model`. Override `--ocr-models-dir`,
`--ocr-device`, `--yolo-model-dir`, `--runs-dir`, and `--output-dir` as needed.
OCR paths are not required for the layout-only variant; YOLO paths are not
required for the OCR-only variant. Existing outputs in the selected destination
are replaced, following the other inference launchers.

Direct inference also works, for example:

```bash
python -m src.inference_pipeline --mode hybrid \
  --model-id runs/qwen-input-comparison/qwen35-9b-hybrid-input-ocr-nf4-r16-chunked/best_model \
  --load-in-4bit --resolution medium --images /path/to/page.png
```

Use the same command with a layout/hard adapter and `--yolo-model-dir` for those
variants. `--hybrid-evidence ocr|layout|hard` explicitly selects the input for an
unspecialized checkpoint; saved adapters infer it automatically. Supplying full
saved fusion JSON through the native Qwen CLI also projects the evidence to the
adapter's selected mode before generation.

Select the representation using full validation extraction metrics, then report
overall/default/challenge F1, precision/recall, schema validity, latency and peak
VRAM. The existing test scores informed this design; results on the same test set
are exploratory. Dropout and masking experiments remain stage two and are not
included in this queue. The older `experiments/qwen/hybrid/queue.json` is retained
as a separate earlier proposal; use this directory's queue for the current stage.
