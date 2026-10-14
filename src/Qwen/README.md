# Qwen QLoRA training

This pipeline fine-tunes Qwen models for CMR and delivery-note extraction using
document images or text-only OCR/layout conversations.
The default run uses `Qwen/Qwen3.5-2B`; its VRAM requirement depends strongly on the
selected resolution and document sequence lengths.

## Files

- `run_qwen_training.py`: training entry point and project defaults
- `qwen_finetune_logic.py`: dataset loading, collation, QLoRA setup, validation previews,
  and Trainer integration
- `experiment_config.py`: validation and loading for versioned experiment/queue JSON files
- `run_qwen_experiment_queue.py`: sequential single-GPU experiment queue runner
- `run_inference.py`: process document images or OCR text, with an optional saved adapter

For the parser baseline, `--ocr-text-paths` accepts UTF-8 OCR output and defaults
to the unadapted `Qwen/Qwen3.5-9B` checkpoint. The prompt includes the schema and
skeleton, and inference uses text tokens only. See the
[PP-OCR instructions](../PP_parser/README.md) for both pipeline stages.

Tracked experiment definitions live in `experiments/qwen/`. See its README for the
JSON formats and queue commands.

Edit `DEFAULT_TRAINING_CONFIG` in `run_qwen_training.py` for the usual experiment setup.
Any command-line argument overrides the corresponding default.

## Configured experiments and queues

Run one version-controlled experiment configuration:

```bash
python src/Qwen/run_qwen_training.py \
  --config experiments/qwen/placeholder_baseline.json \
  --dataset-root /mnt/datasets/250_CMRS_240dpi_20260707
```

Run every enabled entry in a queue sequentially on one GPU:

```bash
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/queue.json \
  -- \
  --dataset-root /mnt/datasets/250_CMRS_240dpi_20260707 \
  --runs-dir /mnt/experiments/qwen
```

Arguments after `--` are applied to every experiment and override values from its
JSON file. Each run uses a separate Python process, releasing its model and CUDA
state before the next experiment starts. The committed placeholder entries are
disabled and dry-run-only until the real campaign is defined.

## Start training

Run these commands from the repository root.

Check that the dataset and annotations can be read. This does not load Qwen:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py --dry-run
```

Start the default 2B QLoRA run:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py
```

With an activated Linux environment, the equivalent command is:

```bash
python src/Qwen/run_qwen_training.py
```

The training launcher enables PyTorch's `expandable_segments:True` CUDA allocator
setting before importing PyTorch. This reduces allocator fragmentation for variable
training allocation sizes. An explicitly exported `PYTORCH_CUDA_ALLOC_CONF` value
takes precedence over this default.

## Memory-efficient text loss and comparison

`--loss-implementation chunked --loss-chunk-size 256` enables an opt-in
text-only Qwen3.5 loss. `standard` remains the default and retains the original
forward. The chunked path runs the same native backbone on the full sequence,
selects positions whose **next-token** labels are supervised, and projects those
positions through the existing LM head in chunks. It uses native PyTorch FP32
cross-entropy and non-reentrant activation checkpointing for projection/loss.
Backward recomputes each chunk rather than retaining vocabulary-sized activations.
The full vocabulary, answer tokens, prompt gradient flow, and per-microbatch mean
loss/Trainer gradient-accumulation scaling are preserved. No optimizer, data,
quantization or learning-rate settings are changed. Unlabeled generation keeps
the original model forward. Teacher-forced evaluation returns loss only.

Chunk recomputation trades some work for memory; actual throughput and VRAM
savings must be measured on the training GPU. Shape changes in GEMM and reduction
order can cause small numerical differences, especially with BF16. This is not a
promise of bitwise-identical trajectories or final quality.

Before starting a new run, compare both implementations at identical weights and
RNG state on the shortest two **complete** training examples (no truncation):

```bash
python src/Qwen/run_qwen_training.py \
  --config experiments/qwen/fusion/separate/q1_nf4_qlora_r16_lr5e5.json \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707/fusion_sft_compact \
  --compare-loss --comparison-samples 2 --loss-chunk-size 256 \
  --output-dir runs/qwen-loss-check
```

`loss_comparison.json` reports losses, all trainable gradients' relative L2 error,
gradient-presence mismatches, measured forward/backward time, and CUDA peak memory.
It exits without optimizer updates, restoring RNG state and clearing gradients.
Pass `--resume-from-checkpoint /path/to/old/checkpoint-N` to compare at existing
checkpoint weights: this comparison loads **weights only**, not optimizer state.
Model-only checkpoints are sufficient for this check. Reference gradients are
held in CPU RAM; checking full tuning can therefore use substantial host memory.
Timings include warmup and are diagnostic rather than a throughput benchmark.
The report gates FP32 relative loss/gradient errors at `1e-5`/`1e-4`, and mixed
precision at `1e-4`/`0.02`. These are numerical regression gates, not quality claims;
review the actual reported differences. A failed comparison exits with code 1.

An interrupted standard run's early epochs can also be used for a short training
comparison. Keep the original dataset, seed, precision, batch/accumulation sizes,
optimizer, preview settings and **planned 20 epochs**. Restart the chunked run
from the same initialization and stop at the same optimizer step:

```bash
python src/Qwen/run_qwen_training.py \
  --config experiments/qwen/fusion/separate/q1_nf4_qlora_r16_lr5e5.json \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707/fusion_sft_compact \
  --loss-implementation chunked --loss-chunk-size 256 \
  --stop-after-steps 75 --save-resume-state --save-total-limit 2 \
  --output-dir runs/qwen-chunked-early-comparison
```

For this 200-example, batch-1, accumulation-8 configuration, each epoch has 25
optimizer steps; step 75 is three epochs. Set the stop step from the actual old
Trainer state/log. `--stop-after-steps` leaves the planned scheduler horizon intact;
`--max-steps 75` or `--num-train-epochs 3` would change the cosine schedule and
invalidate that comparison. Use a separate output directory and preserve the old
checkpoint/log. Compare the matching epoch validation losses:

```bash
python -m src.Qwen.compare_training_histories \
  /path/to/old/checkpoint-75/trainer_state.json \
  runs/qwen-chunked-early-comparison/trainer_state.json
```

Keep library/kernel versions and hardware fixed. The history comparison pairs
validation losses by optimizer step; it cannot verify those experimental controls
for you. It uses the first evaluation at each step so final reevaluation of the
selected best checkpoint cannot replace an epoch-boundary measurement.
Generation previews cover only two documents; assess field metrics on
the same complete validation split if comparing extraction quality. Early epochs
can expose regressions without finishing 20 epochs, but cannot prove final
convergence. For faithful training continuation, optimizer/scheduler/RNG resume
state is required; the fusion queue's model-only checkpoints do not contain it.

After verification, enable the loss on the queue by forwarding
`--loss-implementation chunked --loss-chunk-size 256` after its `--` separator.

The first run downloads the base model from Hugging Face. Use
`--local-files-only` only after the checkpoint is cached.

The default configuration is:

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-2B` |
| Quantization | NF4 4-bit with double quantization |
| Compute dtype | BF16 |
| Resolution | `medium` (2.8 MP maximum; default) |
| Maximum training sequence | Unset (no truncation) |
| Batch size | 1 |
| Gradient accumulation | 8 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| LoRA targets | all matching language-side linear layers |
| Vision encoder | frozen |
| Epochs | 10 |
| Maximum learning rate | `5e-5` |
| Learning-rate schedule | cosine decay |
| Warmup | 5% of optimizer steps |
| Weight decay | `0.01` |
| Evaluation and checkpoints | once per epoch |
| Logging and generated previews | every 50 optimizer steps |

To change one value without editing the launcher:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py `
  --num-train-epochs 6 `
  --learning-rate 3e-5 `
  --run-name qwen35-2b-six-epochs
```

## What is trained

Each project example contains the document image, a system instruction, an extraction
prompt, and the annotation's `content` object as the assistant answer. Loss is calculated
only on the assistant tokens. The prompt, padding, and image tokens are masked.

With `--input-mode text`, each example instead contains the saved system/user
conversation (including fused OCR/layout evidence) and the ground-truth assistant
JSON. It uses the same teacher-forced next-token cross-entropy over the assistant
answer, including its end token. Prompt and padding labels are masked; prompt
tokens still condition the predictions. Images are neither opened nor passed to
the model in training, teacher-forced evaluation, or validation generation.

With the default `--text-tuning lora --vision-tuning frozen
--vision-merger-tuning frozen` modes, the Qwen base weights and all visual
components remain unchanged. LoRA matrices are trained in language-model linear
layers. The saved adapter therefore needs the original base model at inference.

The JSON Schema is used for validation reports; it is not used to constrain decoding. The
model learns the key names, nesting, null handling, and field placement from the training
answers.

## Text-only OCR/layout training

The ready-to-run [9B fusion campaign](../../experiments/qwen/fusion/README.md)
provides five training configurations per mode and a 15-run sequential queue.

Prepare the inputs with the [batch fusion SFT builder](../fusion/README.md#batch-sft-dataset-preparation).
Then select one representation's directory. All three modes use the same training
path and the existing train/validation membership:

```bash
# Validate the prepared conversations without loading model dependencies:
python src/Qwen/run_qwen_training.py \
  --input-mode text \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707/fusion_sft/soft \
  --dry-run

# Also check actual token lengths using the model's tokenizer and configuration:
python src/Qwen/run_qwen_training.py \
  --input-mode text \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707/fusion_sft/soft \
  --dry-run --check-text-lengths

# Train with the standard QLoRA defaults:
python src/Qwen/run_qwen_training.py \
  --input-mode text \
  --dataset-root data/datasets/250_CMRS_240dpi_20260707/fusion_sft/soft \
  --schema-path data/datasets/250_CMRS_240dpi_20260707/fusion_sft/schema.json \
  --run-name qwen35-2b-fusion-soft
```

Replace `soft` with `separate` or `hard` and use a distinct run name for each
experiment. Pointing at `ocr_v6_cmr` instead supports the existing plain-OCR
baseline. Explicit `--train-file`/`--validation-file` values can also select files
such as `soft/train.jsonl` and `soft/val.jsonl` relative to a shared dataset root.
Without `--validation-file`, JSONL datasets try `validation.jsonl`, then `val.jsonl`.

Text JSONL records need a nonempty `messages` conversation ending in an assistant
JSON answer; plain-string content and typed text blocks are accepted. No image
input fields or image placeholders may be supplied. Image paths under `provenance`
are permitted and are not resolved. The loader preserves the supplied prompts
rather than rebuilding them with the image extraction prompt. Optional `target`
objects must match the assistant answers, supplied sample IDs must be unique across splits,
and fusion modes must agree across training and validation.

Normal text training automatically tokenizes every selected complete conversation
**before loading model weights**. It verifies that the inference generation prompt
is an exact token prefix of the training conversation, caches the token IDs, and
checks lengths against the smallest available limit from the language model
configuration, tokenizer, and optional `--max-length`. Oversized examples raise
an error; text training never truncates input evidence or the answer. The model's
chat template supplies the assistant framing and end tokens; special tokens are
not added a second time. `--check-text-lengths` performs the same preflight during
a dry run and may download tokenizer/processor/config files, without model weights.
Use `--local-files-only` when these files are already cached.

Text runs write `text_sequence_lengths.json` with actual per-example prompt,
assistant and complete-conversation token counts, plus split summaries and the
effective context limit. Validation previews generate from the saved input alone,
with their generation budget capped by remaining context space. Choose
`--schema-path` matching the prepared target when using a custom contract.

`--input-mode image` remains the default. Text mode uses the same Qwen3.5
multimodal checkpoint loader and still loads its vision weights, which stay
frozen. `--vision-tuning` and `--vision-merger-tuning` must both be `frozen`;
`--resolution` has no effect on text tokenization. Language LoRA/QLoRA, full
language tuning, checkpoint selection/resume, experiment JSON files/queues and
best/last exports continue through the existing training machinery. Experiment
files can select text training with `"input_mode": "text"`.

## Tuning modes

| Command | Trained parameters | Typical use |
| --- | --- | --- |
| `--text-tuning lora` | Language adapters | Default QLoRA/LoRA mode |
| `--text-tuning full --no-load-in-4bit` | Full language backbone and LM head | Full language adaptation |
| `--vision-tuning frozen\|lora\|full` | Vision blocks | Independent block control |
| `--vision-merger-tuning frozen\|lora\|full` | Vision-to-language merger | Independent interface control |
| `--vision-tuning full --vision-train-last-n-blocks 9` | Last nine vision blocks | Partial vision adaptation |

The code looks for `visual`, `vision_tower`, or `vision_model`. Use
`--vision-module-names` for another model layout.

Example with vision-side LoRA:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py `
  --vision-tuning lora `
  --vision-merger-tuning lora `
  --run-name qwen35-2b-vision-lora
```

## Larger Qwen models

The model loader and LoRA target selection are not tied to the 2B checkpoint. On a larger
machine, select another compatible checkpoint and adjust the memory settings:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py `
  --model-id Qwen/Qwen3.5-9B `
  --gradient-accumulation-steps 16 `
  --run-name qwen35-9b-language-qlora
```

The main memory controls are `--resolution`, batch size, LoRA rank, quantization, and
gradient checkpointing. Training leaves `--max-length` unset so the processor does not
truncate the image, prompt, or target JSON. Only set it after checking that every complete
processed sequence, including the assistant end token, fits below the chosen limit. Keep
`modules_to_save` empty unless an additional non-LoRA module must be trained and stored.

Image resolution is selected exclusively through four project presets. Each value is an
upper pixel budget; the processor preserves aspect ratio and patch-compatible dimensions:

| Preset | Maximum pixels |
| --- | ---: |
| `low` | 1.4 MP |
| `medium` | 2.8 MP (default) |
| `high` | 4.2 MP |
| `native` | 5.6 MP |

For example, add `--resolution high` to training or inference. `native` is the name of
the largest standardized project preset and still caps inputs at 5.6 MP; it does not mean
unlimited source resolution. Images below a preset's cap are not forced to use every pixel.

## Validation during training

The launcher selects two validation documents once and generates their JSON after each
training log. Reports are stored in:

```text
runs/qwen/<run-name>/validation_previews/
|-- latest.html
|-- latest.json
|-- step_00000050.html
`-- step_00000050.json
```

`latest.html` shows the target, prediction, field differences, parse errors, schema errors,
and field metrics. These previews use `model.generate()` without the target answer. Regular
`eval_loss` uses teacher forcing over the complete validation split, so the two measurements
answer different questions.

Preview generation affects runtime but does not create gradients. Adjust it with:

```text
--validation-preview-samples 0           disable previews
--validation-preview-max-new-tokens 2048 default generation budget
--logging-steps 100                      generate less often
```

## Best and last models

Teacher-forced validation and checkpointing run once per epoch by default.
Trainer selects the snapshot with the lowest `eval_loss` and reloads it after
training. The default resumable policy retains it alongside the final snapshot
and requires `save_total_limit` of at least 2.

A normal adapter run exposes both choices through stable model-only directories
and also retains resumable Trainer checkpoints:

```text
runs/qwen/<run-name>/
|-- best_model/     # adapter with the lowest validation loss
|-- last_model/     # adapter at the final training step
|-- checkpoint-*/  # best and last resumable Trainer checkpoints
`-- ...
```

The adapter at the run root is also the best model for backwards compatibility.
`best_model/` and `last_model/` contain processor files and can each be passed directly
to inference.

For one-shot runs, use `--save-only-model --save-total-limit 1`. Epoch snapshots
then omit optimizer, scheduler, scaler, and RNG state and cannot be used to
resume training. At completion, only the best snapshot is renamed in place:

```text
runs/qwen/<run-name>/
`-- best_model/     # lowest validation loss
```

No duplicate model is written at the run root in this mode. The final snapshot
is removed when it is not the best, along with all other epoch snapshots.

## Run output and resume

Without `--output-dir`, runs are created under `runs/qwen/`. Every completed run
contains the retained model directories, processor files,
`training_config.json`, `trainer_state.json`, and `run_metadata.json`. Adapter
runs also retain their resumable checkpoints.

Resume an interrupted run with the same output directory:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py `
  --output-dir runs\qwen\<run-name> `
  --resume-from-checkpoint runs\qwen\<run-name>\checkpoint-<step>
```

## Inference

The [unified inference entry point](../inference_pipeline/README.md) selects base
models or fine-tuned checkpoints for end-to-end inference and also runs
OCR → YOLO → fusion → text-only Qwen generation. It reuses this module's defaults.

Run one image with a saved adapter. The adapter metadata selects the matching base
model automatically, and the saved adapter processor is reused when available:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_inference.py `
  --adapter-path runs\qwen\<run-name>\best_model `
  --image-path path\to\document.png `
  --output-path output\qwen\document.json
```

The output file is the final template-filled content JSON. Raw model text, parse
errors, schema errors, and model provenance are stored separately in
`document.json.diagnostics.json`.

If strict JSON parsing fails, inference conservatively keeps only complete
top-level target fields decoded before the failure. It does not repair values,
rename keys, or continue past ambiguous syntax. The diagnostic record remains a
`parse_error` and stores the stable error kind and recovered field names, so a
recovered prediction never counts as a successfully parsed generation.

Use `--image-paths` to process several independent images in one invocation. The
base model and adapter are loaded only once, and each image receives its own final
JSON file:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_inference.py `
  --adapter-path runs\qwen-screening\qwen35-4b-qlora-r16-screening\best_model `
  --image-paths path\to\first.png path\to\second.png path\to\third.png `
  --output-dir output\qwen35-4b-best\validation
```

The multiple-image output directory contains one `<image-stem>.json` prediction per
input plus `inference_manifest.jsonl` with diagnostics. Input image stems must be
unique to prevent accidental output overwrites. `--template-path` and
`--schema-path` can select another output contract; their defaults are
`json_schema/content.empty.json` and `json_schema/content.schema.json`.

Each diagnostic record also contains `inference_seconds` and `vram` with
per-image peak allocated/reserved CUDA memory in bytes and GiB. Timing includes
input preprocessing, generation and JSON normalization, excluding model loading
and output writes. A single input uses `<prediction>.json.diagnostics.json`.
The prediction JSON itself retains only the target schema. CPU runs record time
and mark VRAM unavailable. Image and OCR-text inference use the same tracking.

Inference uses the same default system and user prompts as project fine-tuning.
They can be overridden with `--system-prompt` and `--user-prompt` for checkpoints
trained with a different prompt contract. The default `--max-new-tokens 2048`
accommodates the longest JSON targets in the current dataset. Inference also defaults to
`--resolution medium`, matching training. Use `--help` to list the remaining options.

The shared defaults are defined once in `src/Qwen/prompts.py`. The user prompt instructs
the model to assign values by semantic meaning rather than physical position, emit
`"<unreadable>"` only for information that is present but cannot be transcribed reliably,
and reserve `null` for information that is not provided.

The canonical default user prompt is:

```text
Extract all relevant document information into the target CMR/Lieferschein content JSON object. Assign information to fields according to its semantic meaning, not merely its physical position on the document.

If the document clearly contains information for a field but the value cannot be transcribed reliably because it is illegible, obscured, or degraded, output "<unreadable>" instead of guessing or inferring the value from context.

Use null when no information for that field is provided, including when its physical area is blank or contains text belonging to another field. Use [] when an array contains no entries.
```

## Dataset formats

### Project layout

The launcher defaults to `data/datasets/250_CMRS_240dpi_20260707`:

```text
dataset_root/
|-- train/
|   |-- metadata.jsonl
|   |-- images/
|   `-- annotations/
`-- val/
    |-- metadata.jsonl
    |-- images/
    `-- annotations/
```

A metadata row contains:

```json
{
  "id": "cmr_example_page_1",
  "image": "train/images/cmr_example_page_1.jpg",
  "annotation": "train/annotations/cmr_example_page_1.json"
}
```

`annotation["content"]` is the default target. Set `--annotation-target-key root` only
when the annotation wrapper should be part of the answer.

### Conversational JSONL

The alternative image layout contains `train.jsonl`, `validation.jsonl` (or `val.jsonl`), and referenced images.
Each row needs an image path and a chat conversation ending in the target assistant answer:

```json
{
  "id": "sample-0001",
  "image": "images/sample-0001.png",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "Extract this document as JSON."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "{\"senderInformation\": {}, \"itemList\": []}"}
      ]
    }
  ]
}
```

Image paths are relative to `--dataset-root`. Use `images` or `image_paths` for multi-page
examples and include one image block per image. The final message must have role
`assistant` and contain valid target JSON.

## References

- [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Qwen3.5 chat template](https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/chat_template.jinja)
- [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [TRL vision-language SFT guide](https://huggingface.co/docs/trl/main/en/training_vlm_sft)
