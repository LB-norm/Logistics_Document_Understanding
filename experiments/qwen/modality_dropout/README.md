# Stage 2: modality dropout with hard fusion + image

Three independent runs start from `Qwen/Qwen3.5-9B` with the completed stage-one
hard-fusion + image run's NF4 QLoRA recipe. The reference is
`runs/qwen-input-comparison/qwen35-9b-hybrid-input-hard-nf4-r16-chunked/best_model`.
It is used for comparison, not as initialization, and is not queued again.
This campaign fixes the evidence mode to hard fusion (OCR + layout + memberships).

| Queue order | Config | Parser absent | Image absent | Both present |
| --- | --- | --- | --- | --- |
| 1 | `parser20_nf4_qlora_r16.json` | 20% | 0% | 80% |
| 2 | `image20_nf4_qlora_r16.json` | 0% | 20% | 80% |
| 3 | `balanced10_10_nf4_qlora_r16.json` | 10% | 10% | 80% |

Percentages are probabilities per training exposure, not fixed dataset subsets
or exact per-epoch quotas. One categorical draw selects parser absence, image
absence, or both present. The modalities are never dropped together; the balanced
run has a 20% total dropout probability. Python RNG is seeded by the existing
seed-42 training setup. All three configs use zero data-loader workers.

Parser dropout removes the dedicated evidence block and keeps the original full
page. Image dropout removes the image block and pixels, replaces the block with
an explicit image-absence notice, and retains all parser evidence. It does not
blank the image or mask image regions. Task instructions, schema, skeleton and
the complete assistant target are retained. The source JSONL and images are not
modified. Image-absent batches skip image processing; mixed batches supply only
the remaining images in placeholder order.

Dropout metadata is attached only to training records. Teacher-forced validation,
generation previews, best-checkpoint selection and default inference use both
modalities without random dropout. Full-input prompts match the stage-one prompt.
The saved training config and retained `hybrid_contract.json` record the policy.

## Shared training recipe

- Qwen3.5-9B, 4-bit NF4 with double quantization, BF16 compute.
- Language LoRA rank 16, alpha 32, adapter dropout 0.05, all language linear targets.
- Vision encoder and merger frozen; each run starts from the pretrained base.
- Checkpointed answer-position cross-entropy, 256 supervised positions per chunk.
- Medium resolution (2.8 MP maximum), complete parser/image inputs without truncation.
- Physical batch 1, accumulation 8, effective batch 8 on one GPU.
- 10 epochs, learning rate `5e-5`, cosine decay, 5% warmup, weight decay 0.01.
- `paged_adamw_8bit`, SDPA, gradient checkpointing, gradient clipping 1.0, seed 42.
- Evaluation/save each epoch; retain the adapter with lowest full-input validation loss.

Only modality-dropout probabilities and output/run names differ from the stage-one
hard-fusion configuration. Adapter dropout stays at 0.05 and is a separate mechanism.
The existing prepared `fusion_sft_input_comparison/hard/` train/val JSONL is reused
(200 training and 50 validation pages). No dataset rebuild or OCR/YOLO rerun is needed.

## Run from the repository root

```bash
# Show the three experiments without training:
python src/Qwen/run_qwen_experiment_queue.py \
  experiments/qwen/modality_dropout/queue.json --list

# Validate dataset paths and training options:
bash experiments/qwen/modality_dropout/run_queue.sh --dry-run

# Train all three sequentially:
bash experiments/qwen/modality_dropout/run_queue.sh
```

The shell launcher uses `QWEN_PYTHON`, the active environment, `.venv/bin/python`,
or `python3`, in that order. Outputs and the appended `queue.log` go under
`runs/qwen-modality-dropout/`; override with `QWEN_DROPOUT_RUNS_DIR`.
Additional shell arguments are forwarded as common training overrides, for example
`--dataset-root /mnt/datasets/fusion_sft_input_comparison --cache-dir /mnt/model-cache`.
Each experiment uses a fresh process, continues after failures, and the queue
returns nonzero if any experiment fails. Existing stage-one outputs are retained.

Run names are:

- `qwen35-9b-hybrid-dropout-parser20-nf4-r16-chunked`
- `qwen35-9b-hybrid-dropout-image20-nf4-r16-chunked`
- `qwen35-9b-hybrid-dropout-balanced10-10-nf4-r16-chunked`

All actual runs preflight the complete multimodal input lengths before loading
weights. Chunked loss reduces vocabulary-projection memory; complete-backbone
activation memory and actual VRAM still need measurement on the training GPU.

## Inference and comparison

```bash
python experiments/qwen/modality_dropout/run_testset.py --split val
python experiments/qwen/modality_dropout/run_testset.py --split test
```

These commands discover completed models under `runs/qwen-modality-dropout`, load
each `best_model` in NF4 in a fresh process, and supply full hard-fusion evidence
and the original full-page image. Predictions and audit outputs go to
`output/qwen-modality-dropout/<split>/<run-name>/`. Use `--run NAME` to select runs,
`--dry-run` to check paths, or the shared inference launcher's `--dataset-root`,
`--yolo-model-dir`, `--ocr-models-dir` and `--ocr-device` overrides when needed.

Compare full-input validation extraction F1, schema validity, latency and peak
VRAM against the stage-one hard-fusion run. Use the same image split and parser
models for every adapter. Report test scores after validation-based selection.
The existing `--omit-parser-evidence` option is available for a separate image-only
diagnostic; it is not enabled by the above commands. Image masking and further
missing-modality inference diagnostics are outside this three-run queue.
