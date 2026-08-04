# Qwen QLoRA training

This pipeline fine-tunes Qwen vision-language models for CMR and delivery-note extraction.
The default run uses `Qwen/Qwen3.5-2B` and fits on an RTX 3080 Ti with 12 GB VRAM.

## Files

- `run_qwen_training.py`: training entry point and project defaults
- `qwen_finetune_logic.py`: dataset loading, collation, QLoRA setup, validation previews,
  and Trainer integration
- `train_finetune.py`: compatibility wrapper for the old entry point
- `run_inference.py`: load a saved adapter and process one image

Edit `DEFAULT_TRAINING_CONFIG` in `run_qwen_training.py` for the usual experiment setup.
Any command-line argument overrides the corresponding default.

## Start training

Run these commands from the repository root.

Check that the dataset and annotations can be read. This does not load Qwen:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py --dry-run
```

Start the default 2B QLoRA run:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py
```

With an activated Linux environment, the equivalent command is:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/Qwen/run_qwen_training.py
```

The first run downloads the base model from Hugging Face. Use
`--local-files-only` only after the checkpoint is cached.

The default configuration is:

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-2B` |
| Quantization | NF4 4-bit with double quantization |
| Compute dtype | BF16 |
| Image budget | 1,048,576 pixels |
| Maximum training sequence | 2,048 tokens |
| Batch size | 1 |
| Gradient accumulation | 8 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| LoRA targets | all matching language-side linear layers |
| Vision encoder | frozen |
| Epochs | 10 |
| Evaluation, checkpoints, previews | every 50 optimizer steps |

To change one value without editing the launcher:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py `
  --num-train-epochs 6 `
  --learning-rate 5e-5 `
  --run-name qwen35-2b-six-epochs
```

## What is trained

Each project example contains the document image, a system instruction, an extraction
prompt, and the annotation's `content` object as the assistant answer. Loss is calculated
only on the assistant tokens. The prompt, padding, and image tokens are masked.

With the default `--vision-tuning frozen` mode, the Qwen base weights and vision encoder
remain unchanged. LoRA matrices are trained in the language-model linear layers. The saved
adapter therefore needs the original base model at inference time.

The JSON Schema is used for validation reports; it is not used to constrain decoding. The
model learns the key names, nesting, null handling, and field placement from the training
answers.

## Vision modes

| Command | Trained parameters | Typical use |
| --- | --- | --- |
| `--vision-tuning frozen` | Language LoRA only | Default 12 GB run |
| `--vision-tuning lora` | Language and vision LoRA | Adapt visual features with a moderate memory increase |
| `--vision-tuning full --no-load-in-4bit` | Language LoRA and the full vision encoder | Larger GPU; the full vision module is saved with the adapter |

The code looks for `visual`, `vision_tower`, or `vision_model`. Use
`--vision-module-names` for another model layout.

Example with vision-side LoRA:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py `
  --vision-tuning lora `
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

The main memory controls are `--max-pixels`, `--max-length`, batch size, LoRA rank,
quantization, and gradient checkpointing. Keep `modules_to_save` empty unless an additional
non-LoRA module must be trained and stored.

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
--validation-preview-max-new-tokens 512 shorten generated answers
--logging-steps 100                      generate less often
```

## Run output and resume

Without `--output-dir`, runs are created under `runs/qwen/`. A completed run contains the
adapter, processor files, retained checkpoints, `training_config.json`,
`trainer_state.json`, and `run_metadata.json`.

Resume an interrupted run with the same output directory:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_qwen_training.py `
  --output-dir runs\qwen\<run-name> `
  --resume-from-checkpoint runs\qwen\<run-name>\checkpoint-<step>
```

## Inference

Run one image with a saved adapter:

```powershell
.\.venv\Scripts\python.exe src\Qwen\run_inference.py `
  --adapter-path runs\qwen\<run-name> `
  --image-path path\to\document.png
```

Use `--help` to list the remaining inference options.

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

The alternative layout contains `train.jsonl`, `validation.jsonl`, and referenced images.
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
- [Transformers Qwen3.5 documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [TRL vision-language SFT guide](https://huggingface.co/docs/trl/main/en/training_vlm_sft)
