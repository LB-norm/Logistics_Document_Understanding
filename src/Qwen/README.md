# Qwen Fine-Tuning Pipeline

This folder contains a Qwen3.5-27B LoRA/QLoRA training script and an inference script for the Lieferschein extraction task.

The trainer supports two dataset layouts:

- the current project dataset under `data/datasets/raw_data_20260527`
- a prepared Qwen conversational JSONL dataset under a folder such as `data/qwen_lora_dataset`

The conversational format is based on the official Hugging Face documentation for:

- Qwen3.5 multimodal support in Transformers
- vision-language SFT dataset formats in TRL
- SFT guidance to avoid truncating image tokens by leaving `max_length=None`

## Project Dataset Layout

The default dataset root is `data/datasets/raw_data_20260527`:

```text
dataset_root/
  train/
    metadata.jsonl
    images/...
    annotations/...
  val/
    metadata.jsonl
    images/...
    annotations/...
  test/
    metadata.jsonl
    images/...
    annotations/...
```

Each metadata row points to an image and annotation:

```json
{
  "id": "cmr_dachser__example_page_1",
  "image": "train/images/cmr_dachser/example_page_1.jpg",
  "annotation": "train/annotations/cmr_dachser/example_page_1_0.json"
}
```

By default, `annotation["content"]` is serialized as the final assistant message. Annotation `metadata` is ignored. Use `--annotation-target-key root` only if you intentionally want the full annotation wrapper as the target.

Run a dataset parsing dry run without loading model dependencies:

```bash
python3 src/Qwen/train_finetune.py \
  --dataset-root data/datasets/raw_data_20260527 \
  --dry-run
```

Runtime dependencies for actual training include `peft`. The default QLoRA path also requires `bitsandbytes`; pass `--no-load-in-4bit` for regular LoRA when you do not want the 4-bit path.

## Qwen JSONL Layout

```text
data/qwen_lora_dataset/
  train.jsonl
  validation.jsonl
  images/
    sample-0001.png
    sample-0002.png
    sample-0003-page1.png
    sample-0003-page2.png
```

- `train.jsonl` and `validation.jsonl` contain one JSON object per line.
- Image paths inside the JSONL files are resolved relative to `dataset-root`.
- Single-page samples should use `image` or `image_path`.
- Multi-page samples should use `images` or `image_paths`.

## Required Record Structure

Each record must contain:

- `messages`: a chat-style conversation
- one of `image`, `image_path`, `images`, or `image_paths`
- a final `assistant` message containing the target JSON string

The training script normalizes string content into typed text blocks, but the recommended format is the typed block structure below because it matches the official TRL vision dataset layout.

### Single-Image Example

```json
{
  "id": "sample-0001",
  "image": "images/sample-0001.png",
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You extract key information from German delivery note scans. Return JSON only."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        { "type": "image" },
        {
          "type": "text",
          "text": "Extract the document into the target Lieferschein JSON schema."
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "{\"document_type\":\"lieferschein\",\"document_language\":\"de\",\"sale_type\":null,\"delivery_note_number\":\"2017042708\",\"document_date\":\"27.04.2017\",\"document_date_iso\":\"2017-04-27\",\"customer_number\":\"10001\",\"clerk\":\"Carsten Hilgers\",\"issuer\":{\"name\":\"Carsten Hilgers Zweiraeder\",\"address\":{\"street\":\"Stahlwerkstr. 57\",\"postal_code\":\"26689\",\"city\":\"Apen\",\"country\":null,\"address_lines\":[\"Stahlwerkstr. 57\",\"26689 Apen\"]},\"contact\":{\"phone\":\"04489-63856\",\"fax\":\"04489-63857\",\"email\":\"zweirad.hilgers@t-online.de\"}},\"recipient\":null,\"items\":[{\"line_number\":1,\"quantity\":\"2\",\"unit\":\"Stk.\",\"article_number\":\"ET0001 - AV\",\"description\":\"RCP Fahrradschlauch 26 Zoll universal - Autoventil\"}],\"notes\":[],\"signatory\":\"Carsten Hilgers\"}"
        }
      ]
    }
  ]
}
```

### Multi-Image Example

```json
{
  "id": "sample-0003",
  "images": [
    "images/sample-0003-page1.png",
    "images/sample-0003-page2.png"
  ],
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You extract key information from German delivery note scans. Return JSON only."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        { "type": "image" },
        { "type": "image" },
        {
          "type": "text",
          "text": "Use both pages and return one JSON object for the full document."
        }
      ]
    },
    {
      "role": "assistant",
      "content": [
        {
          "type": "text",
          "text": "{\"document_type\":\"lieferschein\",\"document_language\":\"de\",\"sale_type\":null,\"delivery_note_number\":\"...\",\"document_date\":\"...\",\"document_date_iso\":null,\"customer_number\":null,\"clerk\":null,\"issuer\":{\"name\":\"...\",\"address\":null,\"contact\":null},\"recipient\":null,\"items\":[],\"notes\":[],\"signatory\":null}"
        }
      ]
    }
  ]
}
```

## Validation Rules Enforced By The Script

- The last message must be an `assistant` message.
- The final assistant message is the supervised target used for loss computation.
- The number of image placeholders in `messages` must match the number of image paths.
- If image placeholders are missing entirely, the script prepends them to the last `user` message automatically.
- Every referenced image file must exist.

## Practical Annotation Rules

- Keep the assistant output as strict JSON without markdown fences.
- Use `null` for missing scalar fields and `[]` for missing lists.
- Keep field names stable across the whole dataset.
- Store the full expected extraction result in the final assistant message.
- If you want the model to follow a schema exactly, include the schema or a short extraction instruction in the system or user message consistently across the dataset.

## Training Command

Dataset-only dry run:

```bash
python3 src/Qwen/train_finetune.py --dry-run
```

QLoRA command for the current project dataset:

```bash
python3 src/Qwen/train_finetune.py \
  --dataset-root data/datasets/raw_data_20260527 \
  --model-id Qwen/Qwen3.5-27B \
  --output-dir models/qwen-lieferschein-lora \
  --load-in-4bit \
  --compute-dtype bfloat16 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

For a prepared JSONL dataset, pass `--dataset-root data/qwen_lora_dataset`.

## Inference Command

After training, run:

```bash
python3 src/Qwen/run_inference.py --adapter-path models/qwen-lieferschein-lora
```

The default inference image and template come from `data/small testing` and use the annotation `content` object as the JSON skeleton.

## Sources

- Hugging Face model card for `Qwen/Qwen3.5-27B`: https://huggingface.co/Qwen/Qwen3.5-27B
- Hugging Face Transformers Qwen3.5 docs: https://huggingface.co/docs/transformers/model_doc/qwen3_5
- Hugging Face TRL dataset formats: https://huggingface.co/docs/trl/main/dataset_formats
- Hugging Face TRL SFT trainer docs: https://huggingface.co/docs/trl/sft_trainer
- Hugging Face TRL multimodal SFT guide: https://huggingface.co/docs/trl/main/en/training_vlm_sft
