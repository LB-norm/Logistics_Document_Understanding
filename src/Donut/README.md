# Donut Fine-Tuning Dataset Format

This folder contains the Donut fine-tuning and inference scripts for the Lieferschein extraction task.

The dataset format below follows:

- the official Donut `metadata.jsonl` / `gt_parse` training format
- the local implementation in `src/Donut/train_finetune.py`
- the project-specific Lieferschein target schema in `src/Donut/lieferschein.schema.json`

## Expected Folder Layout

```text
dataset_name/
  train/
    metadata.jsonl
    sample-0001.png
    sample-0002.png
  validation/
    metadata.jsonl
    sample-0101.png
    sample-0102.png
```

- `train/` and `validation/` are required by the local training script.
- Each split must contain a `metadata.jsonl` file.
- Each JSONL record references an image file inside the same split directory via `file_name`.
- The image path may also include a relative subpath if the file still resolves inside the split directory.

## Required Record Structure

Each line in `metadata.jsonl` must be one JSON object with:

- `file_name`: relative path to the image file inside the split directory
- `ground_truth`: a JSON-dumped string containing at least `gt_parse`

The local Donut training script expects the document information extraction variant of the official Donut format, so `ground_truth` must contain `gt_parse` rather than `gt_parses`.

### Example `metadata.jsonl` Line

```json
{
  "file_name": "sample-0001.png",
  "ground_truth": "{\"gt_parse\":{\"document_type\":\"lieferschein\",\"document_language\":\"de\",\"sale_type\":\"BARVERKAUF\",\"delivery_note_number\":\"2017042708\",\"document_date\":\"27.04.2017\",\"document_date_iso\":\"2017-04-27\",\"customer_number\":\"10001\",\"clerk\":\"Carsten Hilgers\",\"issuer\":{\"name\":\"Carsten Hilgers Zweiraeder\",\"address\":{\"street\":\"Stahlwerkstr. 57\",\"postal_code\":\"26689\",\"city\":\"Apen\",\"country\":null,\"address_lines\":[\"Stahlwerkstr. 57\",\"26689 Apen\"]},\"contact\":{\"phone\":\"04489-63856\",\"fax\":\"04489-63857\",\"email\":\"zweirad.hilgers@t-online.de\"}},\"recipient\":null,\"items\":[{\"line_number\":1,\"quantity\":\"2\",\"unit\":\"Stk.\",\"article_number\":\"ET0001 - AV\",\"description\":\"RCP Fahrradschlauch 26 Zoll universal - Autoventil\"}],\"notes\":[\"Bei Rueckfragen erreichen Sie uns unter einer der angegebenen Telefonnummern.\"],\"signatory\":\"Carsten Hilgers\"}}"
}
```

In practice, `ground_truth` is a stringified JSON object. The outer JSONL record is parsed first, and then the `ground_truth` string is parsed again by the training script.

## `gt_parse` Structure For This Project

For this project, `gt_parse` should match the Lieferschein extraction target shape defined by:

- `src/Donut/lieferschein.schema.json`
- `src/Donut/lieferschein.example.json`

At a minimum, the current schema requires:

- `document_type`
- `issuer`
- `delivery_note_number`
- `document_date`
- `items`

The annotation target is a full JSON object representing the entire delivery note, not token spans or OCR text.

## Validation Rules Enforced By The Script

- `train/metadata.jsonl` and `validation/metadata.jsonl` must exist.
- Each record must contain `file_name`.
- Each record must contain `ground_truth`.
- Every referenced image file must exist.
- The parsed `ground_truth` object must contain `gt_parse`.

During training, the script converts `gt_parse` into Donut’s token sequence format automatically. It also derives special field tokens from:

- the JSON schema
- the actual keys found in the training and validation annotations
- the configured task start token, which defaults to `<s_lieferschein>`

## Practical Annotation Rules

- Keep the annotation keys stable across the full dataset.
- Use `null` for missing optional scalar values.
- Use `[]` for empty lists.
- Store the full structured target in `gt_parse`.
- Keep the JSON values close to the document surface form unless you intentionally want normalized fields such as `document_date_iso`.
- If you add new fields, keep them consistent in both the annotations and the schema.

## Smoke Test Behavior

If `src/Donut/train_finetune.py` is run without `--dataset-root`, it creates a tiny temporary smoke-test dataset automatically from:

- `data/Lieferschein-Beispiel.png`
- `src/Donut/lieferschein.example.json`

That path is useful for verifying the training pipeline, but it is not a real training dataset.

## Training Command

Once the dataset is ready, run:

```bash
python3 src/Donut/train_finetune.py --dataset-root path/to/donut_dataset
```

Example:

```bash
python3 src/Donut/train_finetune.py \
  --dataset-root data/my_donut_dataset \
  --model-id naver-clova-ix/donut-base \
  --output-dir models/donut-lieferschein \
  --task-start-token '<s_lieferschein>'
```

## Inference Command

After training, run:

```bash
python3 src/Donut/run_inference.py \
  --model-id models/donut-lieferschein \
  --task-prompt '<s_lieferschein>'
```

## Sources

- Official Donut repository: https://github.com/clovaai/donut
- Hugging Face Donut model documentation: https://huggingface.co/docs/transformers/model_doc/donut
