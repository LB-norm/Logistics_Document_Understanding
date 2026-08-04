# Donut training

This pipeline fine-tunes Donut for CMR and delivery-note extraction. It trains the full
vision encoder-decoder model and generates the annotation `content` object as Donut tokens.

## Files

- `run_donut_training.py`: training entry point and project defaults
- `donut_train_logic.py`: dataset loading, target serialization, training, previews, and
  checkpoint handling
- `run_donut_inference.py`: inference for a saved checkpoint

Edit `DEFAULT_TRAINING_CONFIG` in `run_donut_training.py` for the usual experiment setup.
Command-line arguments override individual defaults.

## Start training

Run these commands from the repository root.

Check the dataset without loading the model:

```powershell
.\.venv\Scripts\python.exe src\Donut\run_donut_training.py --dry-run
```

Start the configured run on an NVIDIA GPU:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
.\.venv\Scripts\python.exe src\Donut\run_donut_training.py --fp16
```

With an activated Linux environment:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python src/Donut/run_donut_training.py --fp16
```

The launcher uses `local_files_only=True`. If `naver-clova-ix/donut-base` is not cached,
allow the first run to download it:

```powershell
.\.venv\Scripts\python.exe src\Donut\run_donut_training.py --fp16 --no-local-files-only
```

The current defaults are:

| Setting | Value |
| --- | --- |
| Model | `naver-clova-ix/donut-base` |
| Image size | 1920 x 1280 |
| Maximum target length | 1,024 tokens |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Gradient checkpointing | enabled |
| Epochs | 50 |
| Learning rate | `3e-5` |
| Evaluation, checkpoints, logs, previews | every 50 optimizer steps |
| Preview documents | 2 |

Override individual settings directly:

```powershell
.\.venv\Scripts\python.exe src\Donut\run_donut_training.py `
  --fp16 `
  --image-size 1280 960 `
  --num-train-epochs 30 `
  --run-name donut-1280x960
```

If 1920 x 1280 runs out of memory, try `--image-size 1280 960`, then
`--image-size 960 720`. Keep the batch size at 1 and use gradient accumulation to control
the effective batch size.

## Dataset

The default dataset is `data/datasets/250_CMRS_240dpi_20260707`:

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

Each metadata row links an image and annotation:

```json
{
  "id": "cmr_example_page_1",
  "image": "train/images/cmr_example_page_1.jpg",
  "annotation": "train/annotations/cmr_example_page_1.json"
}
```

The annotation has a `content` object and may also contain metadata. Only `content` is used
as the training target by default. Set `--annotation-target-key root` only when the whole
annotation should be generated.

The loader also accepts:

- official Donut split folders whose metadata rows contain `file_name` and `ground_truth`
- flat folders with matching image and JSON files, such as `data/small testing`

For flat folders, one example is reused for validation. This is suitable for a smoke test,
not for measuring model quality.

## Target serialization

The target shape comes from `json_schema/content.empty.json` and
`json_schema/content.schema.json`. The empty JSON file defines field order. Nested objects
and item rows follow the same order; unexpected fields are appended alphabetically.

Field names found in the schema, skeleton, and annotations are added as Donut special
tokens. Before training starts, the code tokenizes every target and checks that
`--max-length` does not truncate it. Decoder position embeddings are extended when the
requested length exceeds the base model limit unless
`--no-resize-decoder-position-embeddings` is set.

## Validation during training

Two fixed validation documents are generated after each training log. Reports are written
to:

```text
runs/donut/<run-name>/validation_previews/
|-- latest.html
|-- latest.json
|-- step_00000050.html
`-- step_00000050.json
```

Open `latest.html` during training to compare the target and prediction. The report includes
parsing status and field-level differences. Use `--validation-preview-samples 0` to disable
previews or `--validation-preview-max-length` to shorten them.

Validation loss and previews measure different behavior:

- `eval_loss` uses teacher forcing over the validation set at `--eval-steps`.
- Previews use autoregressive `model.generate()` on the fixed sample at
  `--logging-steps`.

For generated metrics over the complete validation set, add `--predict-with-generate`.
This reports JSON parse rate, schema validity, document exact match, field
precision/recall/F1, and value similarity. It is slower because it generates every
validation document. With this option, checkpoints are ranked by `eval_json_field_f1`;
otherwise they are ranked by `eval_loss`.

Metric definitions are in [the evaluation-suite README](../eval_suite/README.md).

## Run output and checkpoints

Without `--output-dir`, runs are created under `runs/donut/`. A completed run contains:

- the selected model at the run root
- the best and last checkpoint directories
- `training_config.json`
- `trainer_state.json`
- `run_metadata.json`
- plots under `plots/`
- validation previews

`save_steps` is kept equal to `eval_steps` so checkpoint ranking uses a metric recorded at
the same step.

Regenerate plots for an existing run with:

```powershell
.\.venv\Scripts\python.exe -m src.utils.training_plots runs\donut\<run-name>
```

Resume an interrupted run:

```powershell
.\.venv\Scripts\python.exe src\Donut\run_donut_training.py `
  --fp16 `
  --output-dir runs\donut\<run-name> `
  --resume-from-checkpoint runs\donut\<run-name>\checkpoint-<step>
```

## Inference

Run a saved checkpoint on one image:

```powershell
.\.venv\Scripts\python.exe src\Donut\run_donut_inference.py `
  --model-id runs\donut\<run-name> `
  --task-prompt "<s_lieferschein>" `
  --image-path path\to\document.png `
  --schema-path json_schema\content.schema.json
```

Use `--help` to list the remaining options.
