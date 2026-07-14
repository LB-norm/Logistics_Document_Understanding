# Annotation audit

`annotation_audit.py` creates an explainable manual-review queue for JSON labels. It never changes an annotation.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.utils.annotation_audit
```

The default input is `data/datasets/250_CMRS_240dpi_20260707`. To audit another dataset with the same split layout, pass its root as the first argument:

```powershell
.\.venv\Scripts\python.exe -m src.utils.annotation_audit data\datasets\another_dataset
```

Open `data/datasets/250_CMRS_240dpi_20260707/annotation_audit/report.html` in a browser. The report provides image/JSON links, reasons for every flag, closest annotation peers, a side-by-side field comparison, filters, and local review decisions. The **Export review decisions CSV** button saves browser decisions as a CSV. Decisions are kept in browser local storage until exported; the source annotations are not edited.

The output directory also contains:

- `review_queue.csv`: document-level queue, sorted by risk, with issue types, affected JSON fields, and blank correction columns for spreadsheet review.
- `issues.csv`: one row per detector finding.
- `field_statistics.csv`: coverage, type, frequency, length, and numeric summaries for every field.
- `learned_rules.csv`: repeated entity/address relationships learned from the dataset and their exceptions.
- `summary.json`: machine-readable run summary.

## How suspicion scores work

The audit deliberately combines several weak, explainable signals instead of declaring that a rare value is wrong:

- JSON Schema violations and missing/unexpected fields.
- Semantic format errors such as impossible dates, blank strings, field swaps, and negative quantities.
- Conflicts inside one CMR, for example consignee city versus delivery city or goods-received stamp.
- Consistency exceptions learned from repeated companies and postcodes. A rule is only learned when an anchor occurs at least three times and one dependent value accounts for at least 80% of those occurrences.
- Byte-identical images carrying different annotations.
- Robust numeric outliers.

A high score means “review early,” not “automatically incorrect.” Edit `review_queue.csv` or export decisions from the report, inspect the image, and then change the original JSON manually. Tune learned-rule sensitivity with `--min-rule-support` and `--rule-confidence`; raising either option reduces flags.

For all command-line options:

```powershell
.\.venv\Scripts\python.exe -m src.utils.annotation_audit --help
```

## Normalize city-country location labels

The two route fields can be normalized separately with an evidence-based dry run:

```powershell
.\.venv\Scripts\python.exe -m src.utils.normalize_location_labels
```

The utility only proposes changes from `CITY` to `CITY, country` when completed peer annotations support the suffix. It uses sender/consignee context and the other route location to choose between language variants such as `Österreich` and `Rakousko`. Review `location_normalization/proposed_changes.csv` and `skipped_candidates.csv`, then apply the proposals with:

```powershell
.\.venv\Scripts\python.exe -m src.utils.normalize_location_labels --apply
```

Apply mode copies every original annotation into a timestamped backup directory before changing it, writes timestamped CSV and JSON manifests, and maintains `all_applied_changes.csv` across multiple passes.
