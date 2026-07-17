# Annotation audit

`annotation_audit.py` creates an explainable manual-review workspace for JSON labels.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m src.utils.annotation_audit
```

The default input is `data/datasets/250_CMRS_240dpi_20260707`. To audit another dataset with the same split layout, pass its root as the first argument:

```powershell
.\.venv\Scripts\python.exe -m src.utils.annotation_audit data\datasets\another_dataset
```

Open `data/datasets/250_CMRS_240dpi_20260707/annotation_audit/review.html` in a browser. The self-contained workspace shows one document at a time with its image, reported issues and recommendations, closest peers, and side-by-side field comparison. **Edit JSON in VS Code** uses a platform-native `vscode://file` link to open the document's source annotation directly in VS Code on Windows or Linux. Chrome may ask once for permission to open the external application. Review state is retained in browser local storage; annotations are changed only when you manually edit and save them in VS Code.

At the bottom of each document, select **Reviewed** or use **Mark reviewed and continue**. The default queue contains open documents. **Export checklist CSV** exports every document with a `reviewed` or `open` status; it does not export audit decisions, notes, or proposed corrections. `report.html` is kept as a compatibility copy of the same review workspace.

The output directory also contains:

- `review_queue.csv`: document-level queue, sorted by risk, with issue types, affected JSON fields, and blank correction columns for spreadsheet review.
- `review.html`: primary browser review workspace. `report.html` contains the same page for compatibility with existing links.
- `issues.csv`: one row per detector finding.
- `field_statistics.csv`: one row for every target-schema leaf field, including fields with zero observations. It reports non-null example coverage, typed value counts, rarity/typo/format finding counts, length summaries, and numeric summaries.
- `field_values.csv`: complete typed value inventory. Every distinct non-null JSON value has its occurrence and example counts, frequencies, rarity class, example IDs, and likely typo suggestion. Unlike `top_values` in `field_statistics.csv`, this file is not truncated.
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
- Rare values in otherwise repetitive fields. All singletons remain visible in `field_values.csv`, but they enter the review queue only when the field has at least 20 values, no more than 20% distinct values, and the candidate accounts for no more than 2% of occurrences.
- Minority JSON types and character classes learned independently for each field. A finding requires at least ten values and a dominant profile covering at least 90% of occurrences.
- Inconsistent text variants which differ from a frequent form only through case, accents, spacing, or punctuation.
- Likely typos found by comparing rare text with much more frequent values in the same field. Suggestions require strong normalized edit similarity and preserve numeric components, so different street numbers are not proposed as corrections.

A high score means “review early,” not “automatically incorrect.” Inspect the image and peers, open and edit the source JSON in VS Code when needed, and mark the document reviewed. Tune learned-rule sensitivity with `--min-rule-support` and `--rule-confidence`; raising either option reduces flags.

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
