"""Build a local HTML workspace for reviewing prediction/annotation differences.

The generated page has no runtime dependencies.  It can be opened directly from
disk, keeps review progress in browser local storage, and writes an XLSX review log.
Annotation edits themselves are deliberately delegated to the editor so that the
original JSON formatting and metadata are preserved.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

from src.eval_suite.normalization import (
    NormalizationConfig,
    is_empty_value,
    normalized_edit_similarity,
    values_equal,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "250_CMRS_240dpi_20260707"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "datasets" / DATASET_NAME / DATASET_NAME
DEFAULT_PREDICTION_ROOT = REPO_ROOT / "output" / "qwen" / "qwen35-9b-best"
DEFAULT_OUTPUT_PATH = DEFAULT_PREDICTION_ROOT / "review.html"
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "json_schema" / "content.empty.json"
SPLITS = ("train", "val")
MISSING = object()
PATH_TOKEN_PATTERN = re.compile(r"([^.\[\]]+)|\[(\d+)\]")


@dataclass(frozen=True)
class Difference:
    path: str
    annotation: str
    prediction: str
    kind: str
    similarity: float | None


@dataclass(frozen=True)
class ReviewSample:
    id: str
    split: str
    prediction_path: Path
    annotation_path: Path
    image_path: Path
    differences: tuple[Difference, ...]


@dataclass(frozen=True)
class ReviewData:
    samples: tuple[ReviewSample, ...]
    prediction_count: int
    exact_count: int
    warnings: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a side-by-side Qwen prediction review interface."
    )
    parser.add_argument(
        "--prediction-root",
        type=Path,
        default=DEFAULT_PREDICTION_ROOT,
        help=f"Prediction directory (default: {DEFAULT_PREDICTION_ROOT})",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset directory containing split metadata (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="HTML output path (default: <prediction-root>/review.html)",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=DEFAULT_TEMPLATE_PATH,
        help=(
            "Canonical empty JSON template defining field order "
            f"(default: {DEFAULT_TEMPLATE_PATH})"
        ),
    )
    return parser.parse_args(argv)


def _resolve_dataset_path(dataset_root: Path, relative_path: str) -> Path:
    candidate = (dataset_root / relative_path).resolve()
    root = dataset_root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"metadata path escapes dataset root: {relative_path}")
    return candidate


def _flatten_scalars(value: Any, path: str = "") -> dict[str, Any]:
    """Flatten scalar leaves while omitting empty container structure.

    Empty arrays and an inferred one-item array containing only nulls are therefore
    treated alike.  This avoids filling the review queue with template-only noise.
    """
    if isinstance(value, dict):
        leaves: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            leaves.update(_flatten_scalars(child, child_path))
        return leaves
    if isinstance(value, list):
        leaves = {}
        for index, child in enumerate(value):
            leaves.update(_flatten_scalars(child, f"{path}[{index}]"))
        return leaves
    return {path: value}


def _display_value(value: Any) -> str:
    if value is MISSING:
        return "∅ missing"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value if value else '""'
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _template_order_key(path: str, template: Any) -> tuple[int, ...]:
    """Return a structural sort key based on the canonical JSON template."""
    key: list[int] = []
    node = template
    for match in PATH_TOKEN_PATTERN.finditer(path):
        field, raw_index = match.groups()
        if raw_index is not None:
            key.append(int(raw_index))
            node = node[0] if isinstance(node, list) and node else MISSING
            continue

        if isinstance(node, dict):
            fields = tuple(node)
            try:
                rank = fields.index(field)
            except ValueError:
                rank = len(fields)
            node = node.get(field, MISSING)
        else:
            rank = 0
            node = MISSING
        key.append(rank)
    return tuple(key)


def compare_documents(
    annotation: dict[str, Any],
    prediction: dict[str, Any],
    *,
    normalization: NormalizationConfig | None = None,
    field_order_template: dict[str, Any] | None = None,
) -> tuple[Difference, ...]:
    """Return meaningful leaf differences in stable JSON-path order."""
    config = normalization or NormalizationConfig()
    expected = _flatten_scalars(annotation)
    predicted = _flatten_scalars(prediction)
    differences: list[Difference] = []

    paths = [*expected, *(path for path in predicted if path not in expected)]
    if field_order_template is not None:
        encounter_order = {path: index for index, path in enumerate(paths)}
        paths.sort(
            key=lambda path: (
                _template_order_key(path, field_order_template),
                encounter_order[path],
            )
        )
    for path in paths:
        left = expected.get(path, MISSING)
        right = predicted.get(path, MISSING)
        left_empty = left is MISSING or is_empty_value(left, config)
        right_empty = right is MISSING or is_empty_value(right, config)

        # Missing, null, and blank values are all unpopulated for evaluation.  Do
        # not turn template-shape differences into manual review work.
        if left_empty and right_empty:
            continue
        if left is not MISSING and right is not MISSING and type(left) is type(right) and left == right:
            continue

        similarity: float | None = None
        if left_empty:
            kind = "prediction_only"
        elif right_empty:
            kind = "annotation_only"
        elif values_equal(left, right, config):
            kind = "formatting_only"
            similarity = 1.0
        else:
            kind = "value_mismatch"
            similarity = normalized_edit_similarity(left, right, config)

        differences.append(
            Difference(
                path=path,
                annotation=_display_value(left),
                prediction=_display_value(right),
                kind=kind,
                similarity=similarity,
            )
        )
    return tuple(differences)


def _load_metadata(dataset_root: Path, split: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    metadata_path = dataset_root / split / "metadata.jsonl"
    by_image_stem: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    if not metadata_path.is_file():
        return by_image_stem, [f"Missing metadata: {metadata_path}"]

    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                stem = Path(str(record["image"])).stem
                if stem in by_image_stem:
                    warnings.append(
                        f"{metadata_path}:{line_number}: duplicate image stem {stem}"
                    )
                    continue
                by_image_stem[stem] = record
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                warnings.append(f"{metadata_path}:{line_number}: invalid row: {error}")
    return by_image_stem, warnings


def load_review_data(
    prediction_root: Path,
    dataset_root: Path,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> ReviewData:
    samples: list[ReviewSample] = []
    warnings: list[str] = []
    prediction_count = 0
    exact_count = 0
    try:
        field_order_template = json.loads(template_path.read_text(encoding="utf-8"))
        if not isinstance(field_order_template, dict):
            raise TypeError("template root must be a JSON object")
    except (OSError, json.JSONDecodeError, TypeError) as error:
        field_order_template = None
        warnings.append(f"Could not load field-order template {template_path}: {error}")

    for split in SPLITS:
        metadata, metadata_warnings = _load_metadata(dataset_root, split)
        warnings.extend(metadata_warnings)
        prediction_dir = prediction_root / split
        if not prediction_dir.is_dir():
            warnings.append(f"Missing prediction split: {prediction_dir}")
            continue

        for prediction_path in sorted(prediction_dir.glob("*.json")):
            if prediction_path.name == "inference_manifest.json":
                continue
            prediction_count += 1
            record = metadata.get(prediction_path.stem)
            if record is None:
                warnings.append(f"No metadata match for prediction: {prediction_path}")
                continue
            try:
                image_path = _resolve_dataset_path(dataset_root, str(record["image"]))
                annotation_path = _resolve_dataset_path(
                    dataset_root, str(record["annotation"])
                )
                prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
                annotation_document = json.loads(annotation_path.read_text(encoding="utf-8"))
                annotation = annotation_document["content"]
                if not isinstance(prediction, dict) or not isinstance(annotation, dict):
                    raise TypeError("prediction and annotation['content'] must be JSON objects")
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                warnings.append(f"Could not load {prediction_path}: {error}")
                continue

            differences = compare_documents(
                annotation,
                prediction,
                field_order_template=field_order_template,
            )
            if not differences:
                exact_count += 1
                continue
            samples.append(
                ReviewSample(
                    id=f"{split}:{prediction_path.stem}",
                    split=split,
                    prediction_path=prediction_path.resolve(),
                    annotation_path=annotation_path.resolve(),
                    image_path=image_path.resolve(),
                    differences=differences,
                )
            )

    samples.sort(key=lambda sample: (-len(sample.differences), sample.split, sample.id))
    return ReviewData(tuple(samples), prediction_count, exact_count, tuple(warnings))


def _relative_url(target: Path, report_dir: Path) -> str:
    relative = os.path.relpath(target.resolve(), report_dir.resolve()).replace(os.sep, "/")
    return quote(relative, safe="/.:_-~")


def _vscode_url(target: Path) -> str:
    return "vscode://file" + target.resolve().as_uri().removeprefix("file://")


def _display_path(target: Path) -> str:
    try:
        return str(target.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(target.resolve())


def _json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def _inline_diff_segments(
    annotation: str,
    prediction: str,
    difference_kind: str | None = None,
) -> list[dict[str, str]]:
    """Build a compact character diff for the review page.

    Replacements are emitted as adjacent annotation-only and prediction-only
    segments, allowing the HTML to distinguish each side with its own color.
    """
    if difference_kind in {"annotation_only", "prediction_only"}:
        return [
            {"kind": "annotation_only", "text": annotation},
            {"kind": "prediction_only", "text": prediction},
        ]

    # Pull out shared boundaries before SequenceMatcher sees the remaining text.
    # This keeps repeated characters aligned intuitively: 44009 -> 44099 is
    # shown as 440 [0 -> 9] 9, rather than as a deletion plus a trailing insert.
    prefix_length = 0
    max_prefix = min(len(annotation), len(prediction))
    while (
        prefix_length < max_prefix
        and annotation[prefix_length] == prediction[prefix_length]
    ):
        prefix_length += 1

    suffix_length = 0
    max_suffix = min(
        len(annotation) - prefix_length,
        len(prediction) - prefix_length,
    )
    while (
        suffix_length < max_suffix
        and annotation[len(annotation) - suffix_length - 1]
        == prediction[len(prediction) - suffix_length - 1]
    ):
        suffix_length += 1

    left_end = len(annotation) - suffix_length if suffix_length else len(annotation)
    right_end = len(prediction) - suffix_length if suffix_length else len(prediction)
    left_middle = annotation[prefix_length:left_end]
    right_middle = prediction[prefix_length:right_end]

    segments: list[dict[str, str]] = []
    if prefix_length:
        segments.append({"kind": "equal", "text": annotation[:prefix_length]})
    matcher = difflib.SequenceMatcher(
        None,
        left_middle,
        right_middle,
        autojunk=False,
    )
    for operation, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if operation == "equal":
            segments.append({"kind": "equal", "text": left_middle[left_start:left_end]})
        else:
            if operation in {"delete", "replace"}:
                segments.append(
                    {
                        "kind": "annotation_only",
                        "text": left_middle[left_start:left_end],
                    }
                )
            if operation in {"insert", "replace"}:
                segments.append(
                    {
                        "kind": "prediction_only",
                        "text": right_middle[right_start:right_end],
                    }
                )
    if suffix_length:
        segments.append({"kind": "equal", "text": annotation[-suffix_length:]})
    return segments


def write_review_html(
    data: ReviewData,
    output_path: Path,
    dataset_root: Path,
    prediction_root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_samples = []
    for sample in data.samples:
        serializable_samples.append(
            {
                "id": sample.id,
                "name": sample.prediction_path.stem,
                "split": sample.split,
                "image": _relative_url(sample.image_path, output_path.parent),
                "annotation": _relative_url(sample.annotation_path, output_path.parent),
                "image_name": sample.image_path.name,
                "annotation_name": sample.annotation_path.name,
                "annotation_path": _display_path(sample.annotation_path),
                "prediction_path": _display_path(sample.prediction_path),
                "annotation_editor_url": _vscode_url(sample.annotation_path),
                "prediction_editor_url": _vscode_url(sample.prediction_path),
                "differences": [
                    {
                        **asdict(difference),
                        "segments": _inline_diff_segments(
                            difference.annotation,
                            difference.prediction,
                            difference.kind,
                        ),
                    }
                    for difference in sample.differences
                ],
            }
        )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(dataset_root.resolve()),
        "prediction_root": str(prediction_root.resolve()),
        "prediction_count": data.prediction_count,
        "exact_count": data.exact_count,
        "samples": serializable_samples,
        "warnings": list(data.warnings),
    }
    rendered = HTML_TEMPLATE.replace("__REVIEW_DATA__", _json_for_script(payload))
    output_path.write_text(rendered, encoding="utf-8")


def build_review(
    prediction_root: Path = DEFAULT_PREDICTION_ROOT,
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_path: Path | None = None,
    template_path: Path = DEFAULT_TEMPLATE_PATH,
) -> tuple[ReviewData, Path]:
    prediction_root = prediction_root.resolve()
    dataset_root = dataset_root.resolve()
    output_path = (output_path or prediction_root / "review.html").resolve()
    data = load_review_data(prediction_root, dataset_root, template_path.resolve())
    write_review_html(data, output_path, dataset_root, prediction_root)
    return data, output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data, output_path = build_review(
        prediction_root=args.prediction_root,
        dataset_root=args.dataset_root,
        output_path=args.output,
        template_path=args.template_path,
    )
    difference_count = sum(len(sample.differences) for sample in data.samples)
    print(f"Review page: {output_path}")
    print(
        f"Predictions: {data.prediction_count}; documents with differences: "
        f"{len(data.samples)}; field differences: {difference_count}; "
        f"without meaningful differences: {data.exact_count}"
    )
    if data.warnings:
        print(f"Warnings: {len(data.warnings)} (shown in the review page)")
    return 0


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Qwen prediction review</title>
  <style>
    :root{color-scheme:light;--ink:#172026;--muted:#66737b;--line:#d7dfe3;--bg:#f2f5f6;--blue:#1769aa;--red:#b42318;--red-bg:#fff1f0;--amber-bg:#fff8e8;--green:#087443;--soft:#eef3f5}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}a{color:var(--blue)}
    header{padding:14px 22px;background:#fff;border-bottom:1px solid var(--line)}h1{font-size:21px;margin:0 0 3px}h2{font-size:17px;margin:0}.muted{color:var(--muted)}
    main{max-width:1640px;margin:auto;padding:17px 22px 60px}.progress{display:flex;justify-content:space-between;gap:12px;margin-bottom:10px}.card{overflow:hidden;background:#fff;border:1px solid var(--line);border-radius:8px}.head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 15px;border-bottom:1px solid var(--line)}.head>div:first-child{min-width:0;overflow-wrap:anywhere}.count{flex:none;padding:4px 9px;color:#fff;background:var(--red);border-radius:999px;font-weight:700}
    .layout{display:grid;grid-template-columns:minmax(300px,34%) minmax(0,1fr);min-height:560px}.scan{padding:12px;background:#e7ecef;border-right:1px solid var(--line)}.scan img{position:sticky;top:12px;width:100%;max-height:calc(100vh - 55px);object-fit:contain;background:#fff}.links{display:flex;gap:12px;flex-wrap:wrap;margin-top:9px}.details{min-width:0;padding:13px 15px 20px;overflow:auto}
    table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{padding:8px;border:1px solid var(--line);text-align:left;vertical-align:top;overflow-wrap:anywhere}th{position:sticky;top:0;background:var(--soft);font-size:12px}.field{width:20%}.difference{width:26%}.checked{width:70px;text-align:center;vertical-align:middle}.value,.diff{white-space:pre-wrap}.value{background:var(--red-bg)}.formatting{background:var(--amber-bg)}.diff{background:#fbfcfd;line-height:1.65}.diff-annotation,.diff-prediction{padding:1px 2px;border-radius:3px;font-weight:700;box-decoration-break:clone;-webkit-box-decoration-break:clone}.diff-annotation{background:#ffd9d5;color:#8f1d14;text-decoration:line-through;text-decoration-thickness:1.5px}.diff-prediction{background:#d8f3e5;color:#08663c;text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:2px}.diff-legend{display:block;margin-top:3px;color:var(--muted);font-size:10px;font-weight:400;line-height:1.3}.legend-annotation{color:#8f1d14}.legend-prediction{color:#08663c}.kind{display:inline-block;margin-top:4px;padding:2px 6px;border-radius:999px;background:#fbd5d2;color:#7a271a;font-size:11px}.kind.formatting_only{background:#fde9b4;color:#784c00}.similarity{font-size:11px;color:var(--muted);margin-top:3px}
    input[type=checkbox]{width:19px;height:19px;accent-color:var(--blue);cursor:pointer}.finish-area{margin-top:15px;padding-top:15px;border-top:1px solid var(--line)}.unreadable{display:flex;align-items:center;gap:9px;width:max-content;max-width:100%;font-weight:650}.finish{display:block;width:100%;margin-top:14px;padding:11px 16px;border:1px solid #0e568e;border-radius:6px;background:var(--blue);color:#fff;font:inherit;font-size:15px;font-weight:700;cursor:pointer}.finish:hover{background:#10598f}.finish:disabled{opacity:.6;cursor:wait}.notice{margin-top:9px;color:var(--green)}.empty{padding:60px;text-align:center}.empty h2{margin-bottom:7px;color:var(--green)}details.warning{margin-bottom:11px;padding:10px 14px;background:#fff;border:1px solid var(--line);border-radius:7px}summary{cursor:pointer;font-weight:650}
    @media(max-width:900px){main{padding:12px}.layout{grid-template-columns:1fr}.scan{border-right:0;border-bottom:1px solid var(--line)}.scan img{position:static;max-height:75vh}.field{width:24%}.checked{width:62px}}
  </style>
</head>
<body>
<header><h1>Qwen prediction review</h1><div class="muted">Check the actual annotation issues, optionally flag the document for an &lt;unreadable&gt; token, then finish the review. The first finish chooses the XLSX save location.</div></header>
<main><div id="warnings"></div><div id="progress" class="progress"></div><div id="content"></div></main>
<script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('review-data').textContent);
const namespace='qwenPredictionReview:v2:'+data.prediction_root;
const labels={value_mismatch:'value mismatch',prediction_only:'only in prediction',annotation_only:'missing from prediction',formatting_only:'formatting / representation'};
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const xml=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&apos;'}[c]));
function loadState(){try{const v=JSON.parse(localStorage.getItem(namespace)||'{}');return{reviews:v.reviews||{},drafts:v.drafts||{}}}catch{return{reviews:{},drafts:{}}}}
let state=loadState(),currentId=data.samples.find(s=>!state.reviews[s.id])?.id||'',workbookHandle=null,notice='';
function saveState(){localStorage.setItem(namespace,JSON.stringify(state))}
function draft(s){return state.drafts[s.id]??={checked:[],unreadable:false}}
function inlineDiff(d){return d.segments.map(segment=>segment.kind==='equal'?esc(segment.text):`<span class="${segment.kind==='annotation_only'?'diff-annotation':'diff-prediction'}">${esc(segment.text)}</span>`).join('')}
function rows(s){const checked=new Set(draft(s).checked);return s.differences.map(d=>`<tr><td><code>${esc(d.path)}</code><br><span class="kind ${d.kind}">${esc(labels[d.kind])}</span>${d.similarity!==null&&d.kind==='value_mismatch'?`<div class="similarity">${Math.round(d.similarity*100)}% similar</div>`:''}</td><td class="value ${d.kind==='formatting_only'?'formatting':''}">${esc(d.annotation)}</td><td class="value ${d.kind==='formatting_only'?'formatting':''}">${esc(d.prediction)}</td><td class="diff">${inlineDiff(d)}</td><td class="checked"><input class="issue-check" type="checkbox" data-path="${esc(d.path)}" ${checked.has(d.path)?'checked':''} aria-label="Checked issue ${esc(d.path)}"></td></tr>`).join('')}
function nextOpen(afterId){const start=Math.max(0,data.samples.findIndex(s=>s.id===afterId));for(let n=1;n<=data.samples.length;n++){const s=data.samples[(start+n)%data.samples.length];if(!state.reviews[s.id])return s.id}return''}
function render(){const reviewed=Object.keys(state.reviews).length,s=data.samples.find(x=>x.id===currentId&&!state.reviews[x.id])||data.samples.find(x=>!state.reviews[x.id]);currentId=s?.id||'';document.getElementById('progress').innerHTML=`<b>${reviewed} of ${data.samples.length} documents reviewed</b><span class="muted">${data.samples.length-reviewed} remaining</span>`;if(!s){document.getElementById('content').innerHTML=`<div class="card empty"><h2>Review complete ✓</h2><div>All reviewed documents are recorded in qwen_prediction_review.xlsx.</div>${notice?`<div class="notice">${esc(notice)}</div>`:''}</div>`;return}const d=draft(s);document.getElementById('content').innerHTML=`<article class="card"><div class="head"><div><h2>${esc(s.name)}</h2><div class="muted">${esc(s.split)} · ${esc(s.annotation_name)}</div></div><span class="count">${s.differences.length} differences</span></div><div class="layout"><div class="scan"><a href="${s.image}" target="_blank"><img src="${s.image}" alt="Scan ${esc(s.name)}"></a><div class="links"><a href="${s.image}" target="_blank">Open full-size scan ↗</a><a href="${s.annotation}" target="_blank">Open annotation JSON ↗</a><a href="${s.annotation_editor_url}">Edit annotation JSON</a></div></div><div class="details"><table><thead><tr><th class="field">Field</th><th>Annotation</th><th>Qwen prediction</th><th class="difference">Difference<span class="diff-legend"><span class="legend-annotation">red = annotation only</span> · <span class="legend-prediction">green = Qwen only</span></span></th><th class="checked">Checked</th></tr></thead><tbody>${rows(s)}</tbody></table><div class="finish-area"><label class="unreadable"><input id="unreadable" type="checkbox" ${d.unreadable?'checked':''}> Add &lt;unreadable&gt; token later?</label><button id="finish" class="finish">Finish review</button>${notice?`<div class="notice">${esc(notice)}</div>`:''}</div></div></div></article>`;document.querySelectorAll('.issue-check').forEach(el=>el.onchange=()=>{const values=new Set(draft(s).checked);el.checked?values.add(el.dataset.path):values.delete(el.dataset.path);draft(s).checked=[...values];saveState()});document.getElementById('unreadable').onchange=e=>{draft(s).unreadable=e.target.checked;saveState()};document.getElementById('finish').onclick=()=>finishReview(s)}

const encoder=new TextEncoder();
function joinBytes(parts){const size=parts.reduce((n,p)=>n+p.length,0),out=new Uint8Array(size);let offset=0;for(const part of parts){out.set(part,offset);offset+=part.length}return out}
function u16(n){return new Uint8Array([n&255,(n>>>8)&255])}function u32(n){return new Uint8Array([n&255,(n>>>8)&255,(n>>>16)&255,(n>>>24)&255])}
const crcTable=(()=>{const table=new Uint32Array(256);for(let n=0;n<256;n++){let c=n;for(let k=0;k<8;k++)c=(c&1)?0xedb88320^(c>>>1):c>>>1;table[n]=c>>>0}return table})();
function crc32(bytes){let c=0xffffffff;for(const b of bytes)c=crcTable[(c^b)&255]^(c>>>8);return(c^0xffffffff)>>>0}
function dosStamp(){const d=new Date(),year=Math.max(1980,d.getFullYear());return{time:(d.getHours()<<11)|(d.getMinutes()<<5)|(d.getSeconds()>>1),date:((year-1980)<<9)|((d.getMonth()+1)<<5)|d.getDate()}}
function zip(entries){const local=[],central=[];let offset=0;const stamp=dosStamp();for(const [name,content] of entries){const n=encoder.encode(name),body=encoder.encode(content),crc=crc32(body),header=joinBytes([u32(0x04034b50),u16(20),u16(0x0800),u16(0),u16(stamp.time),u16(stamp.date),u32(crc),u32(body.length),u32(body.length),u16(n.length),u16(0),n]);local.push(header,body);central.push(joinBytes([u32(0x02014b50),u16(20),u16(20),u16(0x0800),u16(0),u16(stamp.time),u16(stamp.date),u32(crc),u32(body.length),u32(body.length),u16(n.length),u16(0),u16(0),u16(0),u16(0),u32(0),u32(offset),n]));offset+=header.length+body.length}const directory=joinBytes(central),end=joinBytes([u32(0x06054b50),u16(0),u16(0),u16(entries.length),u16(entries.length),u32(directory.length),u32(offset),u16(0)]);return joinBytes([...local,directory,end])}
function cell(ref,value,numeric=false){return numeric?`<c r="${ref}"><v>${Number(value)}</v></c>`:`<c r="${ref}" t="inlineStr"><is><t>${xml(value)}</t></is></c>`}
function workbookBlob(){const headers=['Annotation name','Image name','Reviewed at','Checked issues','Add <unreadable> token later?'],records=Object.entries(state.reviews).sort((a,b)=>a[1].reviewed_at.localeCompare(b[1].reviewed_at));let sheet=`<row r="1">${headers.map((v,i)=>cell(String.fromCharCode(65+i)+'1',v)).join('')}</row>`;records.forEach(([id,r],index)=>{const s=data.samples.find(x=>x.id===id),row=index+2;sheet+=`<row r="${row}">${cell('A'+row,s?.annotation_name||id)}${cell('B'+row,s?.image_name||'')}${cell('C'+row,r.reviewed_at)}${cell('D'+row,r.checked_issues,true)}${cell('E'+row,r.add_unreadable?'Yes':'No')}</row>`});const last=Math.max(1,records.length+1),declaration='<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',entries=[['[Content_Types].xml',declaration+'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>'],['_rels/.rels',declaration+'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'],['xl/workbook.xml',declaration+'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Review log" sheetId="1" r:id="rId1"/></sheets></workbook>'],['xl/_rels/workbook.xml.rels',declaration+'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>'],['xl/worksheets/sheet1.xml',declaration+`<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews><cols><col min="1" max="2" width="55" customWidth="1"/><col min="3" max="3" width="26" customWidth="1"/><col min="4" max="5" width="25" customWidth="1"/></cols><sheetData>${sheet}</sheetData><autoFilter ref="A1:E${last}"/></worksheet>`]];return new Blob([zip(entries)],{type:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'})}
function downloadWorkbook(blob){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='qwen_prediction_review.xlsx';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
async function saveWorkbook(){const blob=workbookBlob();if('showSaveFilePicker'in window){try{workbookHandle??=await window.showSaveFilePicker({suggestedName:'qwen_prediction_review.xlsx',types:[{description:'Excel workbook',accept:{'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':['.xlsx']}}]});const writable=await workbookHandle.createWritable();await writable.write(blob);await writable.close();return'Workbook updated.'}catch(error){if(error?.name==='AbortError')return'Review recorded in this browser; XLSX save was cancelled.'}}downloadWorkbook(blob);return'Updated workbook downloaded.'}
async function finishReview(s){const button=document.getElementById('finish');button.disabled=true;button.textContent='Saving…';const d=draft(s);state.reviews[s.id]={reviewed_at:new Date().toISOString(),checked_issues:d.checked.length,add_unreadable:Boolean(d.unreadable)};delete state.drafts[s.id];saveState();notice=await saveWorkbook();currentId=nextOpen(s.id);render();scrollTo({top:0,behavior:'smooth'})}
if(data.warnings.length)document.getElementById('warnings').innerHTML=`<details class="warning"><summary>${data.warnings.length} loading warnings</summary><ul>${data.warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></details>`;render();
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
