"""Build a local HTML workspace for reviewing prediction/annotation differences.

The generated page has no runtime dependencies.  It can be opened directly from
disk, keeps review decisions in browser local storage, and exports those decisions
as CSV.  Annotation edits themselves are deliberately delegated to the editor so
that the original JSON formatting and metadata are preserved.
"""

from __future__ import annotations

import argparse
import json
import os
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
SPLITS = ("train", "val")
MISSING = object()


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


def compare_documents(
    annotation: dict[str, Any],
    prediction: dict[str, Any],
    *,
    normalization: NormalizationConfig | None = None,
) -> tuple[Difference, ...]:
    """Return meaningful leaf differences in stable JSON-path order."""
    config = normalization or NormalizationConfig()
    expected = _flatten_scalars(annotation)
    predicted = _flatten_scalars(prediction)
    differences: list[Difference] = []

    for path in sorted(expected.keys() | predicted.keys()):
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


def load_review_data(prediction_root: Path, dataset_root: Path) -> ReviewData:
    samples: list[ReviewSample] = []
    warnings: list[str] = []
    prediction_count = 0
    exact_count = 0

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

            differences = compare_documents(annotation, prediction)
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


def write_review_html(data: ReviewData, output_path: Path, dataset_root: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable_samples = []
    for sample in data.samples:
        serializable_samples.append(
            {
                "id": sample.id,
                "name": sample.prediction_path.stem,
                "split": sample.split,
                "image": _relative_url(sample.image_path, output_path.parent),
                "annotation_path": _display_path(sample.annotation_path),
                "prediction_path": _display_path(sample.prediction_path),
                "annotation_editor_url": _vscode_url(sample.annotation_path),
                "prediction_editor_url": _vscode_url(sample.prediction_path),
                "differences": [asdict(difference) for difference in sample.differences],
            }
        )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": str(dataset_root.resolve()),
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
) -> tuple[ReviewData, Path]:
    prediction_root = prediction_root.resolve()
    dataset_root = dataset_root.resolve()
    output_path = (output_path or prediction_root / "review.html").resolve()
    data = load_review_data(prediction_root, dataset_root)
    write_review_html(data, output_path, dataset_root)
    return data, output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data, output_path = build_review(
        prediction_root=args.prediction_root,
        dataset_root=args.dataset_root,
        output_path=args.output,
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
    :root{color-scheme:light;--ink:#172026;--muted:#66737b;--line:#d7dfe3;--paper:#fff;--bg:#f2f5f6;--blue:#1769aa;--red:#b42318;--red-bg:#fff1f0;--amber:#a15c00;--amber-bg:#fff8e8;--green:#087443;--soft:#eef3f5}
    *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif} a{color:var(--blue)}
    header{position:sticky;top:0;z-index:5;padding:13px 22px;background:rgba(255,255,255,.97);border-bottom:1px solid var(--line);backdrop-filter:blur(8px)}
    h1{font-size:21px;margin:0 0 2px} h2{font-size:17px;margin:0} .muted{color:var(--muted)} .toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-top:11px}
    input,select,button{font:inherit;min-height:37px;padding:7px 10px;border:1px solid #abb8bf;border-radius:6px;background:white} button{cursor:pointer} button:hover{background:var(--soft)} input:focus,select:focus,button:focus{outline:2px solid var(--blue);outline-offset:1px} #search{min-width:280px;flex:1}
    main{max-width:1640px;margin:auto;padding:17px 22px 60px}.stats{display:grid;grid-template-columns:repeat(4,minmax(130px,1fr));gap:8px;margin-bottom:12px}.stat{padding:9px 12px;border:1px solid var(--line);border-radius:7px;background:white}.stat b{display:block;font-size:20px}
    .nav{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:10px 0}.nav-actions,.links,.review-actions{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.card{overflow:hidden;background:white;border:1px solid var(--line);border-radius:8px}.head{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 15px;border-bottom:1px solid var(--line)}.head>div:first-child{min-width:0;overflow-wrap:anywhere}.count{flex:none;padding:4px 9px;color:white;background:var(--red);border-radius:999px;font-weight:700}
    .layout{display:grid;grid-template-columns:minmax(360px,42%) minmax(0,1fr);min-height:560px}.scan{padding:12px;background:#e7ecef;border-right:1px solid var(--line)}.scan img{position:sticky;top:150px;width:100%;max-height:calc(100vh - 195px);object-fit:contain;background:white}.links{margin-top:9px}.details{min-width:0;padding:13px 15px 20px;overflow:auto}
    table{width:100%;border-collapse:collapse;table-layout:fixed}th,td{padding:7px 8px;border:1px solid var(--line);text-align:left;vertical-align:top;overflow-wrap:anywhere}th{position:sticky;top:0;background:var(--soft);font-size:12px}.field{width:29%}.decision{width:155px}.value{white-space:pre-wrap}.different{background:var(--red-bg)}.formatting{background:var(--amber-bg)}.kind{display:inline-block;margin-top:4px;padding:2px 6px;border-radius:999px;background:#fbd5d2;color:#7a271a;font-size:11px}.kind.formatting_only{background:#fde9b4;color:#784c00}.similarity{font-size:11px;color:var(--muted);margin-top:3px}
    .review-actions{justify-content:space-between;margin-top:13px;padding-top:13px;border-top:1px solid var(--line)}.done{color:var(--green);font-weight:700}.empty{padding:55px;text-align:center;color:var(--muted)}details.warning{margin-bottom:11px;padding:10px 14px;background:white;border:1px solid var(--line);border-radius:7px}summary{cursor:pointer;font-weight:650}
    @media(max-width:900px){header{position:static}main{padding:12px}.stats{grid-template-columns:repeat(2,1fr)}.layout{grid-template-columns:1fr}.scan{border-right:0;border-bottom:1px solid var(--line)}.scan img{position:static;max-height:75vh}#search{min-width:100%}.decision{width:125px}.field{width:24%}}
  </style>
</head>
<body>
<header><h1>Qwen prediction review</h1><div id="subtitle" class="muted"></div><div class="toolbar">
  <input id="search" type="search" placeholder="Search document, field, annotation, or prediction">
  <select id="split"><option value="">Train + validation</option><option value="train">Train</option><option value="val">Validation</option></select>
  <select id="status"><option value="open">Open differences</option><option value="all">All differing documents</option><option value="complete">Fully reviewed</option><option value="label_error">Contains label error</option><option value="prediction_error">Contains prediction error</option><option value="formatting_only">Formatting-only deviations</option></select>
  <button id="export">Export decisions CSV</button>
</div></header>
<main><div id="stats" class="stats"></div><div id="warnings"></div><div id="nav" class="nav"></div><div id="content"></div></main>
<script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script>
const data=JSON.parse(document.getElementById('review-data').textContent);
const namespace='qwenPredictionReview:'+data.dataset;let decisions=load(),currentId=data.samples[0]?.id||'';
const labels={value_mismatch:'value mismatch',prediction_only:'only in prediction',annotation_only:'missing from prediction',formatting_only:'formatting / representation'};
const options=[['','Unreviewed'],['label_error','Label error'],['prediction_error','Prediction error'],['both_unclear','Both / unclear'],['acceptable','Acceptable difference']];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function load(){try{return JSON.parse(localStorage.getItem(namespace)||'{}')}catch{return {}}} function save(){localStorage.setItem(namespace,JSON.stringify(decisions))}
function answer(s,d){return decisions[s.id]?.[d.path]?.decision||''} function complete(s){return s.differences.every(d=>answer(s,d))}
function filtered(){const q=document.getElementById('search').value.toLowerCase(),split=document.getElementById('split').value,status=document.getElementById('status').value;return data.samples.filter(s=>{const answers=s.differences.map(d=>answer(s,d)),hay=JSON.stringify(s).toLowerCase();return(!q||hay.includes(q))&&(!split||s.split===split)&&(status==='all'||status==='open'&&!complete(s)||status==='complete'&&complete(s)||status==='formatting_only'&&s.differences.every(d=>d.kind==='formatting_only')||answers.includes(status))})}
function stats(){const fields=data.samples.flatMap(s=>s.differences),reviewed=fields.filter(d=>data.samples.some(s=>s.differences.includes(d)&&answer(s,d))).length;document.getElementById('stats').innerHTML=[["Predictions",data.prediction_count],["Differing documents",data.samples.length],["Field deviations",fields.length],["Reviewed deviations",reviewed]].map(([k,v])=>`<div class="stat"><span class="muted">${k}</span><b>${v}</b></div>`).join('')}
function selectHtml(s,d){const selected=answer(s,d);return `<select class="decision-select" data-path="${esc(d.path)}" aria-label="Decision for ${esc(d.path)}">${options.map(([v,l])=>`<option value="${v}" ${v===selected?'selected':''}>${l}</option>`).join('')}</select>`}
function rows(s){return s.differences.map(d=>`<tr><td><code>${esc(d.path)}</code><br><span class="kind ${d.kind}">${esc(labels[d.kind])}</span>${d.similarity!==null&&d.kind==='value_mismatch'?`<div class="similarity">${Math.round(d.similarity*100)}% similar</div>`:''}</td><td class="value ${d.kind==='formatting_only'?'formatting':'different'}">${esc(d.annotation)}</td><td class="value ${d.kind==='formatting_only'?'formatting':'different'}">${esc(d.prediction)}</td><td>${selectHtml(s,d)}</td></tr>`).join('')}
function render(){const queue=filtered();if(!queue.some(s=>s.id===currentId))currentId=queue[0]?.id||'';const i=queue.findIndex(s=>s.id===currentId),s=queue[i];stats();document.getElementById('nav').innerHTML=queue.length?`<b>${i+1} of ${queue.length} matching documents</b><div class="nav-actions"><button id="prev" ${i<=0?'disabled':''}>← Previous</button><button id="nextOpen">Next open</button><button id="next" ${i>=queue.length-1?'disabled':''}>Next →</button></div>`:'';if(!s){document.getElementById('content').innerHTML='<div class="card empty">No documents match the current filters.</div>';return}const remaining=s.differences.filter(d=>!answer(s,d)).length;document.getElementById('content').innerHTML=`<article class="card"><div class="head"><div><h2>${esc(s.name)}</h2><div class="muted">${esc(s.split)} · ${esc(s.annotation_path)}</div></div><div>${complete(s)?'<span class="done">Reviewed ✓</span> ':''}<span class="count">${s.differences.length} deviations</span></div></div><div class="layout"><div class="scan"><a href="${s.image}" target="_blank"><img src="${s.image}" alt="Scan ${esc(s.name)}"></a><div class="links"><a href="${s.image}" target="_blank">Open full-size scan ↗</a><a href="${s.annotation_editor_url}">Edit annotation JSON</a><a href="${s.prediction_editor_url}">Open prediction JSON</a></div></div><div class="details"><table><thead><tr><th class="field">Field</th><th>Annotation</th><th>Qwen prediction</th><th class="decision">Assessment</th></tr></thead><tbody>${rows(s)}</tbody></table><div class="review-actions"><span class="${remaining?'muted':'done'}">${remaining?remaining+' deviations still unreviewed':'All deviations reviewed ✓'}</span><div><button id="allPrediction">Mark open as prediction errors</button> <button id="clear">Clear this document</button></div></div></div></div></article>`;
document.querySelectorAll('.decision-select').forEach(el=>el.onchange=()=>setDecision(s,el.dataset.path,el.value));document.getElementById('prev').onclick=()=>go(queue,i-1);document.getElementById('next').onclick=()=>go(queue,i+1);document.getElementById('nextOpen').onclick=()=>nextOpen(s.id);document.getElementById('allPrediction').onclick=()=>{s.differences.forEach(d=>{if(!answer(s,d))put(s,d.path,'prediction_error')});save();render()};document.getElementById('clear').onclick=()=>{delete decisions[s.id];save();render()}}
function put(s,path,decision){decisions[s.id]??={};if(decision)decisions[s.id][path]={decision,reviewed_at:new Date().toISOString()};else delete decisions[s.id][path]}
function setDecision(s,path,decision){put(s,path,decision);save();render()} function go(q,i){if(q[i]){currentId=q[i].id;render();scrollTo({top:0,behavior:'smooth'})}}
function nextOpen(id){const start=data.samples.findIndex(s=>s.id===id);for(let n=1;n<=data.samples.length;n++){const s=data.samples[(start+n)%data.samples.length];if(!complete(s)){currentId=s.id;document.getElementById('status').value='open';render();scrollTo({top:0,behavior:'smooth'});return}}}
function csv(v){return '"'+String(v??'').replaceAll('"','""')+'"'} function download(name,text){const a=document.createElement('a'),blob=new Blob([text],{type:'text/csv;charset=utf-8'});a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),0)}
document.getElementById('export').onclick=()=>{const rows=[['sample_id','split','field','annotation','prediction','difference_kind','decision','reviewed_at','annotation_path','image_path']];data.samples.forEach(s=>s.differences.forEach(d=>{const r=decisions[s.id]?.[d.path]||{};rows.push([s.name,s.split,d.path,d.annotation,d.prediction,d.kind,r.decision||'unreviewed',r.reviewed_at||'',s.annotation_path,s.image])}));download('qwen_prediction_review.csv','\ufeff'+rows.map(r=>r.map(csv).join(',')).join('\r\n'))};
['search','split','status'].forEach(id=>document.getElementById(id).addEventListener(id==='search'?'input':'change',render));document.addEventListener('keydown',e=>{if(['INPUT','SELECT'].includes(document.activeElement.tagName))return;if(e.key==='j'||e.key==='ArrowRight'){const q=filtered(),i=q.findIndex(s=>s.id===currentId);go(q,i+1)}if(e.key==='k'||e.key==='ArrowLeft'){const q=filtered(),i=q.findIndex(s=>s.id===currentId);go(q,i-1)}});
if(data.warnings.length)document.getElementById('warnings').innerHTML=`<details class="warning"><summary>${data.warnings.length} loading warnings</summary><ul>${data.warnings.map(w=>`<li>${esc(w)}</li>`).join('')}</ul></details>`;document.getElementById('subtitle').textContent=`${data.samples.length} documents with deviations · ${data.exact_count} without meaningful deviations · generated ${data.generated_at}`;render();
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())
