"""Curated figures for the concluded CMR experiments.

Reproduce with ``python -m src.report_plots.experiment_suite``. Evaluations are
recomputed from predictions, and saved only alongside the figures.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
from html import escape
import json
from pathlib import Path
import textwrap
from typing import Sequence
import zipfile

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np

from src.eval_suite.paths import REPO_ROOT
from src.eval_suite.results_table import evaluate_results_folder, write_csv, _summary_for
from .data import RunResult, load_results
from .plots import Plot, RC
from .statistics import mean_stages, paired_bootstrap, pareto_indices, runtime_samples, summarize, timing_values

COLORS = {"End-to-End": "#0072B2", "Parser + LLM": "#D55E00", "Hybrid": "#009E73"}
SUBSETS = ("overall", "default", "challenge")
SUBCOLORS = {"overall": "#283747", "default": "#0072B2", "challenge": "#D55E00"}
GROUPS = tuple(COLORS)
RC_SUITE = {**RC, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
            "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9}


def _new(title, subtitle, *, rows=1, cols=1, size=(12, 6), left=.09, bottom=.26, wspace=.4):
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    axes = np.asarray(fig.subplots(rows, cols, squeeze=False))
    fig.subplots_adjust(left=left, right=.97, top=.82, bottom=bottom, wspace=wspace, hspace=.42)
    fig.suptitle(title, x=.025, y=.98, ha="left", va="top", fontsize=17, weight="bold", color="#203040")
    fig.text(.025, .91, subtitle, ha="left", va="top", fontsize=9.5, color="#52606D")
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.7)
    return fig, axes


def _foot(fig, text):
    fig.text(.025, .035, textwrap.fill(text, 175), fontsize=8, color="#52606D", va="bottom", linespacing=1.5)


def _bottom_legend(fig, ax, *, center=.6, columns=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(center, .095),
               ncol=columns, frameon=False, fontsize=9)


def _score(run, subset="overall"):
    return 100 * run.metric("field_f1", subset)


def _resources(run):
    selected = runtime_samples(run)
    memory = runtime_samples(run, memory_kind="process")
    return {**summarize(timing_values(selected, "total")),
            "peak_gib": max(s.memory["process"] for s in memory.samples),
            "sample_ids": [s.sample_id for s in selected.samples],
            "seconds": timing_values(selected, "total"),
            "memory_gib": [s.memory["process"] for s in memory.samples],
            "gpu_ids": sorted({d for s in memory.samples for d in s.devices})}


def _quality_data(runs):
    return [{"run": r.key, "label": r.label, "group": r.group,
             **{subset: r.metrics[subset] for subset in SUBSETS}} for r in runs]


def _plot(fig, recipe, runs, data, **options):
    return Plot(fig, recipe, runs, data, {"campaign": "Latest available saved inference per experiment; details in input_audit.json",
                                        "testset": "30 CMR pages: 15 default, 15 challenge", **options})


def _load_inputs(out: Path) -> list[RunResult]:
    """Fresh evaluation snapshots keep older comparison tables out of the selection."""
    specifications = [
        ("end_to_end", "e2e", "End-to-End", REPO_ROOT / "results/New/qwen-measurements-20261022-112602"),
        ("parser", "parser", "Parser + LLM", REPO_ROOT / "results/New/Parser_based"),
        ("hybrid", "hybrid", "Hybrid", REPO_ROOT / "results/New/Hybrid"),
        ("full_tuning", "e2e", "End-to-End", REPO_ROOT / "results/New/Finetuning_experiment/f1-f4-testset-20261012_125819"),
    ]
    runs = []
    audit = []
    for family, namespace, group, source in specifications:
        if family == "full_tuning":
            evaluations = []
            for directory in sorted(source.iterdir()):
                if directory.is_dir() and (directory / "test").is_dir():
                    item = evaluate_results_folder(directory / "test")[0]
                    evaluations.append(replace(item, run_name=directory.name))
        else:
            evaluations = evaluate_results_folder(source)
        destination = out / "data" / family
        reports = destination / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        write_csv(evaluations, destination / "evaluation_table.csv")
        report_paths = {}
        for evaluation in evaluations:
            report = evaluation.report.to_dict()
            for subset in SUBSETS:
                block = report["overall"] if subset == "overall" else report["subsets"][subset]
                block["summary"] = _summary_for(evaluation, subset)
            report_path = reports / f"{evaluation.run_name}.json"
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            report_paths[evaluation.run_name] = report_path
            records = evaluation.manifest_records or {}
            expected = {s.sample_id for s in evaluation.report.overall.samples}
            if len(expected) != 30 or records.keys() != expected:
                raise ValueError(f"{evaluation.run_name}: need exactly the 30 annotated document IDs in its manifest")
            manifest = evaluation.run_directory / "inference_manifest.jsonl"
            model_settings = sorted({json.dumps(r.get("generation", r).get("model", {}), sort_keys=True) for r in records.values()})
            # Hash original inputs so every figure can be traced to an exact snapshot.
            prediction_hashes = {str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in evaluation.run_directory.rglob("*.json")
                                 if not p.name.endswith(".fusion.json")}
            audit.append({"key": f"{namespace}::{evaluation.run_name}", "group": group, "family": family,
                          "source": str(evaluation.run_directory), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                          "prediction_sha256": prediction_hashes, "model_settings": [json.loads(x) for x in model_settings],
                          "statuses": dict(Counter(r.get("status") for r in records.values())),
                          "sample_ids": sorted(expected), "report": str(report_path)})
        loaded = load_results(destination / "evaluation_table.csv", namespace=namespace, reports=report_paths)
        for run in loaded:
            run.group = group
        runs.extend(loaded)
        print(f"Evaluated {len(loaded)} {family} runs", flush=True)
    if len({r.key for r in runs}) != len(runs):
        raise ValueError("Repeated run identity in figure selection")
    reference = set(runs[0].samples)
    if any(set(r.samples) != reference for r in runs):
        raise ValueError("Figure inputs do not share the same 30 documents")
    (out / "input_audit.json").write_text(json.dumps({
        "generator": {"module": "src.report_plots.experiment_suite", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "runs": audit, "source_policy": "Fresh evaluations of current predictions. Historical snapshots are not substituted for recent inference.",
        "full_tuning_campaign": "F1-F4: 2026-10-12 inference; other End-to-End results: 2026-10-22 inference.",
        "parser_note": "All 15 parser runs evaluated, including soft/F2 absent from the older 14-run table.",
        "hybrid_note": "Includes the successful image20 single-document OCR OOM rerun already merged into its canonical manifest.",
        "comparability": "Exploratory seed-42 experiments; best observed configurations are descriptive, not an unbiased model-selection estimate. Hardware/time scopes are not normalized.",
    }, indent=2) + "\n")
    return runs


def _overview(runs, representatives):
    trained = [r for r in runs if "-base-" not in r.key]
    fig, axes = _new("Three routes to structured document extraction",
                     "Configuration landscape and the strongest observed 9B setting in each category", cols=2, size=(12, 6), wspace=.62)
    ax, bx = axes[0]
    for i, group in enumerate(GROUPS):
        members = sorted([r for r in trained if r.group == group], key=_score)
        values = [_score(r) for r in members]
        jitter = .10 * np.sin(np.arange(len(values))*2.4)
        ax.scatter(values, i+jitter, s=34, alpha=.6, color=COLORS[group], edgecolors="white", linewidths=.4)
        best = max(members, key=_score)
        ax.scatter(_score(best), i, marker="*", s=200, color=COLORS[group], edgecolors="#203040", linewidths=.6, zorder=5)
        ax.annotate(f"{_score(best):.2f}", (_score(best), i), xytext=(6, -17), textcoords="offset points", weight="bold", fontsize=10)
    ax.set_yticks(range(3), [f"{g}\n{sum(r.group==g for r in trained)} settings" for g in GROUPS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Overall micro field F1 (%)")
    ax.set_title("A   Every fine-tuned configuration", loc="left")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x")
    labels = ["End-to-End\nL2 multimodal · BF16", "Parser + LLM\nHard fusion · L1 BF16", "Hybrid\nHard fusion · NF4"]
    y = np.arange(3)
    for j, subset in enumerate(SUBSETS):
        values = [_score(r, subset) for r in representatives]
        bars = bx.barh(y+(j-1)*.23, values, height=.21, color=SUBCOLORS[subset], label=subset.title())
        bx.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    bx.set_yticks(y, labels)
    bx.invert_yaxis()
    bx.set_xlim(0, 108)
    bx.set_xticks([0,25,50,75,100])
    bx.set_xlabel("Micro field F1 (%)")
    bx.set_title("B   Common 9B comparison", loc="left")
    bx.grid(axis="y", visible=False)
    bx.grid(axis="x")
    _bottom_legend(fig,bx,center=.74)
    _foot(fig, "Dots are configurations, not repeated trials; stars mark category maxima. The End-to-End maximum uses 27B; panel B fixes model size at 9B. All scores use the same 30 pages.")
    return _plot(fig,"category_overview",trained,_quality_data(trained),representatives=[r.key for r in representatives])


def _scale(base, tuned):
    fig, axes = _new("End-to-End: adaptation, model size, and cost",
                     "NF4 base models and QLoRA adapters · latest measured inference campaign", cols=3, size=(12,5.5), wspace=.43)
    x=np.arange(3)
    ax,bx,cx=axes[0]
    for offset, values, color, label in [(-.18,[_score(r) for r in base],"#C2CDD4","Base"),
                                          (.18,[_score(r) for r in tuned],COLORS["End-to-End"],"QLoRA")]:
        bars=ax.bar(x+offset, values, width=.33,color=color,label=label)
        ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
    ax.set_ylim(0,103);ax.set_ylabel("Overall field F1 (%)");ax.legend(frameon=False)
    resource=[_resources(r) for r in tuned]
    for axis,key,label in [(bx,"median","Median inference (s/page)"),(cx,"peak_gib","Peak process VRAM (GiB)")]:
        values=[r[key] for r in resource]
        bars=axis.bar(x,values,color=COLORS["End-to-End"],width=.55)
        axis.bar_label(bars,fmt="%.1f",padding=3)
        axis.set_ylim(0,max(values)*1.18);axis.set_ylabel(label)
    for axis,title in zip(axes[0],["A   Structured extraction","B   QLoRA latency","C   QLoRA memory"]):
        axis.set_xticks(x,["4B","9B","27B"]);axis.set_title(title,loc="left");axis.set_xlabel("Model size")
    _foot(fig,"Base-model F1 is zero under the target extraction contract; this does not mean no text was generated. Resource bars describe the fine-tuned runs, with all 30 pages retained and no warmup exclusion.")
    return _plot(fig,"end_to_end_scale",[*base,*tuned],{"quality":_quality_data([*base,*tuned]),"tuned_resources":resource})


def _tuning(runs):
    fig,axes=_new("End-to-End: where to spend adaptation capacity",
                   "Qwen3.5-9B · adapter and full-tuning recipes evaluated on identical test pages",cols=1,size=(11,6.5),left=.35)
    ax=axes[0,0];y=np.arange(len(runs))
    for j,subset in enumerate(SUBSETS):
        values=[_score(r,subset) for r in runs]
        bars=ax.barh(y+(j-1)*.23,values,height=.21,color=SUBCOLORS[subset],label=subset.title())
        ax.bar_label(bars,fmt="%.1f",fontsize=8,padding=2)
    ax.set_yticks(y,[r.label for r in runs]);ax.invert_yaxis();ax.set_xlim(0,108);ax.set_xticks([0,25,50,75,100])
    ax.grid(axis="y",visible=False);ax.grid(axis="x");ax.set_xlabel("Micro field F1 (%)")
    ax.axhline(3.5,color="#BFC8CE",lw=.8)
    _bottom_legend(fig,ax,center=.65)
    _foot(fig,"Q1/Q2/L1/L2: 22 Oct inference; full-tuning F1–F4: latest available 12 Oct inference. Full tuning uses a different learning-rate recipe; these are observed configurations, not an isolated causal test of trainable parameter count.")
    return _plot(fig,"end_to_end_tuning",runs,_quality_data(runs))


def _precision(pairs):
    runs=[r for pair in pairs for r in pair]
    fig,axes=_new("Inference precision changes the same adapter's behavior",
                   "Two BF16-trained End-to-End adapters, each loaded once in NF4 and once in BF16",cols=3,size=(12,5.4))
    resource={r.key:_resources(r) for r in runs}
    data=[]
    for ax,title,key,ylabel in zip(axes[0],["A   Extraction quality","B   Inference latency","C   Process memory"],
                                 ["f1","median","peak_gib"],["Overall field F1 (%)","Median time (s/page)","Peak VRAM (GiB)"]):
        for j,precision in enumerate(["NF4","BF16"]):
            values=[_score(pair[j]) if key=="f1" else resource[pair[j].key][key] for pair in pairs]
            bars=ax.bar(np.arange(2)+(j-.5)*.34,values,width=.32,color=["#56B4E9",COLORS["End-to-End"]][j],label=precision)
            ax.bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
            data.extend({"metric":key,"run":pair[j].key,"value":v} for pair,v in zip(pairs,values))
        ax.set_xticks([0,1],["L1: language LoRA","L2: multimodal LoRA"],rotation=15,ha="right")
        ax.set_title(title,loc="left");ax.set_ylabel(ylabel);ax.set_ylim(0,105 if key=="f1" else max(resource[r.key][key] for r in runs)*1.2)
    axes[0,0].legend(frameon=False,loc="upper left")
    _foot(fig,"Within each pair the adapter path is identical; only base-weight loading precision differs. Do not confuse these NF4 reloads of BF16-trained adapters with the separately NF4-trained Q1/Q2 recipes.")
    return _plot(fig,"inference_precision",runs,{"comparisons":data,"resources":resource})


def _resolution(runs):
    x=np.arange(4);labels=["Low\n1.4 MP","Medium\n2.8 MP","High\n4.2 MP","Native\n5.6 MP"]
    fig,axes=_new("More image pixels do not guarantee better extraction",
                   "End-to-End 9B NF4 QLoRA · matched training and inference resolution presets",cols=3,size=(12,5.6))
    ax,bx,cx=axes[0]
    for subset in SUBSETS:
        values=[_score(r,subset) for r in runs]
        ax.plot(x,values,marker="o",lw=1.6,color=SUBCOLORS[subset],label=subset.title())
        if subset=="overall":
            for a,b in zip(x,values):ax.annotate(f"{b:.2f}",(a,b),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8)
    ax.set_ylim(0,103);ax.set_ylabel("Micro field F1 (%)");ax.legend(frameon=False,loc="lower left")
    resources=[_resources(r) for r in runs]
    med=[r["median"] for r in resources];q=np.asarray([np.quantile(r["seconds"],[.25,.75]) for r in resources])
    bx.fill_between(x,q[:,0],q[:,1],color=COLORS["End-to-End"],alpha=.15,label="Document IQR")
    bx.plot(x,med,marker="o",color=COLORS["End-to-End"],label="Median")
    bx.set_ylim(0,max(q[:,1])*1.2);bx.set_ylabel("Inference time (s/page)");bx.legend(frameon=False,loc="lower left")
    mem=[r["peak_gib"] for r in resources]
    cx.plot(x,mem,marker="s",color=COLORS["End-to-End"])
    for a,b in zip(x,mem):cx.annotate(f"{b:.2f}",(a,b),xytext=(0,7),textcoords="offset points",ha="center",fontsize=8)
    cx.set_ylim(0,max(mem)*1.3);cx.set_ylabel("Peak process VRAM (GiB)")
    for axis,title in zip(axes[0],["A   Quality","B   Latency","C   Memory"]):
        axis.set_xticks(x,labels);axis.set_title(title,loc="left")
    _foot(fig,"Pixel values are maximum budgets, not forced image sizes. Separate training runs use the corresponding inference preset. Shading shows the latency distribution across documents, not a confidence interval for quality.")
    return _plot(fig,"resolution",runs,{"quality":_quality_data(runs),"resources":resources,"maximum_pixels":[1400000,2800000,4200000,5600000]})


def _parser_matrix(runs):
    modes=["separate","hard","soft"];recipes=["q1","l1","l2","f1","f2"]
    fig,axes=_new("Parser + LLM: fusion and tuning interact",
                   "All 15 completed configurations · the same structured extraction schema and 30 test pages",cols=3,size=(12,6),left=.22,wspace=.23)
    matrices={}
    for ax,subset,letter in zip(axes[0],SUBSETS,"ABC"):
        matrix=np.asarray([[_score(next(r for r in runs if f"fusion-{mode}-{recipe}-" in r.key),subset) for mode in modes] for recipe in recipes])
        image=ax.imshow(matrix,cmap="cividis",vmin=0,vmax=100,aspect="auto")
        matrices[subset]=matrix.tolist()
        for i in range(5):
            for j in range(3):ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",color="white" if matrix[i,j]<45 else "#111111",fontsize=10)
        ax.set_xticks(range(3),["Separate","Hard","Soft"])
        ax.set_yticks(range(5),["Q1 · NF4 LoRA\nLR 5e−5","L1 · BF16 LoRA\nLR 5e−5","L2 · BF16 LoRA\nLR 1e−5","F1 · BF16 full\nLR 1e−6","F2 · BF16 full\nLR 5e−6"] if subset=="overall" else [""]*5)
        ax.set_title(f"{letter}   {subset.title()}",loc="left");ax.grid(False)
    fig.colorbar(image,ax=list(axes[0]),fraction=.022,pad=.025,label="Micro field F1 (%)")
    _foot(fig,"Rows change optimization/base-weight precision; columns change evidence representation. All cells share the same 0–100% color scale. The freshly evaluated matrix includes soft/F2, which was missing from the older 14-run table.")
    return _plot(fig,"parser_matrix",runs,{"modes":modes,"recipes":recipes,"matrices":matrices})


def _hybrid_inputs(runs):
    fig,axes=_new("Hybrid input: combine visual and parser evidence",
                   "9B NF4 QLoRA · image-only reference and three Hybrid evidence variants",cols=3,size=(12,5.5),left=.2,wspace=.2)
    for ax,subset,letter in zip(axes[0],SUBSETS,"ABC"):
        values=[_score(r,subset) for r in runs]
        bars=ax.barh(range(4),values,color=[COLORS[r.group] for r in runs],height=.57)
        ax.bar_label(bars,fmt="%.2f",padding=3,fontsize=9)
        ax.set_yticks(range(4),[r.label for r in runs] if subset=="overall" else [""]*4)
        ax.invert_yaxis();ax.set_xlim(0,108);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel("Field F1 (%)")
        ax.grid(axis="y",visible=False);ax.grid(axis="x");ax.set_title(f"{letter}   {subset.title()}",loc="left")
    _foot(fig,"The three Hybrid runs compare evidence inputs under a shared task contract. The image-only reference uses its historical prompt; adding evidence is therefore not the only difference across the reference/Hybrid boundary. OCR/hard use chunked training loss; layout uses standard loss.")
    return _plot(fig,"hybrid_inputs",runs,_quality_data(runs))


def _dropout(runs):
    fig,axes=_new("Modality dropout under full-input evaluation",
                   "All variants receive both image and hard-fusion evidence at test time",cols=2,size=(12,5.8),left=.2,wspace=.45)
    ax,bx=axes[0];y=np.arange(4)
    for j,subset in enumerate(SUBSETS):
        values=[_score(r,subset) for r in runs]
        bars=ax.barh(y+(j-1)*.23,values,height=.21,color=SUBCOLORS[subset],label=subset.title())
        ax.bar_label(bars,fmt="%.1f",padding=2,fontsize=8)
    ax.set_yticks(y,[r.label for r in runs]);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel("Micro field F1 (%)");ax.set_title("A   Observed extraction quality",loc="left")
    ax.grid(axis="y",visible=False);ax.grid(axis="x");_bottom_legend(fig,ax,center=.38)
    intervals=[]
    for i,r in enumerate(runs[1:]):
        interval=paired_bootstrap(runs[0],r,resamples=5000,seed=42)
        intervals.append({"run":r.key,**interval})
        bx.plot([100*interval["low"],100*interval["high"]],[i,i],color=COLORS["Hybrid"],lw=2)
        bx.scatter(100*interval["difference"],i,s=55,color=COLORS["Hybrid"],zorder=3)
    bx.set_yticks(range(3),[r.label for r in runs[1:]]);bx.invert_yaxis();bx.axvline(0,color="#687782",lw=1,ls="--")
    bx.set_xlabel("Micro F1 difference vs. no dropout (pp)");bx.set_title("B   Paired document uncertainty",loc="left")
    bx.grid(axis="y",visible=False);bx.grid(axis="x")
    _foot(fig,"Intervals are 95% paired document-bootstrap percentile intervals (5,000 draws, seed 42), not variation across training seeds. The successful image20 rerun is included. Negative point estimates alone do not establish a significant loss.")
    return _plot(fig,"dropout",runs,{"quality":_quality_data(runs),"paired_bootstrap":intervals})


def _tradeoffs(runs,highlights):
    resources={r.key:_resources(r) for r in runs}
    fig,axes=_new("Accuracy has a latency and memory price",
                   "Measured 9B configurations across all three approaches · one marker per configuration",cols=2,size=(12,6),wspace=.3)
    quality=[_score(r) for r in runs];data=[]
    for ax,resource,label,letter in zip(axes[0],["median","peak_gib"],["Median inference time (s/page)","Peak sampled process VRAM (GiB)"],"AB"):
        values=[resources[r.key][resource] for r in runs];frontier=pareto_indices(values,quality)
        for group,marker in zip(GROUPS,["o","s","^"]):
            indices=[i for i,r in enumerate(runs) if r.group==group]
            ax.scatter([values[i] for i in indices],[quality[i] for i in indices],s=48,color=COLORS[group],marker=marker,
                       alpha=.8,edgecolors="white",linewidths=.5,label=group,zorder=3)
        order=sorted(frontier,key=lambda i:values[i])
        ax.plot([values[i] for i in order],[quality[i] for i in order],ls="--",lw=1,color="#687782",zorder=1)
        # Label reference configurations and a few frontier anchors. Naming every
        # near-identical point obscures the dense high-quality region.
        annotated={i for i,r in enumerate(runs) if r.key in highlights}
        annotated.add(min(frontier,key=lambda i:values[i]))
        if resource=="median":
            annotated.update(i for i,r in enumerate(runs) if "fusion-soft-f2-" in r.key)
        else:
            annotated.update(i for i,r in enumerate(runs) if "hybrid-input-layout-" in r.key)
        for i in sorted(annotated,key=lambda i:quality[i]):
            r=runs[i]
            if resource=="median":
                dx,dy=((-80,-2) if "fusion-soft-f2-" in r.key else
                       (10,-20) if "fusion-hard-l1-" in r.key else
                       (-12,18) if "l2-lora-multimodal-bf16" in r.key else (8,16))
            else:
                dx,dy=((8,-22) if "resolution-low" in r.key else
                       (12,22) if "hybrid-input-layout-" in r.key else
                       (-10,18) if "fusion-hard-l1-" in r.key else
                       (-12,20) if "l2-lora-multimodal-bf16" in r.key else (8,16))
            ax.annotate(r.label,(values[i],quality[i]),xytext=(dx,dy),textcoords="offset points",fontsize=8,
                        ha="left" if dx>=0 else "right",arrowprops={"arrowstyle":"-","lw":.5,"color":"#8B969E"})
        ax.set_xlim(0,max(values)*1.22);ax.set_ylim(45,95);ax.set_xlabel(label);ax.set_ylabel("Overall micro field F1 (%)")
        ax.set_title(f"{letter}   Quality versus {'latency' if resource=='median' else 'memory'}",loc="left")
        if resource=="median":ax.legend(frameon=False,loc="lower right")
        data.append({"resource":resource,"points":[{"run":r.key,"value":v,"field_f1":q/100,"pareto":i in frontier}
                                                   for i,(r,v,q) in enumerate(zip(runs,values,quality))]})
    _foot(fig,"Dashed lines connect the observed Pareto frontiers; they are descriptive, not statistical dominance claims. All plotted records identify the same GPU UUID. Latency excludes model initialization and artifact saving; warmup is retained. Memory includes resident models and caches.")
    return _plot(fig,"resource_tradeoffs",runs,{"points":data,"resources":resources})


def _latency(runs):
    fig,axes=_new("Latency is a distribution, not a single number",
                   "Representative image-only, parser, and Hybrid pipelines · all 30 pages per setting",cols=2,size=(12,6),left=.27,wspace=.4)
    ax,bx=axes[0];resources=[_resources(r) for r in runs]
    import inspect
    orientation={"orientation":"horizontal"} if "orientation" in inspect.signature(ax.boxplot).parameters else {"vert":False}
    artists=ax.boxplot([r["seconds"] for r in resources],positions=range(len(runs)),widths=.5,manage_ticks=False,patch_artist=True,showfliers=False,
                       medianprops={"color":"#202B33","lw":1.5},**orientation)
    for i,(run,res,box) in enumerate(zip(runs,resources,artists["boxes"])):
        box.set_facecolor(COLORS[run.group]);box.set_alpha(.25)
        ax.scatter(res["seconds"],i+.10*np.sin(np.arange(30)*2.4),s=11,color=COLORS[run.group],alpha=.65)
        ax.scatter(res["p95"],i,marker="D",s=28,color="#203040",zorder=4)
        values=sorted(res["seconds"])
        bx.step([0,*values],[0,*np.arange(1,31)/30],where="post",color=COLORS[run.group],
                linestyle=["-","--","-.",":","-","--"][i],label=run.label,lw=1.5)
    ax.set_yticks(range(len(runs)),[r.label for r in runs]);ax.invert_yaxis();ax.set_xlim(left=0)
    ax.grid(axis="y",visible=False);ax.grid(axis="x");ax.set_xlabel("Total inference latency (s/page)");ax.set_title("A   Documents and tail latency",loc="left")
    bx.set_xlim(left=0);bx.set_ylim(0,1.02);bx.set_yticks([0,.25,.5,.75,1],["0%","25%","50%","75%","100%"])
    bx.set_xlabel("Total inference latency (s/page)");bx.set_ylabel("Cumulative fraction of documents");bx.set_title("B   Empirical completion curves",loc="left")
    bx.legend(fontsize=8,loc="lower right",frameon=False)
    _foot(fig,"Boxes: median and interquartile range; whiskers: 1.5×IQR; dots: every document; diamonds: p95 (linear quantile). Parse/schema failures remain timed because inference completed. First calls are retained; no throughput/concurrency claim is inferred.")
    return _plot(fig,"latency_distributions",runs,{r.key:res for r,res in zip(runs,resources)})


def _stage_costs(runs):
    data=[mean_stages(runtime_samples(r)) for r in runs]
    fig,axes=_new("Where the inference budget goes",
                   "Common 9B references · mean wall-clock time on the same 30 documents",cols=2,size=(12,5.8),left=.27,wspace=.35)
    ax,bx=axes[0];stages=["ocr","yolo","fusion","generation","other"]
    colors=["#56B4E9","#E69F00","#CC79A7","#0072B2","#A6B1B9"]
    y=np.arange(len(runs));left=np.zeros(len(runs))
    for stage,color in zip(stages,colors):
        values=[row.get(stage,0) for row in data]
        ax.barh(y,values,left=left,height=.6,color=color,label=stage.upper() if stage in {"ocr","yolo"} else stage.title());left+=values
    for i,value in enumerate(left):ax.text(value+.25,i,f"{value:.1f}s",va="center",fontsize=9)
    left=np.zeros(len(runs))
    for stage,color in zip(stages[:3],colors[:3]):
        values=[row.get(stage,0) for row in data]
        bx.barh(y,values,left=left,height=.6,color=color);left+=values
    for axis,title,xlabel in [(ax,"A   Entire pipeline","Mean time (s/page)"),(bx,"B   Parser stages, expanded","Mean parser time (s/page)")]:
        axis.set_yticks(y,[r.label for r in runs] if axis is ax else [""]*len(runs));axis.invert_yaxis();axis.grid(axis="y",visible=False);axis.grid(axis="x")
        axis.set_xlabel(xlabel);axis.set_title(title,loc="left");axis.set_xlim(left=0)
    ax.set_xlim(0,max(sum(row.values()) for row in data)*1.18)
    _bottom_legend(fig,ax,center=.55,columns=5)
    _foot(fig,"Generation is measured as the stage wrapper wall time. The expanded parser panel prevents small OCR/layout/fusion costs from disappearing in the total. Other is measured pipeline overhead; independent native timings are not added a second time.")
    return _plot(fig,"stage_costs",runs,[{"run":r.key,"mean_seconds":row} for r,row in zip(runs,data)])


SECTION_LABELS={"senderInformation":"Sender","consigneeInformation":"Consignee","takingOverTheGoods":"Pickup",
                "deliveryOfTheGoods":"Delivery location","sendersInstructions":"Sender instructions","carrierInformation":"Carrier",
                "successiveCarrierInformation":"Successive carrier","carrierReservationsAndObservations":"Carrier reservations",
                "documentsHandedToCarrier":"Handed-over documents","itemList[]":"Cargo items",
                "signatureOrStampOfTheSender":"Sender signature","signatureOrStampOfTheCarrier":"Carrier signature",
                "goodsReceived":"Receipt confirmation","referenceIdentificationNumber":"Reference number"}


def _section_counts(run,subset):
    result={}
    for field,metrics in run.fields[subset].items():
        name=field.split(".")[0]
        row=result.setdefault(name,np.zeros(3,dtype=int))
        row += [metrics[k] for k in ("true_positives","false_positives","false_negatives")]
    total=np.sum(list(result.values()),axis=0)
    if not np.isclose(2*total[0]/(2*total[0]+total[1]+total[2]),run.metric("field_f1",subset)):
        raise ValueError("Section aggregation did not preserve micro F1")
    return result


def _fields(runs):
    fig,axes=_new("Which parts of the document remain difficult?",
                   "Best observed 9B configurations · section scores recomputed from pooled leaf-field counts",cols=2,size=(12,8),left=.23,bottom=.22,wspace=.23)
    sections=list(SECTION_LABELS);data={}
    for ax,subset,letter in zip(axes[0],["overall","challenge"],"AB"):
        counts=[_section_counts(r,subset) for r in runs]
        matrix=np.full((len(sections),len(runs)),np.nan);support=np.zeros_like(matrix,dtype=int)
        for i,section in enumerate(sections):
            for j,items in enumerate(counts):
                tp,fp,fn=items.get(section,np.zeros(3,dtype=int));support[i,j]=tp+fn
                if tp+fn>0:matrix[i,j]=100*2*tp/(2*tp+fp+fn)
        cmap=mpl.colormaps["cividis"].copy();cmap.set_bad("#E9EDF0")
        image=ax.imshow(np.ma.masked_invalid(matrix),aspect="auto",cmap=cmap,vmin=0,vmax=100)
        for i in range(len(sections)):
            for j in range(len(runs)):
                if np.isfinite(matrix[i,j]):ax.text(j,i,f"{matrix[i,j]:.0f}",ha="center",va="center",fontsize=9,color="white" if matrix[i,j]<45 else "#111111")
        ax.set_yticks(range(len(sections)),[SECTION_LABELS[s] for s in sections] if subset=="overall" else [""]*len(sections))
        ax.set_xticks(range(3),["End-to-End\nL2 BF16","Parser + LLM\nHard L1 BF16","Hybrid\nHard NF4"],fontsize=9)
        ax.set_title(f"{letter}   {subset.title()}",loc="left");ax.grid(False)
        data[subset]={"sections":sections,"support":support.tolist(),"f1":[[None if not np.isfinite(v) else float(v/100) for v in row] for row in matrix]}
    fig.colorbar(image,ax=list(axes[0]),fraction=.025,pad=.03,label="Section micro field F1 (%)")
    _foot(fig,"Section F1 is computed from summed TP/FP/FN, not an average of field F1 scores. Grey means no populated ground-truth support. Sparse sections may rely on very few annotated values; exact support counts accompany the figure data.")
    return _plot(fig,"field_sections",runs,data)


def _paired_documents(representatives):
    e2e,parser,hybrid=representatives
    fig,axes=_new("The same documents reveal where Hybrid gains",
                   "Per-document field F1 · best observed 9B references · every point is a matched test page",cols=2,size=(12,6),wspace=.3)
    data=[]
    for ax,ref,letter in zip(axes[0],[e2e,parser],"AB"):
        ids=sorted(hybrid.samples)
        if set(ids)!=set(ref.samples):raise ValueError("Unpaired document scatter")
        rows=[]
        for subset,marker,color in [("default","o",SUBCOLORS["default"]),("challenge","^",SUBCOLORS["challenge"])]:
            selected=[key for key in ids if hybrid.samples[key]["subset"]==subset]
            x=[100*ref.samples[key]["metrics"]["f1"] for key in selected]
            y=[100*hybrid.samples[key]["metrics"]["f1"] for key in selected]
            ax.scatter(x,y,marker=marker,color=color,s=45,alpha=.8,edgecolors="white",linewidths=.5,label=subset.title(),zorder=3)
            rows.extend({"sample_id":key,"subset":subset,"reference_f1":a/100,"hybrid_f1":b/100} for key,a,b in zip(selected,x,y))
        wins=sum(row["hybrid_f1"]>row["reference_f1"]+1e-9 for row in rows)
        losses=sum(row["hybrid_f1"]<row["reference_f1"]-1e-9 for row in rows)
        ax.plot([0,100],[0,100],ls="--",color="#8B969E",lw=1)
        ax.set_xlim(0,103);ax.set_ylim(0,103);ax.set_aspect("equal",adjustable="box")
        ax.set_xlabel(f"{ref.group} document F1 (%)");ax.set_ylabel("Hybrid document F1 (%)")
        ax.set_title(f"{letter}   Hybrid vs. {ref.group}",loc="left")
        ax.text(.035,.96,f"Hybrid: {wins} better / {losses} worse / {30-wins-losses} tied",transform=ax.transAxes,va="top",fontsize=9)
        ax.legend(frameon=False,loc="lower right")
        data.append({"reference":ref.key,"hybrid":hybrid.key,"documents":rows,"wins":wins,"losses":losses})
    _foot(fig,"Above the diagonal favors Hybrid. Per-document F1 treats each plotted page separately; the headline micro F1 pools all field counts and therefore does not equal the mean of these points. Default/challenge pages share the same axes.")
    return _plot(fig,"paired_documents",representatives,data)


def _validity(runs):
    fig,axes=_new("Structural validity and exact extraction answer different questions",
                   "Best observed 9B configurations · rates from raw generation diagnostics and exact saved content",cols=3,size=(12,5.8),left=.18,wspace=.2)
    metrics=["parse_rate","schema_valid_rate","document_exact_match_rate"]
    for ax,metric,title in zip(axes[0],metrics,["A   JSON parse rate","B   Schema validity","C   Document exact match"]):
        for i,subset in enumerate(SUBSETS):
            values=[100*r.metric(metric,subset) for r in runs]
            bars=ax.barh(np.arange(3)+(i-1)*.23,values,height=.21,color=SUBCOLORS[subset],label=subset.title())
            ax.bar_label(bars,fmt="%.1f",fontsize=8,padding=2)
        ax.set_yticks(range(3),[r.group for r in runs] if metric==metrics[0] else [""]*3)
        ax.invert_yaxis();ax.set_xlim(0,113);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel("Documents (%)")
        ax.set_title(title,loc="left");ax.grid(axis="y",visible=False);ax.grid(axis="x")
    _bottom_legend(fig,axes[0,0],center=.48)
    _foot(fig,"A schema flag can reflect numeric notation such as an unquoted 5.784 package count; it need not indicate an OCR reading failure. Parse/schema validity is evaluated before template filling. Exact match is strict and unnormalized.")
    return _plot(fig,"output_validity",runs,_quality_data(runs))


def _figures(runs):
    lookup={r.key:r for r in runs}
    def get(namespace,name,label=None):
        r=lookup[f"{namespace}::{name}"]
        if label is not None:r.label=label
        return r
    prefix="qwen35-9b-"
    nf4=get("e2e",prefix+"qlora-r16-screening-untruncated","E2E · Q1 NF4")
    l1=get("e2e",prefix+"l1-lora-text-bf16","E2E · L1 BF16")
    l2=get("e2e",prefix+"l2-lora-multimodal-bf16","E2E · L2 BF16")
    q2=get("e2e",prefix+"q2-qlora-multimodal","E2E · Q2 NF4")
    phard=get("parser",prefix+"fusion-hard-l1-bf16-r16-lr5e5","Parser · hard L1 BF16")
    pnf4=get("parser",prefix+"fusion-hard-q1-nf4-r16-lr5e5","Parser · hard Q1 NF4")
    hhard=get("hybrid",prefix+"hybrid-input-hard-nf4-r16-chunked","Hybrid · hard NF4")
    representatives=[l2,phard,hhard]
    # Compact, unique labels keep dense resource figures readable.
    for r in runs:
        if r.group=="Parser + LLM":
            parts=r.key.split("fusion-")[1].split("-")
            r.label=f"Parser · {parts[0]} {parts[1].upper()}"
        elif r.group=="Hybrid":
            tail=r.key.split("hybrid-")[1]
            r.label="Hybrid · "+tail.replace("input-","").replace("dropout-","").split("-nf4")[0]
        elif r is not nf4 and r is not l1 and r is not l2 and r is not q2:
            r.label=r.key.split("::")[1].replace("qwen35-9b-", "E2E · ").replace("-qlora-r16-screening-untruncated", " QLoRA")
            if "qlora-r16-resolution-" in r.key:
                r.label="E2E · "+r.key.split("resolution-")[1]+" resolution"
    yield "01_category_overview", "All three approaches", _overview(runs,representatives)
    base=[get("e2e",f"qwen35-{size}b-base-nf4") for size in (4,9,27)]
    tuned=[get("e2e",f"qwen35-{size}b-qlora-r16-screening-untruncated") for size in (4,9,27)]
    yield "02_end_to_end_model_size", "End-to-End: model size", _scale(base,tuned)
    full=[get("e2e",prefix+name,label) for name,label in [
        ("f1-full-text","F1 · full language / frozen vision"),("f2-full-text-merger","F2 · full language + merger"),
        ("f3-full-last9","F3 · full language + last 9 vision blocks"),("f4-full-all","F4 · full model")]]
    tuning=[replace(nf4,label="Q1 · NF4 language LoRA"),replace(q2,label="Q2 · NF4 multimodal LoRA"),
            replace(l1,label="L1 · BF16 language LoRA"),replace(l2,label="L2 · BF16 multimodal LoRA"),*full]
    yield "03_end_to_end_tuning", "End-to-End: tuning scope", _tuning(tuning)
    pairs=[]
    for name in ("l1-lora-text","l2-lora-multimodal"):
        pair=[get("e2e",prefix+name+"-"+precision) for precision in ("nf4","bf16")]
        manifests=[json.loads(Path(next(p for p in r.sources if p.endswith("inference_manifest.jsonl"))).read_text().splitlines()[0]) for r in pair]
        if manifests[0]["adapter_path"]!=manifests[1]["adapter_path"]:
            raise ValueError("Precision comparison must use the same adapter")
        pairs.append(pair)
    yield "04_end_to_end_inference_precision", "End-to-End: inference precision", _precision(pairs)
    resolutions=[get("e2e",prefix+"qlora-r16-resolution-low"),nf4,
                 get("e2e",prefix+"qlora-r16-resolution-high"),get("e2e",prefix+"qlora-r16-resolution-native")]
    yield "05_end_to_end_resolution", "End-to-End: image resolution", _resolution(resolutions)
    parser=[r for r in runs if r.group=="Parser + LLM"]
    yield "06_parser_fusion_matrix", "Parser + LLM: the complete matrix", _parser_matrix(parser)
    hocr=get("hybrid",prefix+"hybrid-input-ocr-nf4-r16-chunked")
    hlayout=get("hybrid",prefix+"hybrid-input-layout-nf4-r16")
    inputs=[replace(nf4,label="Image only"),replace(hlayout,label="Image + layout"),replace(hocr,label="Image + OCR"),replace(hhard,label="Image + hard fusion")]
    yield "07_hybrid_input_evidence", "Hybrid: input evidence", _hybrid_inputs(inputs)
    dropout=[replace(hhard,label="No modality dropout"),
             replace(get("hybrid",prefix+"hybrid-dropout-image20-nf4-r16-chunked"),label="Image dropout 20%"),
             replace(get("hybrid",prefix+"hybrid-dropout-parser20-nf4-r16-chunked"),label="Parser dropout 20%"),
             replace(get("hybrid",prefix+"hybrid-dropout-balanced10-10-nf4-r16-chunked"),label="Balanced 10% / 10%")]
    yield "08_hybrid_modality_dropout", "Hybrid: modality dropout", _dropout(dropout)
    measured=[r for r in runs if "qwen35-9b-" in r.key and "-base-" not in r.key and any("process" in s.memory for s in r.measurements)]
    gpu_ids={d for r in measured for s in r.measurements for d in s.devices}
    if len(gpu_ids)!=1:raise ValueError("Resource figures assume the recorded same-GPU campaign")
    yield "09_quality_resource_tradeoffs", "Across categories: quality and resource cost", _tradeoffs(measured,{r.key for r in representatives})
    latency=[replace(nf4,label="E2E · 9B QLoRA NF4"),replace(tuned[2],label="E2E · 27B QLoRA NF4"),
             replace(l2,label="E2E · 9B L2 BF16"),replace(pnf4,label="Parser · hard Q1 NF4"),replace(phard,label="Parser · hard L1 BF16"),
             replace(hhard,label="Hybrid · hard NF4")]
    yield "10_latency_distributions", "Across categories: latency distributions", _latency(latency)
    yield "11_pipeline_stage_costs", "Across categories: pipeline stage costs", _stage_costs([latency[i] for i in (0,3,5,2,4)])
    yield "12_field_section_strengths", "Across categories: field-level strengths", _fields(representatives)
    yield "13_paired_document_comparison", "Across categories: paired documents", _paired_documents(representatives)
    yield "14_output_validity", "Across categories: structural validity", _validity(representatives)


def _gallery(out, entries):
    cards=[]
    for entry in entries:
        stem,title,caption=entry["stem"],entry["title"],entry["caption"]
        cards.append(f'<article id="{escape(stem)}"><h2>{escape(title)}</h2><a href="{stem}.png"><img loading="lazy" src="{stem}.png" alt="{escape(title)}"></a><p>{escape(caption)}</p><nav><a href="{stem}.pdf">PDF</a><a href="{stem}.svg">SVG</a><a href="{stem}.png">PNG</a><a href="{stem}.data.json">Figure data</a></nav></article>')
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>CMR experiment figures</title>
<style>body{margin:0;background:#f2f5f7;color:#203040;font:16px/1.65 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:36px 24px}h1{font-size:38px;line-height:1.2}h2{font-size:23px}header{padding:20px 0 32px}a{color:#006a9c}article{margin:30px 0;padding:24px;background:white;border:1px solid #dce4e9;border-radius:8px}img{display:block;width:100%;height:auto}nav{display:flex;gap:24px;flex-wrap:wrap}.note{max-width:1000px;color:#52606d}nav a{font-weight:600}</style>
<main><header><h1>Three routes to structured document extraction</h1><p>End-to-End, Parser + LLM, and Hybrid — the concluded CMR experiments.</p><nav><a href="experiment_figures.pdf">Complete PDF atlas</a><a href="experiment_figures.zip">Download all figures and data</a><a href="input_audit.json">Input provenance</a></nav><p class="note">Same 30 test pages: 15 default, 15 challenge. Evaluations were recomputed from the latest saved predictions. Category maxima describe these exploratory experiments; they are not an unbiased estimate of post-selection performance. The latest 27B End-to-End run and common-size 9B comparisons are shown separately. Older full-tuning outputs are labeled by inference campaign.</p></header>'''+"\n".join(cards)+"</main></html>"
    (out/"index.html").write_text(html,encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=REPO_ROOT/"output/report_figures/experiment_suite")
    args=parser.parse_args(argv)
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    with mpl.rc_context(RC_SUITE):
        runs=_load_inputs(out)
        entries=[]
        with PdfPages(out/"experiment_figures.pdf",metadata={"Title":"CMR extraction experiments: End-to-End, Parser + LLM, Hybrid"}) as atlas:
            for stem,title,plot in _figures(runs):
                plot.figure.canvas.draw()
                renderer=plot.figure.canvas.get_renderer()
                for legend in plot.figure.legends:
                    bounds=legend.get_window_extent(renderer)
                    for axis in plot.figure.axes:
                        if bounds.overlaps(axis.xaxis.label.get_window_extent(renderer)):
                            raise ValueError(f"{stem}: legend overlaps an axis label")
                    for text in plot.figure.texts:
                        if text.get_position()[1]<.1 and bounds.overlaps(text.get_window_extent(renderer)):
                            raise ValueError(f"{stem}: legend overlaps its caption")
                plot.save(out/stem,dpi=300)
                atlas.savefig(plot.figure,bbox_inches="tight")
                footer=[text.get_text() for text in plot.figure.texts if text.get_position()[1]<.1]
                entries.append({"stem":stem,"title":title,"caption":" ".join(footer)})
                plot.figure.clear()
                print(f"Saved {stem}",flush=True)
        _gallery(out,entries)
        (out/"figure_catalog.json").write_text(json.dumps(entries,indent=2)+"\n")
        (out/"reproduce.txt").write_text("Run from the repository root:\npython -m src.report_plots.experiment_suite --output-dir "+str(out)+"\n")
        with zipfile.ZipFile(out/"experiment_figures.zip","w",compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(out.rglob("*")):
                if path.is_file() and path.name!="experiment_figures.zip":bundle.write(path,path.relative_to(out))
    print(f"Generated {len(entries)} figures, PDF atlas, and gallery at {out}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
