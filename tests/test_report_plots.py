"""Scientific aggregation, artifact loading, and synthetic rendering checks."""
from __future__ import annotations

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.report_plots import Measurement, RunResult, load_config, load_results
from src.report_plots.data import read_manifest
from src.report_plots.statistics import (mean_stages, paired_bootstrap, pareto_indices,
                                         runtime_samples, summarize, timing_values)


def synthetic_run(key="baseline", second=False):
    counts = [(9, 1, 1), (1, 1, 9)] if not second else [(8, 2, 2), (4, 1, 6)]
    samples = {}
    for name, subset, (tp, fp, fn) in zip(("a", "b"), ("default", "challenge"), counts):
        samples[name] = {"sample_id": name, "subset": subset, "metrics": {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "f1": 2*tp/(2*tp+fp+fn)}}
    tp, fp, fn = np.asarray(counts).sum(axis=0)
    overall = 2*tp/(2*tp+fp+fn)
    metrics = {"overall": {"field_f1": overall, "field_precision": tp/(tp+fp), "field_recall": tp/(tp+fn),
                           "parse_rate": 1., "schema_valid_rate": 1.},
               "default": {"field_f1": samples["a"]["metrics"]["f1"]},
               "challenge": {"field_f1": samples["b"]["metrics"]["f1"]}}
    measurements = [Measurement("a", "ok", "default", 10., 8., {"ocr": 1., "generation": 8.},
                                {"process": 4., "torch_allocated": 2.}, ("gpu-A",)),
                    Measurement("b", "schema_error", "challenge", 20., 15., {"ocr": 2., "generation": 15.},
                                {"process": 6., "torch_allocated": 3.}, ("gpu-A",))]
    fields = {subset: {"sender.name": {"f1": .8, "support": 2},
                       "itemList[].weight": {"f1": .6 if second else .7, "support": 1}}
              for subset in ("overall", "default", "challenge")}
    return RunResult(key, "Candidate" if second else "Baseline", "Hybrid", metrics,
                     {"overall": 2, "default": 1, "challenge": 1}, measurements, fields, samples,
                     sources=["synthetic-fixture"])


class PlotStatisticsTests(unittest.TestCase):
    def test_completed_schema_and_parse_failures_count_but_oom_does_not(self):
        run = synthetic_run()
        run.measurements.extend([
            Measurement("c", "parse_error", "challenge", 30., 28., {}, {}),
            Measurement("d", "inference_error", "challenge", 1., None, {}, {}),
            Measurement("e", "template_error", "challenge", 40., 38., {}, {}),
            Measurement("f", "ok", "challenge", None, None, {}, {})])
        selected = runtime_samples(run)
        self.assertEqual(timing_values(selected, "total"), [10., 20., 30., 40.])
        self.assertEqual((selected.failures, selected.missing), (1, 1))
        summary = summarize(timing_values(selected, "total"))
        self.assertEqual(summary["median"], 25.)
        self.assertAlmostEqual(summary["p95"], 38.5)
        challenge = runtime_samples(run, subset="challenge", drop_first=1)
        self.assertEqual(challenge.warmup_removed, 1)
        self.assertEqual(len(challenge.samples), 3)

    def test_missing_subset_or_memory_is_not_treated_as_zero(self):
        run = synthetic_run()
        run.measurements[0].subset = None
        with self.assertRaisesRegex(ValueError, "subset timing"):
            runtime_samples(run, subset="default")
        with self.assertRaisesRegex(ValueError, "no completed samples"):
            runtime_samples(run, memory_kind="paddle_reserved")
        with self.assertRaises(ValueError):
            runtime_samples(run, drop_first=-1)

    def test_stage_means_use_same_documents_and_reject_overlaps(self):
        selected = runtime_samples(synthetic_run())
        means = mean_stages(selected)
        self.assertEqual(means, {"generation": 11.5, "ocr": 1.5, "other": 2.})
        self.assertEqual(sum(means.values()), 15.)
        selected.samples[0].stages["generation"] = 20.
        with self.assertRaisesRegex(ValueError, "exceed pipeline"):
            mean_stages(selected)

    def test_pareto_minimizes_cost_and_maximizes_quality(self):
        self.assertEqual(pareto_indices([1, 2, 3, 1, 4], [.5, .7, .6, .5, .9]), [0, 1, 3, 4])

    def test_bootstrap_pairs_ids_and_aggregates_micro_counts(self):
        base, candidate = synthetic_run(), synthetic_run("candidate", True)
        first = paired_bootstrap(base, candidate, resamples=200, seed=7)
        second = paired_bootstrap(base, candidate, resamples=200, seed=7)
        self.assertEqual(first, second)
        expected = 24/35 - 20/32
        self.assertAlmostEqual(first["difference"], expected)
        # Macro-F1 averaging would give a different answer on this unequal-support fixture.
        macro = np.mean([s["metrics"]["f1"] for s in candidate.samples.values()]) - np.mean([s["metrics"]["f1"] for s in base.samples.values()])
        self.assertNotAlmostEqual(first["difference"], macro)
        equal = paired_bootstrap(base, base, resamples=100)
        self.assertEqual((equal["difference"], equal["low"], equal["high"]), (0., 0., 0.))
        candidate.samples.pop("b")
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_bootstrap(base, candidate, resamples=100)


class PlotDataTests(unittest.TestCase):
    def test_manifest_memory_provenance_duplicates_and_foreign_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"manifest.jsonl"
            record = {"image_path": r"C:\remote\a_240dpi.png", "status": "ok", "inference_seconds": 2.,
                      "vram": {"available": True, "devices": [
                          {"device": "GPU0", "peak_used_bytes": 4*1024**3},
                          {"device": "GPU1", "peak_used_bytes": 7*1024**3}],
                          "framework_peaks": {"torch": {"available": True, "devices": [
                              {"peak_allocated_gib": 2., "peak_reserved_gib": 3.}]}}}}
            path.write_text(json.dumps(record)+'\n')
            item = read_manifest(path, {"a": "default"})[0]
            self.assertEqual(item.sample_id, "a")
            self.assertEqual(item.memory, {"process": 7., "torch_allocated": 2., "torch_reserved": 3.})
            record["vram"] = {"available": True, "source": "torch CUDA allocator", "devices": [{"peak_allocated_gib": 5.}]}
            path.write_text(json.dumps(record)+'\n')
            self.assertEqual(read_manifest(path, {})[0].memory, {"torch_allocated": 5.})
            path.write_text((json.dumps(record)+'\n')*2)
            with self.assertRaisesRegex(ValueError, "duplicate sample"):
                read_manifest(path, {})

    def _table(self, root, value=.8):
        run = root/"model"
        run.mkdir()
        record = {"image_path": "/remote/a_240dpi.png", "status": "ok", "inference_seconds": 5.}
        (run/"inference_manifest.jsonl").write_text(json.dumps(record)+'\n')
        path = root/"evaluation_table.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run", "run_directory", "overall_samples", "overall_field_f1"])
            writer.writeheader()
            writer.writerow({"run": "model", "run_directory": "/missing/remote/model", "overall_samples": 1, "overall_field_f1": value})
        return path

    def test_relocated_csv_uses_local_sibling_and_rejects_invalid_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._table(root)
            original = path.read_bytes()
            run = load_results(path, labels={"model": "Readable label"})[0]
            self.assertEqual(run.metric("field_f1"), .8)
            self.assertEqual(run.label, "Readable label")
            self.assertEqual(len(run.measurements), 1)
            self.assertEqual(path.read_bytes(), original)
            path.write_text(path.read_text().replace("0.8", "80"))
            with self.assertRaisesRegex(ValueError, "fraction"):
                load_results(path)

    def test_config_paths_namespaces_selection_and_stale_report_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._table(root)
            report = {"overall": {"summary": {"samples": 1, "field_f1": .8}, "field_breakdown": {}}, "samples": []}
            (root/"report.json").write_text(json.dumps(report))
            config = {"sources": [{"path": "evaluation_table.csv", "namespace": "experiment", "reports": {"model": "report.json"}}],
                      "runs": ["experiment::model"], "labels": {"experiment::model": "Model A"}}
            path = root/"figures.json"
            path.write_text(json.dumps(config))
            runs, _ = load_config(path)
            self.assertEqual(runs[0].key, "experiment::model")
            self.assertEqual(runs[0].label, "Model A")
            report["overall"]["summary"]["field_f1"] = .7
            (root/"report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_config(path)

    def test_manifest_count_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._table(root)
            (root/"model/inference_manifest.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "manifest count"):
                load_results(root)

    def test_prediction_folder_without_table_evaluates_wrapped_outputs_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root/"predictions"
            predictions.mkdir()
            for subset, name, value in (("default", "a", "A"), ("challenge", "b", "B")):
                folder = root/"test/annotations/ground_truths"/subset
                folder.mkdir(parents=True)
                (folder/f'gt_{name}.json').write_text(json.dumps({"content": {"value": value}}))
                (predictions/f'{name}_240dpi.json').write_text(json.dumps({"generated": {"value": value}}))
            schema = root/"schema.json"
            schema.write_text(json.dumps({"type": "object", "properties": {"value": {"type": "string"}}}))
            runs = load_results(predictions, testset_path=root/"test", schema_path=schema,
                                prediction_key="generated", details=True)
            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].metric("field_f1"), 1.)
            self.assertEqual(len(runs[0].samples), 2)
            self.assertTrue(runs[0].fields["overall"])
            self.assertFalse((predictions/"evaluation_table.csv").exists())


class SyntheticRenderingTests(unittest.TestCase):
    def test_all_recipes_render_without_interactive_backend_or_global_style_changes(self):
        import matplotlib as mpl
        from src.report_plots import plots
        original_font = copy.deepcopy(mpl.rcParams["font.family"])
        runs = [synthetic_run(), synthetic_run("candidate", True)]
        recipes = [plots.plot_quality(runs, metrics=("field_f1", "field_precision", "field_recall")),
                   plots.plot_subsets(runs), plots.plot_deltas(runs, baseline="baseline"),
                   plots.plot_latency(runs), plots.plot_latency(runs, measure="generation"),
                   plots.plot_ecdf(runs), plots.plot_stages(runs), plots.plot_memory(runs),
                   plots.plot_tradeoff(runs), plots.plot_tradeoff(runs, resource="memory"),
                   plots.plot_fields(runs), plots.plot_paired(runs, baseline="baseline", resamples=100)]
        self.assertEqual(mpl.rcParams["font.family"], original_font)
        for plot in recipes:
            with self.subTest(plot=plot.recipe):
                plot.figure.canvas.draw()
                buffer = io.BytesIO()
                plot.figure.savefig(buffer, format="pdf")
                self.assertTrue(buffer.getvalue().startswith(b"%PDF"))
        with tempfile.TemporaryDirectory() as directory:
            paths = recipes[0].save(Path(directory)/"quality", dpi=72)
            self.assertEqual(len(paths), 4)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))
            payload = json.loads(paths[-1].read_text())
            self.assertEqual(payload["recipe"], "quality")
            self.assertEqual(payload["runs"][0]["sources"], ["synthetic-fixture"])
        self.assertEqual(mpl.rcParams["font.family"], original_font)

    def test_cli_generates_only_requested_synthetic_artifacts_and_listing_writes_nothing(self):
        from src.report_plots.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PlotDataTests()._table(root)
            output = root/"figures"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(root), "--list-runs", "--output-dir", str(output)]), 0)
                self.assertFalse(output.exists())
                self.assertEqual(main([str(root), "--plots", "quality", "latency", "--formats", "svg", "--output-dir", str(output)]), 0)
            self.assertEqual(len(list(output.glob("*.svg"))), 2)
            self.assertEqual(len(list(output.glob("*.data.json"))), 2)

    def test_cli_validates_all_requested_data_before_writing_figures(self):
        from src.report_plots.__main__ import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            PlotDataTests()._table(root)
            output = root/"figures"
            with contextlib.redirect_stderr(io.StringIO()):
                result = main([str(root), "--plots", "quality", "stages", "--output-dir", str(output)])
            self.assertEqual(result, 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
