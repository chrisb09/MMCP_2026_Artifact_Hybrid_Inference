"""Run: python -m unittest discover -s scripts -p test_export_cmi_aix_stage_tables.py."""

import itertools
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

import export_cmi_aix_stage_tables as export


class WindowTests(unittest.TestCase):
    def test_window_includes_gaps_and_is_origin_independent(self):
        for origin in (0, 10000):
            intervals = [dict(event="range_inference", start_s=origin + 1, end_s=origin + 2),
                         dict(event="range_inference", start_s=origin + 4, end_s=origin + 7),
                         dict(event="ml_step", start_s=origin, end_s=origin + 8)]
            self.assertEqual(export.ready_range_window(intervals), 6000)
        with self.assertRaisesRegex(ValueError, "Missing ready-range"):
            export.ready_range_window([])

    def test_formula_example_and_critical_switch(self):
        p = export.prediction
        first = dict(workgroup=0, **p.pipeline([50, 30, 200, 20, 80], 10))
        self.assertEqual(first["pipeline_ms"], 218)
        other = dict(workgroup=1, **p.pipeline([0, 0, 300, 0, 0], 10))
        result = p.job_prediction([first, other], 500)
        self.assertEqual(result["before_envelope_ms"], 380)
        self.assertEqual(result["after_envelope_ms"], 300)
        self.assertEqual(result["predicted_ms"], 420)


class CanonicalTablesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = export.collect()

    def test_complete_all_layouts_repetitions_and_calls(self):
        tables, summaries, calls, observations, predictions, metadata = self.data
        self.assertEqual(len(tables), 24)
        self.assertEqual(len(summaries), 24 * len(export.METRICS))
        self.assertEqual((len(calls), len(predictions), len(observations)), (360, 360, 2700))
        self.assertEqual(len(metadata["selected_runs"]), 216)
        for layout in export.LAYOUTS:
            self.assertEqual({(r["model"], r["placement"]) for r in tables if r["layout"] == layout},
                             set(itertools.product(export.prediction.MODELS, export.prediction.HARDWARE)))
        for r in calls:
            self.assertIn(r["rep"], (1, 2, 3))
            self.assertIn(r["call"], range(2, 7))
            self.assertEqual(r["inference_call"], r["call"] + 1)
            self.assertAlmostEqual(r["stage_sum_ms"], sum(r[k] for k in export.STAGE_FIELDS))
            self.assertEqual(r["c1_ideal_pipeline_ms"], r["full_ideal_pipeline_ms"])
            self.assertEqual(r["c1_ideal_mlstep_ms"], r["full_ideal_mlstep_ms"])
            self.assertAlmostEqual(r["c1_ideal_mlstep_ms"], r["residual_ms"] + r["c1_ideal_pipeline_ms"])
            self.assertAlmostEqual(r["c1_ideal_mlstep_ms"], r["collective_mlstep_ms"] - r["savings_ms"])

    def test_stages_same_critical_group_and_windows_local_max(self):
        _, _, calls, observations, predictions, _ = self.data
        groups = {}
        for g in observations:
            groups.setdefault((g["cell"], g["call"]), []).append(g)
        for r, p in zip(calls, predictions):
            coll = groups[r["collective_cell"], r["call"]]
            critical = max(coll, key=lambda g: g["stage_sum_ms"])
            self.assertEqual(r["before_critical_wg"], critical["workgroup"])
            for metric in export.STAGE_FIELDS:
                self.assertEqual(r[metric], critical[metric])
            self.assertEqual(r["c1_ideal_pipeline_ms"], max(g["pipeline_ms"] for g in coll))
            self.assertEqual(r["stage_sum_ms"], p["before_envelope_ms"])
            for prefix in ("c1", "full"):
                selected = groups[r[prefix + "_cell"], r["call"]]
                self.assertEqual(r[prefix + "_ready_range_window_ms"],
                                 max(g["ready_range_window_ms"] for g in selected))
                self.assertEqual(r[prefix + "_measured_mlstep_ms"], selected[0]["observed_mlstep_ms"])

    def test_run_means_ci_and_stage_sum(self):
        tables, summaries, calls, *_ = self.data
        for table in tables:
            self.assertAlmostEqual(table["stage_sum_ms"], sum(table[k] for k in export.STAGE_FIELDS))
        for s in summaries:
            selected = [r for r in calls if all(r[k] == s[k] for k in ("layout", "model", "placement"))]
            reps = [statistics.mean(r[s["metric"]] for r in selected if r["rep"] == rep)
                    for rep in range(1, 4)]
            self.assertEqual({k: s[k] for k in export.prediction.summarize(reps)},
                             export.prediction.summarize(reps))

    def test_exports_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            with patch.object(export, "collect", return_value=self.data):
                export.main(["--out-dir", str(out)])
            for layout in export.LAYOUTS:
                rows = export.base.read_csv(out / f"{layout}.csv")
                self.assertEqual(len(rows), 8)
                markdown = (out / f"{layout}.md").read_text()
                latex = (out / f"{layout}.tex").read_text()
                self.assertIn("ONE COMMON credit-independent", markdown)
                self.assertIn("\\begin{landscape}", latex)
                self.assertEqual(sum(line.startswith("| mlp |") or line.startswith("| transformer |")
                                     for line in markdown.splitlines()), 8)
                self.assertEqual(sum(line.startswith("mlp &") or line.startswith("transformer &")
                                     for line in latex.splitlines()), 8)
            self.assertEqual(len(list(out.iterdir())), 15)
            self.assertIn("NO credit term", (out / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
