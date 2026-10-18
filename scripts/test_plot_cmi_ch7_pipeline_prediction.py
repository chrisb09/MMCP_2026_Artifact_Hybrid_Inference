"""Run: python -m unittest discover -s scripts -p test_plot_cmi_ch7_pipeline_prediction.py."""

import math
import statistics
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import plot_cmi_ch7_pipeline_prediction as plot


class PipelineTests(unittest.TestCase):
    def test_thesis_synthetic(self):
        # Per-rank times sum to 38, max 20; collective A is N*tau.
        result = plot.pipeline([50, 30, 200, 20, 80], 10)
        self.assertEqual(result["stage_sum_ms"], 380)
        self.assertEqual(result["pipeline_ms"], 218)
        self.assertEqual(result["savings_ms"], 162)

    def test_zero_single_rank_and_dominant_stage(self):
        for stages, n in (([0]*5, 10), ([1, 2, 3, 4, 5], 1), ([0, 0, 100, 0, 0], 48)):
            p = plot.pipeline(stages, n)
            self.assertEqual(p["savings_ms"], 0)
            self.assertAlmostEqual(p["pipeline_ms"], sum(stages))
        for stages, n in (([1]*4, 10), ([1]*5, 0), ([1]*5, 1.5), ([-1]*5, 10), ([math.nan]*5, 10)):
            with self.assertRaises(ValueError):
                plot.pipeline(stages, n)

    def test_device_cpu_not_double_counted(self):
        row = dict(is_controller="1", workgroup_rank="0", diagnostic_barriers="0",
                   gather_arrival_wait_s="0", scatter_arrival_wait_s="0", gather_collective_s=".05",
                   scatter_collective_s=".08", h2d_gpu_ms="30", forward_gpu_ms="200", d2h_gpu_ms="20",
                   device_cpu_s="999")
        self.assertEqual(plot.collective_stages(row), [50, 30, 200, 20, 80])
        row["device_cpu_s"] = ".25"
        self.assertEqual(sum(plot.collective_stages(row)), 380)
        for key, value in (("is_controller", "0"), ("workgroup_rank", "1"),
                           ("diagnostic_barriers", "1"), ("gather_arrival_wait_s", ".1")):
            with self.assertRaises(ValueError):
                plot.collective_stages(dict(row, **{key: value}))

    def test_no_gpu_scaling_residual_and_critical_switch(self):
        g = dict(workgroup=0, **plot.pipeline([50, 30, 200, 20, 80], 10))
        for count in (1, 4):
            p = plot.job_prediction([dict(g, workgroup=i) for i in range(count)], 500)
            self.assertEqual(p["savings_ms"], 162)
            self.assertEqual(p["residual_ms"], 120)
            self.assertEqual(p["predicted_ms"], 338)
        other = dict(workgroup=1, **plot.pipeline([0, 0, 300, 0, 0], 10))
        p = plot.job_prediction([g, other], 500)
        self.assertEqual((p["before_critical_wg"], p["after_critical_wg"]), (0, 1))
        self.assertEqual(p["savings_ms"], 80)
        self.assertEqual(plot.job_prediction([g], 300)["residual_ms"], -80)
        with self.assertRaisesRegex(ValueError, "Negative predicted"):
            plot.job_prediction([g], 100)

    def test_three_rep_ci(self):
        s = plot.summarize([100, 200, 300])
        self.assertEqual(s["mean_ms"], 200)
        self.assertAlmostEqual(s["ci_high_ms"]-200, plot.T95*100/math.sqrt(3))
        self.assertLess(s["ci_low_ms"], 0)
        with self.assertRaises(ValueError):
            plot.summarize([100]*15)

    def test_canonical_all_calls(self):
        observations, predictions, calls, values, metadata = plot.collect()
        self.assertEqual(len(metadata["selected_runs"]), 144)
        self.assertEqual((len(observations), len(predictions), len(calls), len(values)), (1800, 240, 960, 64))
        self.assertEqual({r["rep"] for r in observations}, {1, 2, 3})
        self.assertEqual({r["call"] for r in observations}, set(range(2, 7)))
        for p in predictions:
            groups = [g for g in observations if g["cell"] == p["cell"] and g["call"] == p["call"]]
            self.assertAlmostEqual(p["predicted_ms"], p["observed_mlstep_ms"]-p["savings_ms"])
            self.assertAlmostEqual(p["predicted_ms"], p["residual_ms"]+p["after_envelope_ms"])
            self.assertGreaterEqual(p["predicted_ms"], 0)
            self.assertLessEqual(p["savings_ms"], max(g["savings_ms"] for g in groups)+1e-9)
            for g in groups:
                self.assertEqual(g["n_members"], len(g["members"].split(";")))
                self.assertAlmostEqual(g["stage_sum_ms"], sum(g[s+"_ms"] for s in plot.STAGES))
                self.assertAlmostEqual(g["gpu_work_sum_ms"], sum(g[s+"_ms"] for s in plot.base.PHASES))
        for v in values:
            for rep in range(1, 4):
                selected = [r["value_ms"] for r in calls if all(r[k] == v[k] for k in
                            ("model", "placement", "layout", "mode")) and r["rep"] == rep]
                self.assertEqual(len(selected), 5)
                self.assertEqual(v[f"rep{rep}_ms"], statistics.mean(selected))
        for g in observations:
            self.assertIn(str(plot.base.ROOT / g["source_path"]), metadata["sources"])

    def test_caption_free_layout_preserves_axis_height(self):
        sys.path.insert(0, str(Path.home() / "Master-Thesis/gitlab/sourcecode"))
        import thesis_style as ts
        import matplotlib.pyplot as plt

        values = plot.base.read_csv(plot.base.ROOT / "figures/ch7_pipeline_prediction_draft/plot_values.csv")
        values = [{k: float(v) if k in ("mean_ms", "ci_low_ms", "ci_high_ms") else v
                   for k, v in row.items()} for row in values]

        def check_figure(fig, path):
            preview = path.name.endswith("combined_preview")
            height = 72.8052 if preview else 68.6052
            self.assertAlmostEqual(fig.get_figwidth() * 25.4, 150)
            self.assertAlmostEqual(fig.get_figheight() * 25.4, height)
            self.assertFalse(fig.texts)
            renderer = fig.canvas.get_renderer()
            legend = fig.legends[0].get_window_extent(renderer)
            for ax in fig.axes:
                self.assertAlmostEqual(ax.get_position().height * height, 41.6052)
                self.assertFalse(ax.texts)
                self.assertEqual(len(ax.patches), 16)
                self.assertEqual(sum(type(c).__name__ == "ErrorbarContainer" for c in ax.containers), 16)
                if preview:
                    self.assertIn("overlap", ax.get_title(loc="left"))
                    self.assertLess(ax._left_title.get_window_extent(renderer).y1, legend.y0)
                else:
                    self.assertEqual(ax.get_title(loc="left"), "")
                bounds = ax.get_tightbbox(renderer)
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLessEqual(bounds.x1, fig.bbox.x1)
                self.assertLessEqual(bounds.y1, fig.bbox.y1)
            plt.close(fig)

        with patch.object(ts, "save", side_effect=check_figure) as save, patch.object(Path, "is_file", return_value=True):
            names = plot.render(values, ("default", "numa"), plot.base.ROOT)
        self.assertEqual(len(names), 6)
        self.assertEqual(save.call_count, 6)

    def test_rendered_exports(self):
        from PIL import Image

        out = plot.base.ROOT / "figures/ch7_pipeline_prediction_draft"
        for layout in ("default", "numa"):
            for panel in ("combined_preview", "mlp", "transformer"):
                name = f"{layout}_{panel}"
                with self.subTest(figure=name):
                    for suffix in ("png", "pdf", "pgf"):
                        self.assertGreater((out / f"{name}.{suffix}").stat().st_size, 0)
                    height = 72.8052 if panel == "combined_preview" else 68.6052
                    with Image.open(out / f"{name}.png") as image:
                        self.assertLessEqual(abs(image.width - 150 / 25.4 * 300), 1)
                        self.assertLessEqual(abs(image.height - height / 25.4 * 300), 1)
                    pgf = (out / f"{name}.pgf").read_text()
                    for note in ("repetition means", "95%", "Prediction:", "proxy", "makespan", "(a)", "(b)"):
                        self.assertNotIn(note, pgf)
                    for model in ("MLP (overlap 4)", "Transformer (overlap 0)"):
                        if panel == "combined_preview":
                            self.assertIn(f"{model} / {layout}", pgf)
                        else:
                            self.assertNotIn(model, pgf)


if __name__ == "__main__":
    unittest.main()
