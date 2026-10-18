"""Chapter 6 plotting contracts against the canonical consolidated snapshot."""

import unittest
import json
from unittest.mock import patch

from matplotlib.collections import PathCollection
import numpy as np

import plot_cmi_ch6 as plot


class Chapter6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs, cls.selected, _, _, cls.latency, cls.memory, cls.devices = plot.collect(
            plot.ROOT / "results/cmi_bench_consolidated")

    def tearDown(self):
        plot.plt.close("all")

    def test_timing_statistics(self):
        self.assertEqual(len(self.runs), 288)
        self.assertEqual(len(self.selected), 72)
        self.assertEqual(len(self.latency), 24)
        for row in self.latency:
            runs = [r for r in self.selected if all(r[k] == row[k] for k in
                    ("model", "placement", "provider"))]
            means = np.array([np.mean([float(v) for v in r["measured_ms"].split(";")])
                              / 1000 for r in runs])
            self.assertEqual(row["n_reps"], 3)
            self.assertAlmostEqual(row["mean_s"], means.mean())
            self.assertAlmostEqual(row["sd_s"], means.std(ddof=1))
            half = 4.302652729911275 * means.std(ddof=1) / np.sqrt(3)
            self.assertAlmostEqual(row["ci_low_s"], means.mean() - half)
            self.assertAlmostEqual(row["ci_high_s"], means.mean() + half)

    def test_no_dots_and_timing_errorbars(self):
        with patch.object(plot.plt.Axes, "scatter", side_effect=AssertionError("No dots")):
            for model in plot.MODELS:
                fig, axes = plot.figure_panels()
                ax = axes[0, 0]
                plot.latency_panel(ax, self.latency, model)
                self.assertEqual(ax.get_title(loc="left"), "")
                self.assertEqual(len(ax.patches), 12)
                self.assertEqual(sum(type(c).__name__ == "ErrorbarContainer"
                                     for c in ax.containers), 12)
                self.assertFalse(fig.findobj(PathCollection))
                for panel in range(3):
                    fig, axes = plot.figure_panels()
                    ax = axes[0, 0]
                    plot.memory_panel(ax, self.memory, model, panel)
                    self.assertEqual(ax.get_title(loc="left"), "")
                    self.assertTrue(all(t.get_text() in ("N/A", "missing", "*") for t in ax.texts))
                    self.assertFalse(fig.findobj(PathCollection))
                    self.assertFalse(ax.collections)  # Maxima are not mean/CI estimates.

    def test_memory_maxima_and_missing(self):
        for row in self.memory:
            present = [row[f"rep{i}_gib"] for i in (1, 2, 3) if row[f"rep{i}_gib"] != ""]
            effective = [row[f"effective_rep{i}_gib"] for i in (1, 2, 3) if row[f"effective_rep{i}_gib"] != ""]
            self.assertEqual(row["max_gib"], max(effective) if effective else "")
        missing = [r for r in self.devices if not r["peak_gib"]]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["cell"], "aix-24c1g-default-coll-transformer-r2")
        row = next(r for r in self.memory if (r["model"], r["placement"], r["provider"], r["metric"])
                   == ("transformer", "24c1g", "aix", "gpu_vram"))
        self.assertEqual(row["n_reps_observed"], 2)
        self.assertEqual(row["rep2_gib"], "")
        self.assertEqual(row["effective_rep2_gib"], 16969 / 1024)
        self.assertEqual(row["supplemental_observations"], 1)
        self.assertEqual(row["supplemental_jobid"], "5106286")
        self.assertEqual(row["original_missing_jobid"], "5009522")
        repeat = next(r for r in self.devices if r["observation_kind"] == "supplemental_repeat")
        self.assertEqual(repeat["measurement_jobid"], "5106286")
        self.assertEqual(repeat["original_jobid"], "5009522")
        self.assertEqual(repeat["scope"], "device_wide")
        self.assertEqual(repeat["device"], "0")
        fig, axes = plot.figure_panels()
        plot.memory_panel(axes[0, 0], self.memory, "transformer", 2)
        self.assertEqual([t.get_text() for t in axes[0, 0].texts], ["*"])
        fig, axes = plot.figure_panels()
        original = plot.collect(plot.ROOT / "results/cmi_bench_consolidated", supplement=None)
        plot.memory_panel(axes[0, 0], original[5], "transformer", 2)
        self.assertEqual([t.get_text() for t in axes[0, 0].texts], ["*"])

    def test_supplement_only_changes_designated_capacity(self):
        original = plot.collect(plot.ROOT / "results/cmi_bench_consolidated", supplement=None)
        current = plot.collect(plot.ROOT / "results/cmi_bench_consolidated")
        for index in (0, 1, 2, 3, 4):
            self.assertEqual(original[index], current[index])
        self.assertEqual(sum(r["status"] == "missing" and r["expected"] == "True"
                             and r["metric"] == "gpu_vram" for r in current[2]), 1)
        for old, new in zip(original[5], current[5]):
            if new["supplemental_observations"]:
                self.assertEqual((new["model"], new["placement"], new["provider"], new["metric"]),
                                 ("transformer", "24c1g", "aix", "gpu_vram"))
                self.assertEqual(new["max_gib"], max(old["max_gib"], 16969 / 1024))
            else:
                self.assertEqual(old, new)
        self.assertEqual(original[6], current[6][:-1])

    def test_reject_supplement_mismatches(self):
        _, selected, resources, report, *_ = plot.collect(
            plot.ROOT / "results/cmi_bench_consolidated", supplement=None)
        manifest = json.loads(plot.SUPPLEMENT.read_text())
        for field, value in (("measurement_jobid", "999"), ("original_jobid", "999"),
                             ("hashes", {**manifest["hashes"], next(iter(manifest["hashes"])): "0" * 64})):
            with self.subTest(field=field), patch.object(plot.Path, "read_text", return_value=json.dumps({**manifest, field: value})):
                with self.assertRaises(ValueError):
                    plot.supplemental_peak(selected, resources, report)
        original_parse = plot.consolidator.parse_selected_run
        with patch.object(plot.consolidator, "parse_selected_run", wraps=original_parse) as parse:
            def mismatch(*args, **kwargs):
                row, calls = original_parse(*args, **kwargs)
                return {**row, "samples_aggregate": 1}, calls
            parse.side_effect = mismatch
            with self.assertRaisesRegex(ValueError, "workload mismatch"):
                plot.supplemental_peak(selected, resources, report)
        extract = plot.consolidator.extract_resources
        for field, value in (("scope", "process"), ("device", 1), ("baseline_gib", "")):
            def mismatch_resource(*args, **kwargs):
                rows, warnings = extract(*args, **kwargs)
                return [{**r, field: value} if r["metric"] == "gpu_vram" else r for r in rows], warnings
            with self.subTest(field=field), patch.object(plot.consolidator, "extract_resources", side_effect=mismatch_resource):
                with self.assertRaisesRegex(ValueError, "device-wide"):
                    plot.supplemental_peak(selected, resources, report)

    def test_export_dimensions(self):
        for count, width, height in ((1, 150, 54), (2, 150, 55), (3, 150, 149), (1, 73.5, 64)):
            fig, axes = plot.figure_panels(count, width=width)
            with patch.object(plot.ts, "save") as save:
                plot.finish(fig, plot.ROOT, "unused")
            self.assertFalse(fig.texts)
            save.assert_called_once_with(fig, plot.ROOT / "unused")
            np.testing.assert_allclose(fig.get_size_inches() * 25.4, (width, height))
            for ax in axes.flat:
                self.assertAlmostEqual(ax.get_position().height * height, 32)
                self.assertEqual(bool(ax.get_xticklabels()), ax.get_subplotspec().is_last_row())
            if count == 2:
                self.assertEqual(axes.shape, (1, 2))
                left, right = [ax.get_position() for ax in axes.flat]
                self.assertAlmostEqual(left.y0, right.y0)
                self.assertLess(left.x1, right.x0)
                self.assertAlmostEqual((right.x0 - left.x0) * width, 76.5)
                self.assertAlmostEqual(left.width * width, 58.5)

    def test_paired_rendered_text_bounds(self):
        for kind in ("latency", "vram"):
            fig, axes = plot.figure_panels(2)
            for ax, model in zip(axes.flat, plot.MODELS):
                if kind == "latency":
                    plot.latency_panel(ax, self.latency, model)
                else:
                    plot.memory_panel(ax, self.memory, model, 2)
            with patch.object(plot.ts, "save"):
                plot.finish(fig, plot.ROOT, "unused")
            renderer = fig.canvas.get_renderer()
            legend = fig.legends[0].get_window_extent(renderer)
            self.assertGreaterEqual(legend.x0, 0)
            self.assertLessEqual(legend.x1, fig.bbox.x1)
            self.assertLessEqual(legend.y1, fig.bbox.y1)
            for ax in axes.flat:
                title = ax._left_title.get_window_extent(renderer)
                gap_mm = (legend.y0 - title.y1) / fig.dpi * 25.4
                self.assertGreater(gap_mm, 0)
                self.assertLessEqual(gap_mm, 3)
                if kind == "vram":
                    self.assertEqual(ax.get_ylabel(), "Max. per-GPU VRAM (GiB)")
                bounds = ax.get_tightbbox(renderer)
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, fig.bbox.x1)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLess(bounds.y1, legend.y0)
                for text in ax.texts:
                    self.assertLess(text.get_window_extent(renderer).y1, title.y0)
                    self.assertGreaterEqual(text.get_window_extent(renderer).x0, ax.bbox.x0)
                    self.assertLessEqual(text.get_window_extent(renderer).x1, ax.bbox.x1)

    def test_rendered_exports(self):
        from PIL import Image

        out = plot.ROOT / "figures/ch6_gpu_draft"
        names = {"latency_baselines": (150, 55), "gpu_device_peaks": (150, 55)}
        for model in plot.MODELS:
            names[f"{model}_memory"] = (150, 149)
            for kind in ("latency", "cpu_ram", "ml_host_ram", "vram"):
                names[f"{model}_{kind}"] = (150, 54)
                names[f"{model}_{kind}_halfwidth"] = (73.5, 64)
        for name, (width, height) in names.items():
            with self.subTest(figure=name):
                for suffix in ("png", "pdf", "pgf"):
                    self.assertGreater((out / f"{name}.{suffix}").stat().st_size, 0)
                with Image.open(out / f"{name}.png") as image:
                    self.assertLessEqual(abs(image.width - width / 25.4 * 300), 1)
                    self.assertLessEqual(abs(image.height - height / 25.4 * 300), 1)
                pgf = (out / f"{name}.pgf").read_text()
                models = plot.MODELS if name in ("latency_baselines", "gpu_device_peaks") else (name.split("_")[0],)
                for model in models:
                    if name in ("latency_baselines", "gpu_device_peaks") or name.endswith("_memory"):
                        self.assertIn(plot.MODEL_LABELS[model], pgf)
                    else:
                        self.assertNotIn(plot.MODEL_LABELS[model], pgf)
                for note in ("repetition means", "Student-t", "No CI", "no imputation",
                             "2 original", "r2 missing", "(a)", "(b)"):
                    self.assertNotIn(note, pgf)


if __name__ == "__main__":
    unittest.main()
