"""Canonical selection, statistics, rendering and chaining-coverage contracts."""
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import plot_cmi_ch7_readiness as ready


class ReadinessTests(unittest.TestCase):
    def test_student_t_uses_three_run_means(self):
        mean, sd, half = ready.mean_ci([1, 2, 3])
        self.assertEqual((mean, sd), (2, 1))
        self.assertAlmostEqual(half, 4.30265/math.sqrt(3))
        for bad in ([1]*15, [1, 2], [1, 2, float("nan")], [-1, 2, 3]):
            with self.assertRaises(ValueError):
                ready.mean_ci(bad)

    def test_canonical_selection(self):
        values, runs = ready.collect()
        self.assertEqual(len(values), 24)
        self.assertEqual(len(runs), 72)
        self.assertEqual({r["provider"] for r in runs}, {"aix", "phydll"})
        self.assertEqual({r["placement"] for r in values}, set(ready.HARDWARE))
        for row in values:
            reps = [row[f"rep{i}_s"] for i in (1, 2, 3)]
            self.assertEqual(ready.mean_ci(reps), (row["mean_s"], row["sd_s"], row["ci_half_s"]))

    def test_bad_source_rejected(self):
        for defect in ("duplicate", "missing", "failed", "mean", "schedule", "overlap", "report"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                for name in ("runs.csv", "inferences.csv", "report.json"):
                    shutil.copyfile(ready.DATA/name, path/name)
                runs = ready.read_csv(path/"runs.csv")
                if defect == "duplicate":
                    runs.append(runs[0])
                elif defect == "missing":
                    runs.pop(0)
                elif defect == "failed":
                    runs[0]["exit_status"] = "1"
                elif defect == "mean":
                    runs[0]["measured_mean_ms"] = "1"
                elif defect == "overlap":
                    runs[0]["overlap"] = "4"
                elif defect == "report":
                    (path/"report.json").write_text('{"status":"error","errors":["bad"]}')
                else:
                    calls = ready.read_csv(path/"inferences.csv")
                    calls[2]["phase"] = "warmup"
                    ready.write_csv(path/"inferences.csv", calls)
                ready.write_csv(path/"runs.csv", runs)
                with self.assertRaises(ValueError):
                    ready.collect(path)

    def test_native_layout_no_dots_or_captions(self):
        ready.style()
        rows, _ = ready.collect()
        for models, width in ((ready.MODELS, 150), (("mlp",), 73.5)):
            fig = ready.figure(rows, models)
            self.assertAlmostEqual(fig.get_figwidth()*25.4, width)
            self.assertFalse(fig.texts)
            for ax in fig.axes:
                self.assertEqual(len(ax.patches), 12)
                self.assertEqual(ax.get_yscale(), "linear")
                self.assertFalse(ax.texts)
                self.assertTrue(all(type(c).__name__ != "PathCollection" for c in ax.collections))
                self.assertIn(ax.get_title(loc="left"), ready.MODEL_LABELS.values())
            ready.plt.close(fig)

    @unittest.skipUnless(shutil.which("pdflatex") and shutil.which("pdftocairo"), "TeX tools required")
    def test_real_render_dimensions_fonts_and_pgf(self):
        with tempfile.TemporaryDirectory() as directory:
            ready.main(["--out-dir", directory])
            for name, width in (("readiness_latency_1x2", 150), ("mlp_latency", 73.5),
                                ("transformer_latency", 73.5)):
                path = Path(directory)/name
                pgf = path.with_suffix(".pgf").read_text()
                self.assertIn(f"{width/25.4:.6f}in", pgf)
                self.assertNotIn("\\caption", pgf)
                self.assertNotIn("\\includegraphics", pgf)
                info = subprocess.check_output(["pdfinfo", str(path.with_suffix(".pdf"))], text=True)
                size = next(line for line in info.splitlines() if line.startswith("Page size:")).split()
                self.assertAlmostEqual(float(size[2])*25.4/72, width, places=2)
                self.assertAlmostEqual(float(size[4])*25.4/72, ready.HEIGHT_MM, places=2)
                fonts = subprocess.check_output(["pdffonts", str(path.with_suffix(".pdf"))], text=True)
                entries = fonts.splitlines()[2:]
                self.assertTrue(entries)
                self.assertTrue(all("Type 1" in line and "yes" in line and "LMRoman" in line for line in entries))
                self.assertGreater(path.with_suffix(".png").stat().st_size, 1000)

    def test_chaining_coverage_matches_canonical_and_controller_logs(self):
        coverage = ready.read_csv(ready.ROOT/"figures/ch7_smartsim_chaining_draft/coverage.csv")
        runs = ready.read_csv(ready.DATA/"runs.csv")
        self.assertEqual(len(coverage), 24)
        self.assertEqual(len({(r["model"], r["placement"], r["chain"]) for r in coverage}), 24)
        for row in coverage:
            if row["chain"] != "c0":
                self.assertEqual(row["status"], "absent")
                self.assertEqual(row["rep_means_ms"], "")
                continue
            selected = sorted((r for r in runs if r["provider"] == "smartsim"
                               and r["layout"] == "shared" and r["model"] == row["model"]
                               and r["placement"] == row["placement"]), key=lambda r: r["rep"])
            self.assertEqual(len(selected), 3)
            self.assertEqual(row["cells"], ";".join(r["cell"] for r in selected))
            for mean, run in zip(row["rep_means_ms"].split(";"), selected):
                self.assertAlmostEqual(float(mean), float(run["measured_mean_ms"]))
                log = (ready.ROOT/run["run_dir"]/"controller.log").read_text()
                for label, value in (("Threads per Queue", row["tpq"]),
                                     ("Intra-op Threads", "4"), ("Inter-op Threads", "1"),
                                     ("DB Layout", "shared")):
                    self.assertIn(f"{label}: {value}\n", log)


if __name__ == "__main__":
    unittest.main()
