"""Run with unittest discover -s scripts -p test_plot_cmi_pipeline_spans.py."""

import copy
import itertools
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import plot_cmi_pipeline_spans as spans


def synthetic(origin=0):
    rows = []
    def add(event, time, peer=-1, first=-1, end=-1):
        rows.append(dict(event=event, time_s=str(origin+time), step="2", world_rank="7",
                         is_controller="1", workgroup_rank="0", peer_workgroup_rank=str(peer),
                         range_first_rank=str(first), range_end_rank=str(end),
                         sample_start="0", sample_count="2"))
    add("ml_step_start", 0)
    add("input_listen_start", 1)
    add("input_credit_send", 2, 1)
    add("input_ready", 3, 0)
    add("input_ready", 4, 1)
    for family in ("range_inference", "torch_h2d", "torch_forward", "torch_d2h"):
        add(family+"_start", 5, -1, 0, 2)
        add(family+"_end", 6, -1, 0, 2)
    add("result_send_start", 7, 0)
    add("result_send_start", 8, 1)
    add("result_send_complete", 10, 1)
    add("ml_step_end", 12)
    return rows


class SpanTests(unittest.TestCase):
    def test_local_envelope_not_ready_range_or_mlstep(self):
        for origin in (0, 10000):
            value = spans.exchange_span(synthetic(origin), 2, 7)
            self.assertEqual(value["exchange_ms"], 9000)
            self.assertEqual(value["start_event"], "input_listen_start")
            self.assertEqual(value["end_event"], "result_send_complete")
            self.assertEqual(value["inference_members"], 2)

    def test_missing_duplicate_and_local_membership_rejected(self):
        rows = synthetic()
        for event in ("input_listen_start", "input_credit_send", "input_ready",
                      "result_send_start", "result_send_complete", "torch_h2d_end"):
            index = next(i for i, r in enumerate(rows) if r["event"] == event)
            with self.subTest(event=event), self.assertRaises(ValueError):
                spans.exchange_span(rows[:index]+rows[index+1:], 2, 7)
        with self.assertRaises(ValueError):
            spans.exchange_span(rows+[rows[2]], 2, 7)
        changed = copy.deepcopy(rows)
        changed[0]["world_rank"] = "8"
        with self.assertRaises(ValueError):
            spans.exchange_span(changed, 2, 7)

    def test_formula_and_independent_critical_wg(self):
        p = spans.prediction
        groups = [dict(workgroup=0, **p.pipeline([50, 30, 200, 20, 80], 10)),
                  dict(workgroup=1, **p.pipeline([0, 0, 300, 0, 0], 10))]
        result = p.job_prediction(groups, 500)
        self.assertEqual(groups[0]["pipeline_ms"], 218)
        self.assertEqual((result["before_critical_wg"], result["after_critical_wg"]), (0, 1))
        self.assertEqual(result["after_envelope_ms"], 300)


class CanonicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = spans.collect()

    def test_coverage_and_all_members(self):
        observations, calls, summaries, meta = self.data
        self.assertEqual((len(observations), len(calls), len(summaries)), (2700, 360, 240))
        self.assertEqual(len(meta["selected_runs"]), 216)
        self.assertEqual(meta["counts"]["measured_workgroup_calls"], 1800)
        self.assertEqual({(r["layout"], r["model"], r["placement"]) for r in calls},
                         set(itertools.product(spans.LAYOUTS, spans.prediction.MODELS, spans.prediction.HARDWARE)))
        for g in observations:
            self.assertEqual(len(g["members"].split(";")), g["n_members"])
            if g["mode"] != "coll":
                for key in ("input_members", "inference_members", "output_members"):
                    self.assertEqual(g[key], g["n_members"])
                self.assertAlmostEqual(g["exchange_ms"], 1000*(g["end_s"]-g["start_s"]))

    def test_maxima_stage_identity_and_pipeline_only(self):
        observations, calls, *_ = self.data
        groups = {}
        for g in observations:
            groups.setdefault((g["cell"], g["call"]), []).append(g)
        for row in calls:
            coll = groups[row["collective_cell"], row["call"]]
            critical = max(coll, key=lambda g: g["stage_sum_ms"])
            self.assertEqual(row["collective_ms"], critical["stage_sum_ms"])
            self.assertAlmostEqual(sum(row[k] for k in spans.STAGE_FIELDS), row["collective_ms"])
            for k in spans.STAGE_FIELDS:
                self.assertEqual(row[k], critical[k])
            ideal = max(g["pipeline_ms"] for g in coll)
            for prefix in ("c1", "full"):
                self.assertEqual(row[prefix+"_ideal_ms"], ideal)
                measured = groups[row[prefix+"_cell"], row["call"]]
                self.assertEqual(row[prefix+"_measured_ms"], max(g["exchange_ms"] for g in measured))

    def test_statistics_and_current_hashes(self):
        _, calls, summaries, meta = self.data
        for s in summaries:
            selected = [r for r in calls if all(r[k] == s[k] for k in ("layout", "model", "placement"))]
            reps = [statistics.mean(r[s["metric"]] for r in selected if r["rep"] == rep)
                    for rep in range(1, 4)]
            self.assertEqual(s["mean_ms"], statistics.mean(reps))
            self.assertAlmostEqual(s["ci_high_ms"]-s["mean_ms"],
                                   spans.prediction.T95*statistics.stdev(reps)/math.sqrt(3))
        for path, digest in meta["sources"].items():
            self.assertEqual(spans.base.sha256(path), digest)

    def test_individual_phase_columns_and_own_cis(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex

        summaries = self.data[2]
        inspected = []
        panels = []

        def inspect(fig, path):
            self.assertFalse(fig.texts)
            self.assertTrue(all(not ax.get_title(loc="left") and not ax.get_title() for ax in fig.axes))
            self.assertAlmostEqual(fig.get_figheight()*25.4, 33.592)
            if path.parent.name == "panels":
                self.assertAlmostEqual(fig.get_figwidth()*25.4, 73.5)
                visible = [ax for ax in fig.axes if ax.get_visible()]
                self.assertEqual(len(visible), 1)
                self.assertAlmostEqual(visible[0].get_position().height*33.592, 20.592)
                self.assertEqual(sum(legend.get_visible() for legend in fig.legends), 1)
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                bounds = next(legend.get_window_extent(renderer) for legend in fig.legends if legend.get_visible())
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertLessEqual(bounds.x1, fig.bbox.x1)
                self.assertLess((bounds.y0-visible[0].bbox.y1)/fig.dpi*25.4, 3)
                panels.append(path.name)
                plt.close(fig)
                return
            left, right = fig.axes
            model, hw = next((model, hw) for model, hw in itertools.product(
                spans.prediction.MODELS, spans.prediction.HARDWARE)
                if path.name == f"{model}_{hw}_default_pipeline_spans")
            values = {s["metric"]: s for s in summaries if
                      (s["layout"], s["model"], s["placement"]) == ("default", model, hw)}
            self.assertEqual(len(left.patches), 5)
            self.assertEqual(len(left.collections), 5)
            for i, (bar, metric, color) in enumerate(zip(left.patches, spans.STAGE_FIELDS,
                    ("#4C78A8", "#72B7B2", "#F58518", "#54A24B", "#B279A2"))):
                v = values[metric]
                self.assertEqual(bar.get_y(), 0)  # No cumulative/stacked heights.
                self.assertAlmostEqual(bar.get_x()+bar.get_width()/2, i)
                self.assertAlmostEqual(bar.get_height(), v["mean_ms"]/1000)
                self.assertEqual(to_hex(bar.get_facecolor()).upper(), color)
                segment = left.collections[i].get_segments()[0]
                self.assertAlmostEqual(segment[0][1], v["ci_low_ms"]/1000)
                self.assertAlmostEqual(segment[1][1], v["ci_high_ms"]/1000)
            self.assertEqual([t.get_text() for t in left.get_xticklabels()],
                             ["Gather", "H2D", "Fwd", "D2H", "Scatter"])
            self.assertEqual([t.get_text() for t in right.get_xticklabels()],
                              ["Collective\n5-stage sum", "P2P\nCredit=1", "P2P\nFullcredit"])
            legend_labels = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
            self.assertIn("Predicted", legend_labels)
            self.assertIn("Collective 5-stage sum", legend_labels)
            self.assertNotIn("Predicted (ideal)", legend_labels)
            self.assertEqual(right.get_ylabel(), "Inference time (s)")
            self.assertAlmostEqual(left.get_position().height*33.592, 78*.44*.6)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            first, second = [legend.get_window_extent(renderer) for legend in fig.legends]
            self.assertAlmostEqual(first.y1, second.y1)
            self.assertLess(first.x1, second.x0)
            self.assertLess((first.y0-left.bbox.y1)/fig.dpi*25.4, 3)
            for ax, fields in ((left, spans.STAGE_FIELDS), (right, spans.METRICS[5:])):
                self.assertEqual(len(ax.texts), 5)
                for text, metric in zip(ax.texts, fields):
                    self.assertEqual(text.get_text(), f"{values[metric]['mean_ms']/1000:.3g}")
                    self.assertAlmostEqual(text.xy[1], values[metric]["ci_high_ms"]/1000)
                    self.assertLess(text.get_window_extent(renderer).y1, ax.bbox.y1)
            self.assertFalse(left.get_shared_y_axes().joined(left, right))
            self.assertEqual(left.yaxis.get_major_formatter()(-.2, 0), "-0.2")
            inspected.append(path.name)
            plt.close(fig)

        style = SimpleNamespace(apply=lambda: None, save=inspect)
        with patch.dict("sys.modules", {"thesis_style": style}), patch.object(Path, "is_file", return_value=True):
            names = spans.render(summaries, ["default"], Path("unused"))
        self.assertEqual(names, inspected)
        self.assertEqual(len(names), 8)
        self.assertEqual(len(panels), 16)


if __name__ == "__main__":
    unittest.main()
