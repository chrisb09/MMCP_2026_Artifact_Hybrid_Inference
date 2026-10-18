"""Run with python -m unittest discover -s scripts -p test_plot_cmi_aix_rank_timeline.py."""

import unittest
from unittest.mock import patch

import plot_cmi_aix_rank_timeline as plot


class RankTimelineTests(unittest.TestCase):
    def rows(self):
        events = ("ml_step_start", "input_credit_wait_start", "input_send_start", "input_send_complete",
                  "result_wait_start", "result_received", "ml_step_end")
        return [dict(event=event, time_s=str(100+i), range_first_rank=str(i),
                     peer_workgroup_rank=str(i-1), sample_count=str(i*3)) for i, event in enumerate(events)]

    def test_family_pairing_ignores_different_endpoint_metadata(self):
        intervals = plot.worker_intervals(list(reversed(self.rows())))
        self.assertEqual(len(intervals), 4)
        self.assertEqual(intervals[0]["duration_ms"], 6000)
        self.assertEqual(intervals[2]["duration_ms"], 1000)

    def test_damaged_worker_endpoints(self):
        rows = self.rows()
        for damaged in (rows[:-1], rows+[rows[2]], rows[:3]+[dict(rows[3], time_s="99")]+rows[4:]):
            with self.subTest(damaged=damaged), self.assertRaises(ValueError):
                plot.worker_intervals(damaged)

    def test_causal_bounds_and_independent_clock_origins(self):
        self.assertEqual(plot.causal_bounds(10, 14, 102, 107), (-92, -88, -90))
        self.assertEqual(plot.causal_bounds(1010, 1014, 102, 107), (908, 912, 910))
        with self.assertRaisesRegex(ValueError, "Infeasible.*--alignment local"):
            plot.causal_bounds(15, 14, 102, 107)

    def test_peer_ids_are_local_not_world_ranks(self):
        row = dict(event="input_ready", peer_workgroup_rank="3", range_first_rank="3",
                   range_end_rank="4", time_s="12")
        self.assertEqual(plot.controller_endpoint([row], "input_ready", 3), 12)
        with self.assertRaises(ValueError):
            plot.controller_endpoint([row], "input_ready", 9)
        with self.assertRaisesRegex(ValueError, "range/peer"):
            plot.controller_endpoint([dict(row, range_first_rank="9")], "input_ready", 3)

    def test_actual_default_batch_24_rank_coverage(self):
        comparisons, metrics, events, metadata, intervals, alignments = plot.collect(
            ["transformer", "mlp"], ["default", "numa"], ["p2p1", "p2pfull"])
        self.assertEqual((len(comparisons), len(metrics), len(intervals), len(alignments)), (8, 16, 736, 184))
        self.assertTrue(all(r["call"] == 6 and r["maia_step"] == 189 for r in intervals))
        self.assertEqual(len([p for p in metadata["sources"] if "aix_p2p_timeline_rank_" in p]), 192)
        for comparison in comparisons:
            mapping = comparison["rank_mapping"]
            self.assertEqual(set(mapping), set(range(24)))
            self.assertEqual(set(mapping.values()), set(range(24)))
            selected = [r for r in alignments if r["comparison"] == comparison["name"]]
            self.assertEqual({r["world_rank"] for r in selected}, set(mapping)-{comparison["p2p"][0]["controller"]})
        for r in alignments:
            self.assertLessEqual(r["delta_low_s"], r["delta_est_s"])
            self.assertLessEqual(r["delta_est_s"], r["delta_high_s"])
            self.assertLessEqual(r["c_grant_s"], r["w_send_start_s"]+r["delta_est_s"]+1e-12)
            self.assertLessEqual(r["w_send_start_s"]+r["delta_est_s"], r["c_ready_s"]+1e-12)
            self.assertLessEqual(r["c_ready_s"], r["w_send_complete_s"]+r["delta_est_s"]+1e-12)
        for r in intervals:
            self.assertGreaterEqual(r["duration_ms"], 0)
            self.assertAlmostEqual(r["display_end_ms"]-r["display_start_ms"], r["duration_ms"])

    def test_actual_local_has_own_zeros_and_no_clock_claim(self):
        _, _, _, _, intervals, alignments = plot.collect(["mlp"], ["numa"], ["p2pfull"], alignment="local")
        self.assertTrue(all(r["delta_est_s"] is None and r["delta_low_s"] is None for r in alignments))
        self.assertTrue(all(r["display_start_ms"] == 0 for r in intervals if r["family"] == "ml_step"))
        self.assertTrue(all(r["display_start_ms"] == r["own_start_ms"] for r in intervals))

    def test_render_limits_markers_and_local_separation(self):
        for alignment in ("causal", "local"):
            comparisons, _, events, _, intervals, alignments = plot.collect(
                ["mlp"], ["numa"], ["p2pfull"], alignment=alignment)
            with patch.object(plot.base, "save") as save:
                plot.render_detail(comparisons[0], events, intervals, alignments, None)
            fig = save.call_args.args[0]
            try:
                gpu, collective, workers = fig.axes
                self.assertEqual(len(workers.get_yticklabels()), 23)
                low, high = workers.get_xlim()
                self.assertLess(low, min(r["display_start_ms"]-r["uncertainty_ms"] for r in intervals))
                self.assertGreater(high, max(r["display_end_ms"]+r["uncertainty_ms"] for r in intervals))
                self.assertIn("independent zero", collective.get_title(loc="left"))
                self.assertTrue(any(line.get_marker() == "o" for line in gpu.lines))
                self.assertEqual(any(line.get_marker() == "o" for line in workers.lines), alignment == "causal")
            finally:
                plot.base.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
