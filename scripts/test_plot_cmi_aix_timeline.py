"""Local validation: python -m unittest discover -s scripts -p test_plot_cmi_aix_timeline.py."""

import unittest

import plot_cmi_aix_timeline as plot


class TimelineTests(unittest.TestCase):
    def rows(self):
        rows = []
        for base in ("ml_step", "range_inference", "torch_h2d", "torch_forward", "torch_d2h"):
            for side, time in (("start", "1"), ("end", "2")):
                rows.append(dict(event=base+"_"+side, time_s=time,
                                 **{key: "-1" if base == "ml_step" else "0" for key in plot.IDENTITY}))
        return rows

    def test_unordered_pairs(self):
        pairs = plot.pair_intervals(list(reversed(self.rows())), 6, 2)
        self.assertEqual(len(pairs), 5)
        self.assertTrue(all(r["duration_ms"] == 1000 for r in pairs))

    def test_bad_endpoints(self):
        rows = self.rows()
        for damaged in (rows + [rows[0]], rows[:-1],
                        rows[:-1] + [dict(rows[-1], time_s="0")],
                        rows[:-2], rows[:-1] + [dict(rows[-1], sample_count="1")]):
            with self.subTest(damaged=damaged):
                with self.assertRaises(ValueError):
                    plot.pair_intervals(damaged, 6, 2)

    def test_default_batch(self):
        comparisons, metrics, events, metadata = plot.collect(
            ["transformer", "mlp"], ["96c4g", "96p96c4g"], ["default", "numa"], ["p2p1", "p2pfull"])
        self.assertEqual((len(comparisons), len(metrics), metadata["workgroup_comparisons"]), (16, 128, 64))
        self.assertTrue(all(r["call"] == 6 and r["maia_step"] == 189 and r["inference_call"] == 7 for r in metrics))
        self.assertTrue(all(r["end_s"] >= r["start_s"] and r["duration_ms"] >= 0 for r in events))
        self.assertTrue(all(r["controller_start_s"] == "" for r in metrics if r["mode"] == "coll"))
        for row in metrics:
            self.assertAlmostEqual(row["gpu_sum_ms"], sum(row[p+"_ms"] for p in plot.PHASES))

    def test_one_gpu_and_affine(self):
        comparisons, metrics, _, _ = plot.collect(
            ["transformer", "mlp"], ["24c1g", "24p24c1g"], ["default", "affine", "numa"], ["p2p1", "p2pfull"])
        self.assertEqual(len(comparisons), 24)
        self.assertEqual(len(metrics), 48)
        self.assertTrue(all(row["gpu_ordinal"] == 0 for row in metrics))
        comparisons, metrics, _, _ = plot.collect(
            ["transformer", "mlp"], ["96c4g", "96p96c4g"], ["affine"], ["p2p1", "p2pfull"])
        self.assertEqual(len(comparisons), 8)
        self.assertEqual(len(metrics), 64)

    def test_warmup_rejected(self):
        with self.assertRaisesRegex(ValueError, "measured call"):
            plot.collect(["mlp"], ["96c4g"], ["default"], ["p2p1"], call=0)

    def test_exact_selection_required(self):
        with self.assertRaisesRegex(ValueError, "selected"):
            plot.collect(["missing-model"], ["96c4g"], ["default"], ["p2p1"])


if __name__ == "__main__":
    unittest.main()
