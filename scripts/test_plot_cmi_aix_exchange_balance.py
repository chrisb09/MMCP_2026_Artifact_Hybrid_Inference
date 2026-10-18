"""Run only this suite: python -m unittest discover -s scripts -p test_plot_cmi_aix_exchange_balance.py."""

import unittest
from unittest.mock import patch

import plot_cmi_aix_exchange_balance as plot
from matplotlib.figure import Figure


class ExchangeBalanceTests(unittest.TestCase):
    def test_published_data_hashes_unchanged(self):
        expected = {
            'per_rank_durations.csv': 'e39f525e23239a322371455510b14a8204f6f4c7bd1d24a88b9af2e99b0d5aba',
            'workgroup_durations.csv': '0fbd5075a964a11f6ec59e50ba3246fe04fde0e5af279bb4030c489f2824cd8c',
            'balance_summary.csv': 'e87e388d5bb98de7fe3b87abbe238b37c00d37bd8d0a6acf8246f4e497d67f33',
        }
        for name, digest in expected.items():
            self.assertEqual(plot.base.sha256(plot.base.ROOT/'figures/aix_exchange_balance_draft'/name), digest)

    def test_render_no_narrative_and_crop_preserves_axes(self):
        comparisons, _, _, summaries, _ = plot.collect(
            ['mlp'], ['96c4g'], ['default'], ['p2p1'])
        original_save = plot.base.save
        def inspect(fig, out, name):
            self.assertEqual(fig.texts, [fig._suptitle])
            self.assertEqual([len(ax.texts) for ax in fig.axes], [8, 2, 4])
            self.assertTrue(all(t.get_text().replace('.', '').isdigit() for ax in fig.axes for t in ax.texts))
            self.assertEqual(fig.axes[1].get_ylim(), (0, 1))
            self.assertEqual(fig.axes[0].get_ylim(), fig.axes[2].get_ylim())
            before = [ax.bbox.bounds for ax in fig.axes]
            original_save(fig, out, name)
            self.assertEqual(savefig.call_count, 2)
            bounds = savefig.call_args.kwargs['bbox_inches']
            self.assertGreater(bounds.y0, .4)
            self.assertEqual((bounds.x0, bounds.x1, bounds.y1), (0, 12, 4.2))
            self.assertEqual([ax.bbox.bounds for ax in fig.axes], before)
            self.assertAlmostEqual(min(ax.get_tightbbox(fig.canvas.get_renderer()).y0/fig.dpi
                                       for ax in fig.axes)-bounds.y0, .08)
        with patch.object(Figure, 'savefig') as savefig, \
                patch.object(plot.base, 'save', side_effect=inspect):
            plot.render(comparisons[0], summaries, plot.base.ROOT)

    def endpoints(self, offset=0):
        return [dict(event="ml_step_"+side, time_s=str(offset+time),
                     **{key: "0" for key in plot.base.IDENTITY})
                for side, time in (("start", 1), ("end", 3))]

    def test_balance(self):
        self.assertEqual(plot.summarize([dict(duration_ms=10)]*4)["balance"], 1)
        self.assertEqual(plot.summarize([dict(duration_ms=v) for v in (10, 10, 10, 20)])["balance"], .625)
        with self.assertRaises(ValueError):
            plot.summarize([dict(duration_ms=10)]*3)

    def test_rank_local_offset_invariance_and_worker_zero(self):
        for offset in (0, 100, 100000):
            self.assertEqual(plot.rank_duration(self.endpoints(offset), "p2p1")["duration_ms"], 2000)

    def test_bad_endpoints(self):
        rows = self.endpoints()
        for damaged in (rows[:1], rows+[rows[0]], rows+[rows[1]],
                        [rows[0], dict(rows[1], sample_count="1")],
                        [rows[0], dict(rows[1], time_s="0")]):
            with self.subTest(damaged=damaged), self.assertRaises(ValueError):
                plot.rank_duration(damaged, "p2pfull")

    def test_collective_no_double_count(self):
        worker = dict.fromkeys(plot.STAGES, "0")
        worker.update(is_controller="0", diagnostic_barriers="0", gather_collective_s=".01",
                      scatter_collective_s=".09")
        controller = dict(worker, is_controller="1", device_cpu_s=".08", scatter_collective_s=".01")
        self.assertAlmostEqual(plot.rank_duration([worker], "coll")["duration_ms"], 100)
        self.assertAlmostEqual(plot.rank_duration([controller], "coll")["duration_ms"], 100)
        for damaged in (dict(worker, device_cpu_s=".08"), dict(worker, diagnostic_barriers="1"),
                        dict(worker, gather_arrival_wait_s=".1")):
            with self.assertRaises(ValueError):
                plot.rank_duration([damaged], "coll")
        with self.assertRaises(ValueError):
            plot.rank_duration([worker, worker], "coll")
        with self.assertRaises(ValueError):
            plot.rank_duration([], "coll")
        # One slow worker, not its faster controller, determines the WG maximum.
        slow = dict(worker, scatter_collective_s=".19")
        wg_max = max(plot.rank_duration([r], "coll")["duration_ms"] for r in (controller, slow))
        self.assertAlmostEqual(wg_max, 200)
        self.assertAlmostEqual(plot.summarize([dict(duration_ms=v) for v in (100, 100, 100, wg_max)])["balance"], .625)

    def test_all_selected_matches_coverage(self):
        comparisons, ranks, groups, summaries, metadata = plot.collect(
            ["transformer", "mlp"], ["96c4g", "96p96c4g"], ["default", "numa"], ["p2p1", "p2pfull"])
        self.assertEqual((len(comparisons), len(groups), len(summaries)), (16, 128, 32))
        self.assertEqual(len(metadata["cells"]), 24)
        for comparison in comparisons:
            for side in ("collective", "p2p"):
                for group in comparison[side]:
                    selected = [r for r in ranks if r["comparison"] == comparison["name"]
                                and r["cell"] == group["cell"] and r["workgroup"] == group["workgroup"]]
                    self.assertEqual(sorted(r["world_rank"] for r in selected), list(map(int, group["members"].split(";"))))
                    self.assertEqual(group["duration_ms"], max(r["duration_ms"] for r in selected))
                    self.assertTrue(all(r["source_path"] in metadata["sources"] for r in selected))
                    self.assertTrue(all(r["call"] == 6 and r["maia_step"] == 189 for r in selected))
            self.assertEqual([(r["members"], r["gpu_ordinal"]) for r in comparison["collective"]],
                             [(r["members"], r["gpu_ordinal"]) for r in comparison["p2p"]])
        self.assertTrue(all(0 < r["balance"] <= 1 for r in summaries))
        # Corrupt a worker only: controller-only validation must not conceal the error.
        original = plot.base.read_csv
        worker_path = next(r["source_path"] for r in ranks if not r["is_controller"])
        for fault, message in (("metadata", "metadata"), ("missing", "Missing rank timing"),
                               ("duplicate", "collective rank call"), ("wg_rank", "CSV workgroup rank")):
            def corrupt(path):
                rows = original(path)
                if str(path).endswith(worker_path):
                    if fault == "metadata":
                        rows = [dict(r, world_rank="9999") for r in rows]
                    elif fault == "missing":
                        rows = [r for r in rows if r["call"] != "6"]
                    elif fault == "duplicate":
                        rows = rows + [r for r in rows if r["call"] == "6"]
                    else:
                        rows = [dict(r, workgroup_rank="9999") for r in rows]
                return rows
            with self.subTest(fault=fault), patch.object(plot.base, "read_csv", side_effect=corrupt):
                with self.assertRaisesRegex(ValueError, message):
                    plot.collect(["transformer"], ["96c4g"], ["default"], ["p2p1"])

    def test_warmup_rejected(self):
        with self.assertRaisesRegex(ValueError, "measured call"):
            plot.collect(["mlp"], ["96c4g"], ["default"], ["p2p1"], call=0)


if __name__ == "__main__":
    unittest.main()
