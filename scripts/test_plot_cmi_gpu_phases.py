import itertools
import unittest

import plot_cmi_gpu_phases as plot


class PhaseTests(unittest.TestCase):
    def test_same_workgroup_not_phase_maxima_or_gpu_sum(self):
        rows = [dict(workgroup=0, a=9, b=1), dict(workgroup=1, a=2, b=9)]
        self.assertIs(plot.select_workgroup(rows, ("a", "b")), rows[1])
        self.assertEqual(sum(plot.select_workgroup(rows, ("a", "b"))[p] for p in ("a", "b")), 11)
        self.assertIs(plot.select_workgroup(list(reversed(rows)), ("a",)), rows[0])
        self.assertEqual(plot.select_workgroup([dict(workgroup=1, a=1), dict(workgroup=0, a=1)], ("a",))["workgroup"], 0)

    @classmethod
    def setUpClass(cls):
        cls.groups, cls.calls, cls.values, cls.coverage, cls.metadata = plot.collect()

    def test_complete_canonical_data_and_coverage(self):
        self.assertEqual(len(self.groups), 900)
        self.assertEqual(len(self.calls), 720)
        self.assertEqual(len(self.values), 192)
        self.assertEqual(len(self.coverage), 288)
        self.assertEqual(sum(r["rendered"] for r in self.coverage), 72)
        self.assertEqual({r["provider"] for r in self.coverage if r["rendered"]}, {"aix"})
        self.assertEqual({r["call"] for r in self.calls}, set(range(2, 7)))
        self.assertEqual(len(self.metadata["cells"]), 72)
        self.assertGreater(self.metadata["negative_residual_workgroup_calls"], 0)

    def test_residual_and_nonoverlapping_scope_definitions(self):
        self.assertNotIn("device_cpu_ms", plot.FIVE)
        self.assertNotIn("forward_ms", plot.HOST)
        for row in self.groups:
            self.assertAlmostEqual(row["device_minus_cuda_ms"], row["device_cpu_ms"]-row["gpu_sum_ms"])
            self.assertAlmostEqual(row["host_stage_sum_ms"]-row["measured_five_sum_ms"], row["device_minus_cuda_ms"])

    def test_winners_and_exact_repetition_means(self):
        for row in self.calls:
            group = [r for r in self.groups if r["cell"] == row["cell"] and r["call"] == row["call"]]
            self.assertEqual(row["workgroup"], plot.select_workgroup(group, plot.SCOPES[row["scope"]])["workgroup"])
        for row in self.values:
            subset = [r for r in self.calls if all(r[k] == row[k] for k in ("model", "placement", "layout", "scope"))]
            for rep in range(1, 4):
                mean = sum(r[row["phase"]] for r in subset if int(r["rep"]) == rep)/5
                self.assertAlmostEqual(row[f"rep{rep}_ms"], mean)
            self.assertAlmostEqual(row["mean_ms"], sum(row[f"rep{rep}_ms"] for rep in range(1, 4))/3)

    def test_missing_duplicate_or_warmup_calls_rejected(self):
        for damaged in (self.calls[1:], self.calls+[self.calls[0]],
                        [dict(r, call=0) if i == 0 else r for i, r in enumerate(self.calls)]):
            with self.assertRaises(ValueError):
                plot.summarize(damaged)

    def test_figure_width_bar_counts_and_phase_only_labels(self):
        for scope, models in itertools.product(plot.SCOPES, (plot.style.MODELS, ("transformer",), ("mlp",))):
            fig = plot.render(self.values, scope, models)
            self.assertAlmostEqual(fig.get_figwidth()*25.4, 150)
            self.assertAlmostEqual(fig.get_figheight()*25.4, 62)
            self.assertEqual(self.metadata["dimensions_mm"],
                             dict(width=150, combined_height=62, standalone_height=62))
            self.assertEqual(fig.texts, [])
            self.assertLessEqual(fig.subplotpars.bottom, .20)
            self.assertEqual(len(fig.axes), len(models))
            for ax, model in zip(fig.axes, models):
                self.assertEqual(len(ax.texts), 0)
                self.assertEqual(ax.get_title(), "")
                self.assertEqual(ax.get_title(loc="left"),
                                 plot.style.MODEL_LABELS[model] if len(models) > 1 else "")
                self.assertEqual(len(ax.patches), 12*len(plot.SCOPES[scope]))
                self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()],
                                 [plot.LABELS[p] for p in plot.SCOPES[scope]])
                self.assertEqual(ax.get_ylim()[0], 0)
                self.assertEqual(ax.get_ylabel(), "Measured stage sum (s)")
            plot.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
