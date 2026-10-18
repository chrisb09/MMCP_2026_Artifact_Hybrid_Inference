import itertools
import unittest

import plot_cmi_ch6_phases as plot


class ChapterSixPhasesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups, cls.calls, cls.values, cls.coverage, cls.metadata = plot.collect()

    def test_only_actual_default_aix_is_measured(self):
        self.assertEqual(len(self.values), 40)
        self.assertEqual({r["provider"] for r in self.values}, {"aix"})
        self.assertEqual({r["layout"] for r in self.values}, {"default"})
        self.assertEqual(len(self.calls), 120)
        self.assertEqual(len(self.groups), 300)
        self.assertEqual(len(self.metadata["cells"]), 24)
        self.assertEqual({r["call"] for r in self.calls}, set(range(2, 7)))

    def test_missing_is_not_zero_or_measured(self):
        missing = [r for r in self.coverage if r["status"] != "measured"]
        self.assertEqual(len(missing), 20)
        self.assertTrue(all(r["mean_ms"] == "" and r["measured_phase_count"] == 0 for r in missing))
        per_gpu = [r for r in missing if r["provider"] == "ss_pergpu"]
        self.assertEqual(sum(r["required_runs"]-r["selected_runs"] for r in per_gpu), 12)
        self.assertTrue(all("4g" in r["placement"] for r in per_gpu))

    def test_same_workgroup_and_exact_means(self):
        for call in self.calls:
            candidates = [r for r in self.groups if (r["cell"], r["call"]) == (call["cell"], call["call"])]
            self.assertEqual(call["workgroup"], plot.phases.select_workgroup(candidates, plot.phases.FIVE)["workgroup"])
        for row in self.values:
            for rep in range(1, 4):
                calls = [r for r in self.calls if (r["model"], r["placement"], int(r["rep"]))
                         == (row["model"], row["placement"], rep)]
                self.assertEqual(len(calls), 5)
                self.assertAlmostEqual(row[f"rep{rep}_ms"], sum(r[row["phase"]] for r in calls)/5)
            self.assertAlmostEqual(row["mean_ms"], sum(row[f"rep{rep}_ms"] for rep in range(1, 4))/3)

    def test_adjacent_zero_based_phase_columns_and_coverage_annotations(self):
        for model, hw in itertools.product(plot.style.MODELS, plot.style.HARDWARE):
            fig = plot.render(self.values, self.coverage, model, hw)
            ax = fig.axes[0]
            measured = [p for p in ax.patches if p.get_gid() == "measured-phase"]
            missing = [t for t in ax.texts if t.get_text() == "No phase data"]
            self.assertEqual(len(measured), 5)
            self.assertEqual(len(ax.patches), 5)
            self.assertEqual(len(missing), 3 if "4g" in hw else 2)
            for text in missing:
                self.assertEqual(text.get_transform(), ax.get_xaxis_transform())
            values = {r["phase"]: r["mean_ms"]/1000 for r in self.values
                      if r["model"] == model and r["placement"] == hw}
            for j, (bar, phase) in enumerate(zip(measured, plot.phases.FIVE)):
                self.assertEqual(bar.get_y(), 0)
                self.assertAlmostEqual(bar.get_height(), values[phase])
                self.assertAlmostEqual(bar.get_x()+bar.get_width()/2, (j-2)*.19)
                if j:
                    self.assertGreaterEqual(bar.get_x(), measured[j-1].get_x()+measured[j-1].get_width())
            expected = max(values.values())
            self.assertAlmostEqual(ax.dataLim.ymax, expected)
            self.assertAlmostEqual(fig.get_figwidth()*25.4, 150)
            self.assertAlmostEqual(fig.get_figheight()*25.4, 60)
            self.assertEqual(self.metadata["dimensions_mm"], dict(width=150, height=60))
            self.assertEqual(fig.texts, [])
            self.assertEqual(ax.get_title(loc="left"), "")
            self.assertEqual(ax.get_title(), "")
            self.assertEqual(len(ax.texts), 5 + len(missing))
            self.assertLess(fig.subplotpars.bottom, .20)
            self.assertEqual(ax.get_ylim()[0], 0)
            self.assertEqual(ax.get_ylabel(), "Phase duration (s)")
            self.assertEqual([t.get_text() for t in fig.legends[0].get_texts()], list(plot.PHASE_LABELS))
            plot.style.plt.close(fig)

    def test_forward_color_matches_cpu_not_generic_aix_provider(self):
        self.assertEqual(plot.COLORS[2], plot.SEMANTICS["cpu_reference"]["PhyDLL"]["Forward"])
        self.assertNotIn("device_cpu_ms", plot.phases.FIVE)


if __name__ == "__main__":
    unittest.main()
