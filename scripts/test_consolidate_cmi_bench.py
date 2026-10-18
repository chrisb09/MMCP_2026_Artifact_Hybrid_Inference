"""Focused, stdlib-only regression tests; all writes stay in temporary directories."""
import math
from pathlib import Path
import tempfile
import unittest

import consolidate_cmi_bench as bench


class ConsolidateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "run"
        self.directory.mkdir()
        self.spec = bench.expected_matrix()["aix-24p24c1g-default-coll-transformer-r1"]
        self.selection = {"jobid": "101", "selection": "original", "lineage": []}
        self.info = dict(
            CELL=self.spec["cell"], REP=1, SOLVER_EXIT_STATUS=0, EXIT_STATUS=0,
            ML_CUBE_OVERLAP=0, RUN_STEPS=200, SOLUTION_INTERVAL=10000,
            MODEL=bench.MODELS["transformer"], PROVIDER="AIX", HET_CPU_TASKS=24,
            HET_GPU_TASKS=24, HET_GPUS=1, AIX_COMM="collective",
            AIX_CREDITS="unset", AIX_LAYOUT_LIB="default", MAIA_AIX_BATCHSIZE=50000,
            BENCHMARK_SOLVER_WALL_SECONDS="9.25", BATCH_HOST="cpu",
            HET_GPU_NODELIST="gpu", NODE_RAM_PEAK_MIB_cpu=2048,
            NODE_RAM_PEAK_MIB_gpu=4096)
        self.log = (
            "ML_WORKLOAD_LOG cubes_per_rank=2 samples_per_rank=16 "
            "cubes_aggregate=96 samples_aggregate=768 overlap=0 solver_ranks=48\n"
            + "".join(f"ML_STEP_TIMING step={step} ml_ms={value}\n"
                      for step, value in zip((15, 44, 73, 102, 131, 160, 189),
                                             ("100.00", "200.0", "1", "2", "3", "4", "5"))))

    def parse(self):
        (self.directory / "run.info").write_text(
            "".join(f"{key}={value}\n" for key, value in self.info.items()))
        (self.directory / "solver.log").write_text(self.log)
        return bench.parse_selected_run(self.root, self.directory, self.spec, self.selection)

    def test_matrix(self):
        matrix = bench.expected_matrix()
        self.assertEqual(len(matrix), 288)
        configs = {tuple(row[key] for key in bench.CONFIG_FIELDS) for row in matrix.values()}
        self.assertEqual(len(configs), 96)
        for row in matrix.values():
            self.assertEqual(row["solver_ranks"], row["cpu_solver_ranks"] + row["gpu_solver_ranks"])

    def test_seven_calls_and_heterogeneous_workload(self):
        row, calls = self.parse()
        self.assertEqual([call["step"] for call in calls], [15, 44, 73, 102, 131, 160, 189])
        self.assertEqual([call["call"] for call in calls], list(range(1, 8)))
        self.assertEqual([call["phase"] for call in calls], ["warmup"] * 2 + ["measured"] * 5)
        self.assertEqual((row["n_inferences"], row["n_measured"]), (7, 5))
        self.assertEqual(row["warmup_ms"], "100.00;200.0")
        self.assertEqual(row["measured_ms"], "1;2;3;4;5")
        self.assertEqual(row["measured_mean_ms"], 3)
        self.assertAlmostEqual(row["measured_stdev_ms"], math.sqrt(2.5))
        self.assertEqual(tuple(row[key] for key in bench.WORKLOAD_FIELDS), (2, 16, 96, 768, 48))

    def test_reject_missing_duplicate_or_malformed_timing(self):
        original = self.log
        for log in (original.replace("ML_STEP_TIMING step=189 ml_ms=5\n", ""),
                    original + "ML_STEP_TIMING step=189 ml_ms=5\n",
                    original.replace("ml_ms=5", "ml_ms=invalid")):
            with self.subTest(log=log), self.assertRaisesRegex(ValueError, "expected exactly steps"):
                self.log = log
                self.parse()

    def test_reject_nonzero_status(self):
        for key in ("SOLVER_EXIT_STATUS", "EXIT_STATUS"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.info[key] = 1
                self.parse()
            self.info[key] = 0

    def test_reject_wrong_workload_rank_total(self):
        self.log = self.log.replace("solver_ranks=48", "solver_ranks=24")
        with self.assertRaisesRegex(ValueError, "invalid workload"):
            self.parse()

    def test_missing_gpu_is_explicit_not_zero(self):
        row, _ = self.parse()
        resources, warnings = bench.extract_resources(self.root, self.directory, row)
        gpu, = [resource for resource in resources if resource["metric"] == "gpu_vram"]
        self.assertEqual((gpu["host"], gpu["device"], gpu["expected"]), ("gpu", 0, True))
        self.assertEqual((gpu["status"], gpu["peak_mib"], gpu["peak_gib"]), ("missing", "", ""))
        self.assertTrue(any("missing gpu_vram host=gpu device=0" in warning for warning in warnings))

    def test_retry_precedence_and_intermediate_smartsim_lineage(self):
        spec = bench.expected_matrix()["smartsim-24c1g-shared-transformer-r1"]
        cell, group = spec["cell"], spec["group"]
        results = self.root / "results"
        results.mkdir()
        ledgers = {
            "cmi_bench_grouped_jobids.txt": f"grp-{group},101\n",
            "cmi_bench_quota_retry_jobids.txt": f"quota-retry-grp-{group},103\n",
            "cmi_bench_smartsim_retry_jobids.csv":
                f"cell,original_group_jobid,retry_jobid,reason\n{cell},101,102,smart\n",
            "cmi_bench_quota_retry_cells.csv":
                f"cell,original_jobid,reason\n{cell},102,quota\n",
            "cmi_bench_shutdown_retry_cells.csv":
                f"cell,failed_jobid,retry_jobid,reason\n{cell},103,104,shutdown\n",
            "cmi_bench_final_retry_cells.csv":
                f"cell,original_jobid,retry_jobid,reason\n{cell},104,105,final\n",
            "final_retry_group_24c1g_transformer_r1_cells.csv": f"{cell},0\n",
        }
        for name, text in ledgers.items():
            (results / name).write_text(text)
        sources = {}
        selected = bench.select_cells(self.root, {cell: spec}, sources)[cell]
        self.assertEqual((selected["jobid"], selected["original_jobid"]), ("105", "101"))
        self.assertEqual((selected["selection"], selected["retry_reason"]), ("final_retry", "final"))
        self.assertEqual([entry["jobid"] for entry in selected["lineage"]],
                         ["101", "102", "103", "104", "105"])
        self.assertEqual([entry["selection"] for entry in selected["lineage"]],
                         ["original", "smartsim_retry", "quota_retry", "shutdown_retry", "final_retry"])
        self.assertEqual(len(sources), len(ledgers))
        path = results / "cmi_bench_quota_retry_cells.csv"
        path.write_text(ledgers[path.name].replace(",102,", ",101,"))
        with self.assertRaisesRegex(ValueError, "lineage 101 != selected 102"):
            bench.select_cells(self.root, {cell: spec}, {})
        path.write_text(ledgers[path.name])
        (results / "final_retry_group_24c1g_transformer_r1_cells.csv").write_text(f"{cell},1\n")
        with self.assertRaisesRegex(ValueError, "selected status is '1', not 0"):
            bench.select_cells(self.root, {cell: spec}, {})

    def test_summary_uses_three_independent_rep_means_and_sample_sd(self):
        runs = [dict(self.spec, rep=rep, measured_mean_ms=mean,
                     measured_median_ms=median, wall_seconds=str(rep * 10),
                     measured_ms="999;999;999;999;999")
                for rep, mean, median in ((1, 3, 2), (2, 6, 4), (3, 12, 9))]
        summary, = bench.summarize(runs)
        self.assertEqual((summary["n_reps"], summary["mean_ms"]), (3, 7))
        self.assertAlmostEqual(summary["sample_sd_ms"], math.sqrt(21))
        self.assertEqual((summary["min_ms"], summary["max_ms"]), (3, 12))
        self.assertEqual(summary["mean_rep_median_ms"], 5)
        self.assertAlmostEqual(summary["sample_sd_rep_median_ms"], math.sqrt(13))
        self.assertEqual(summary["mean_wall_seconds"], 20)
        for invalid in (runs[:2], [runs[0], runs[0], runs[2]]):
            with self.subTest(reps=[row["rep"] for row in invalid]):
                with self.assertRaisesRegex(ValueError, "expected reps 1,2,3"):
                    bench.summarize(invalid)


if __name__ == "__main__":
    unittest.main()
