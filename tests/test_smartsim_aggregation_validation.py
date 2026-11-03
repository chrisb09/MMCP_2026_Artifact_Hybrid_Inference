import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_smartsim_aggregation_reruns.py"
STEPS = (15, 44, 73, 102, 131, 160, 189)
WORKLOAD = (
    "ML_WORKLOAD_LOG cubes_per_rank=1152 samples_per_rank=3456 "
    "cubes_aggregate=27648 samples_aggregate=82944 cubeD=8 overlap=0 solver_ranks=24"
)


class TestAggregationValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scratch = self.root / "scratch"
        self.run_dir = self.scratch / "smartsim_gpu_123_abcd"
        self.run_dir.mkdir(parents=True)
        timings = "\n".join(f"ML_STEP_TIMING step={step} ml_ms={100 + step / 10}"
                            for step in STEPS)
        (self.run_dir / "solver.log").write_text(timings + "\n" + WORKLOAD + "\n")
        (self.run_dir / "config_smartsim.toml").write_text(
            "[library]\nbatch_size = 7000\nmin_batch_size = 7000\nmin_batch_timeout = 1\n")
        (self.run_dir / "run.info").write_text(
            "CELL=smartsim-24c1g-agg-transformer-r1\nREP=1\nRUN_STEPS=200\n"
            "SOLUTION_INTERVAL=10000\nML_CUBE_OVERLAP=0\nDEBUG_EXPORT=0\n"
            "EXIT_STATUS=0\nSOLVER_EXIT_STATUS=0\n"
            "SMARTSIM_AGGREGATION_ENABLED=1\n"
            "SMARTSIM_BATCH_SIZE_EFFECTIVE=7000\n"
            "SMARTSIM_MIN_BATCH_SIZE_EFFECTIVE=7000\n"
            "SMARTSIM_MIN_BATCH_TIMEOUT_MS_EFFECTIVE=1\n"
            "MODEL=./input/transformer_inference_scripted_fw2.pt\n"
            "PROVIDER=SMARTSIM\nMAIA_SMARTSIM_NUM_GPUS=1\n"
            "BENCHMARK_SOLVER_WALL_SECONDS=150\n"
            "NODE_RAM_PEAK_MIB_n1=1000\nGPU_VRAM_PEAK_MIB_n1_gpu0=2000\n")
        self.canonical_csv = self.root / "runs.csv"
        with self.canonical_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "cell", "run_dir", "cubes_per_rank", "samples_per_rank",
                "cubes_aggregate", "samples_aggregate", "solver_ranks", "gpu_count",
            ])
            writer.writeheader()
            writer.writerow({
                "cell": "smartsim-24c1g-shared-transformer-r1",
                "run_dir": str(self.root / "missing"),
                "cubes_per_rank": 1152, "samples_per_rank": 3456,
                "cubes_aggregate": 27648, "samples_aggregate": 82944,
                "solver_ranks": 24, "gpu_count": 1,
            })
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "cell", "jobid", "placement", "model", "rep", "batch_size",
                "min_batch_size", "timeout_ms", "revision", "dry_run"])
            writer.writeheader()
            writer.writerow({
                "cell": "smartsim-24c1g-agg-transformer-r1", "jobid": "123",
                "placement": "24c1g", "model": "transformer", "rep": "1",
                "batch_size": "7000", "min_batch_size": "7000",
                "timeout_ms": "1", "revision": "abc", "dry_run": "0",
            })

    def tearDown(self):
        self.tmp.cleanup()

    def run_validator(self):
        return subprocess.run(
            [sys.executable, str(VALIDATOR), "--manifest", str(self.manifest),
             "--canonical", str(self.canonical_csv), "--scratch", str(self.scratch)],
            capture_output=True, text=True,
        )

    def test_valid_fixture_passes(self):
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1/1 passed", result.stdout)
        self.assertTrue((self.root / "validation.csv").is_file())

    def test_wrong_effective_batch_fails(self):
        path = self.run_dir / "run.info"
        path.write_text(path.read_text().replace("BATCH_SIZE_EFFECTIVE=7000",
                                                 "BATCH_SIZE_EFFECTIVE=50000"))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("SMARTSIM_BATCH_SIZE_EFFECTIVE", result.stdout)

    def test_missing_telemetry_fails(self):
        path = self.run_dir / "run.info"
        path.write_text(path.read_text().replace("GPU_VRAM_PEAK_MIB_n1_gpu0=2000\n", ""))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("VRAM telemetry", result.stdout)

    def test_workload_mismatch_fails(self):
        path = self.run_dir / "solver.log"
        path.write_text(path.read_text().replace("samples_per_rank=3456",
                                                 "samples_per_rank=3457"))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("workload", result.stdout)


if __name__ == "__main__":
    unittest.main()
