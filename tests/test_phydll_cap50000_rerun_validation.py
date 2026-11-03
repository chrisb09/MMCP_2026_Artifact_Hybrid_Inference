import csv
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate_phydll_cap50000_reruns.py"
STEPS = (15, 44, 73, 102, 131, 160, 189)
WORKLOAD = (
    "ML_WORKLOAD_LOG cubes_per_rank=1152 samples_per_rank=3456 "
    "cubes_aggregate=27648 samples_aggregate=82944 cubeD=8 overlap=0 solver_ranks=24"
)
PROPS = "[library]\nlayout = 'per-ml-node'\n"


class TestRerunValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.canonical_dir = self.root / "canonical"
        self.run_dir = self.root / "run_123_suffix"
        self.canonical_dir.mkdir()
        self.run_dir.mkdir()
        for directory in (self.canonical_dir, self.run_dir):
            (directory / "properties.toml").write_text(PROPS)
            lines = [f"ML_STEP_TIMING step={step} ml_ms={100 + step / 10}" for step in STEPS]
            (directory / "solver.log").write_text("\n".join(lines + [WORKLOAD]) + "\n")
            (directory / "run.info").write_text(
                "CELL=phydll-24c1g-default-transformer-cap50000-r1\n"
                "REP=1\nRUN_STEPS=200\nSOLUTION_INTERVAL=10000\n"
                "ML_CUBE_OVERLAP=0\nMAIA_PHYDLL_BATCHSIZE=50000\nDEBUG_EXPORT=0\n"
                "EXIT_STATUS=0\nSOLVER_EXIT_STATUS=0\n"
                "MODEL=./input/transformer_inference_scripted_fw2.pt\n"
                "BENCHMARK_SOLVER_WALL_SECONDS=150\n"
                "NODE_RAM_PEAK_MIB_n1=1000\nGPU_VRAM_PEAK_MIB_n1_gpu0=2000\n"
                "CASE=phydll-blocking\nPHYDLL_WAIT=default\n"
            )
        self.canonical_csv = self.root / "runs.csv"
        with self.canonical_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "cell", "run_dir", "cubes_per_rank", "samples_per_rank",
                "cubes_aggregate", "samples_aggregate", "solver_ranks",
            ])
            writer.writeheader()
            writer.writerow({
                "cell": "phydll-24c1g-default-transformer-r1",
                "run_dir": str(self.canonical_dir),
                "cubes_per_rank": 1152, "samples_per_rank": 3456,
                "cubes_aggregate": 27648, "samples_aggregate": 82944,
                "solver_ranks": 24,
            })
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "cell", "jobid", "placement", "model", "wait", "rep", "cap",
                "status_dir", "revision", "dry_run",
            ])
            writer.writeheader()
            writer.writerow({
                "cell": "phydll-24c1g-default-transformer-cap50000-r1",
                "jobid": "123", "placement": "24c1g", "model": "transformer",
                "wait": "default", "rep": "1", "cap": "50000",
                "status_dir": str(self.run_dir), "revision": "abc", "dry_run": "0",
            })

    def tearDown(self):
        self.tmp.cleanup()

    def run_validator(self):
        return subprocess.run(
            [sys.executable, str(VALIDATOR), "--manifest", str(self.manifest),
             "--canonical", str(self.canonical_csv)],
            capture_output=True, text=True,
        )

    def test_valid_fixture_passes(self):
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1/1 passed", result.stdout)
        self.assertTrue((self.root / "validation.csv").is_file())

    def test_missing_manifest_directory_resolves_both_naming_patterns(self):
        for name in ("phydll-blocking_gpu_123_random", "cell_123_random"):
            with self.subTest(name=name):
                target = self.root / name
                self.run_dir.rename(target)
                result = self.run_validator()
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                target.rename(self.run_dir)

    def test_wrong_cell_and_similar_jobid_are_not_matches(self):
        self.run_dir.rename(self.root / "run_1234_suffix")
        other = self.root / "other_123_suffix"
        shutil.copytree(self.canonical_dir, other)
        path = other / "run.info"
        path.write_text(path.read_text().replace("CELL=phydll-", "CELL=wrong-"))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("found 0", result.stdout)

    def test_ambiguous_matches_fail_even_if_manifest_path_exists(self):
        shutil.copytree(self.run_dir, self.root / "another_123_suffix")
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("found 2", result.stdout)

    def test_existing_manifest_path_with_wrong_cell_is_not_trusted(self):
        shutil.copytree(self.run_dir, self.root / "correct_123_suffix")
        path = self.run_dir / "run.info"
        path.write_text(path.read_text().replace("CELL=phydll-", "CELL=wrong-"))
        result = self.run_validator()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_wrong_cap_fails(self):
        path = self.run_dir / "run.info"
        path.write_text(path.read_text().replace("BATCHSIZE=50000", "BATCHSIZE=2048"))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("MAIA_PHYDLL_BATCHSIZE", result.stdout)

    def test_missing_telemetry_fails(self):
        path = self.run_dir / "run.info"
        path.write_text(path.read_text().replace("GPU_VRAM_PEAK_MIB_n1_gpu0=2000\n", ""))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("VRAM telemetry", result.stdout)

    def test_workload_mismatch_fails(self):
        path = self.run_dir / "solver.log"
        path.write_text(path.read_text().replace("samples_per_rank=3456", "samples_per_rank=3457"))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("workload", result.stdout)

    def test_missing_step_fails(self):
        path = self.run_dir / "solver.log"
        path.write_text(path.read_text().replace("ML_STEP_TIMING step=189", "ML_STEP_TIMING step=188"))
        result = self.run_validator()
        self.assertEqual(result.returncode, 1)
        self.assertIn("inference steps", result.stdout)


if __name__ == "__main__":
    unittest.main()
