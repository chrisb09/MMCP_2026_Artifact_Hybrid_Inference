import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts/analyze_smartsim_calibration.py"
STEPS = (15, 44, 73, 102, 131, 160, 189)
COUNT = 3 * 2 * 2 * 512


class TestCalibrationAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()
        self.cells = []
        for variant, timeout in (("ctrl", "0"), ("t1ms", "1"), ("t5ms", "5")):
            cell = f"smartsim-24c1g-calib-transformer-{variant}"
            run = self.scratch / f"{cell}_123"
            (run / "debug_dumps").mkdir(parents=True)
            for inference in (1, 2):
                values = np.arange(COUNT, dtype=np.float64) + inference
                values.tofile(run / "debug_dumps" /
                              f"current_rank_0_inference_{inference}_denormalized_output.bin")
                (run / "debug_dumps" /
                 f"current_rank_0_inference_{inference}_manifest.txt").write_text(
                    "implementation=current\nrank=0\ncube_dimension=8\n"
                    "cube_overlap=0\ninput_sequence_length=5\nforecast_window=2\n"
                    "n_ghost_layers=2\nnum_cubes=2\ncube_size=512\nn_fields=3\n"
                    "assembled_input_dtype=float32\nassembled_input_count=15360\n"
                    f"raw_provider_output_dtype=float32\nraw_provider_output_count={COUNT}\n"
                    "reconstructed_field_dtype=float64\nreconstructed_field_count=1024\n")
            timing = {"ctrl": 100, "t1ms": 90, "t5ms": 95}[variant]
            timings = "\n".join(
                f"ML_STEP_TIMING step={step} ml_ms={timing}" for step in STEPS)
            (run / "solver.log").write_text(
                timings + "\nML_WORKLOAD_LOG cubes_per_rank=1 samples_per_rank=1 "
                "cubes_aggregate=1 samples_aggregate=1 cubeD=8 overlap=0 solver_ranks=24\n")
            (run / "run.info").write_text(
                f"CELL={cell}\n"
                "EXIT_STATUS=0\nSOLVER_EXIT_STATUS=0\nML_CUBE_OVERLAP=0\n"
                "MODEL=./input/transformer_inference_scripted_fw2.pt\n"
                f"SMARTSIM_AGGREGATION_ENABLED={0 if variant == 'ctrl' else 1}\n"
                f"SMARTSIM_BATCH_SIZE_EFFECTIVE={0 if variant == 'ctrl' else 7000}\n"
                f"SMARTSIM_MIN_BATCH_SIZE_EFFECTIVE={0 if variant == 'ctrl' else 7000}\n"
                f"SMARTSIM_MIN_BATCH_TIMEOUT_MS_EFFECTIVE={timeout}\n")
            config = "[library]\n"
            if variant != "ctrl":
                config += f"batch_size = 7000\nmin_batch_size = 7000\nmin_batch_timeout = {timeout}\n"
            (run / "config_smartsim.toml").write_text(config)
            self.cells.append((cell, variant, timeout))
        self.manifest = self.root / "manifest.csv"
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "cell", "jobid", "placement", "model", "batch_size",
                "min_batch_size", "timeout_ms", "revision", "dry_run"])
            writer.writeheader()
            for cell, variant, timeout in self.cells:
                writer.writerow({
                    "cell": cell, "jobid": "123", "placement": "24c1g",
                    "model": "transformer",
                    "batch_size": "0" if variant == "ctrl" else "7000",
                    "min_batch_size": "0" if variant == "ctrl" else "7000",
                    "timeout_ms": timeout, "revision": "abc", "dry_run": "0",
                })

    def tearDown(self):
        self.tmp.cleanup()

    def run_analyzer(self):
        return subprocess.run(
            [sys.executable, str(ANALYZER), "--manifest", str(self.manifest),
             "--scratch", str(self.scratch)],
            capture_output=True, text=True,
        )

    def test_control_and_candidate_pass(self):
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[pick] 24c1g/transformer: t1ms", result.stdout)
        with (self.root / "analysis.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)

    def test_output_mismatch_fails(self):
        path = (self.scratch / "smartsim-24c1g-calib-transformer-t1ms_123" /
                "debug_dumps" / "current_rank_0_inference_1_denormalized_output.bin")
        (np.arange(COUNT, dtype=np.float64) + 100).tofile(path)
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 1)
        self.assertIn("output differs", result.stdout + result.stderr)

    def test_missing_candidate_fails(self):
        self.manifest.write_text(
            self.manifest.read_text().replace("smartsim-24c1g-calib-transformer-t1ms", "missing"))
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 1)

    def test_float32_coupling_output(self):
        for cell, _, _ in self.cells:
            root = self.scratch / f"{cell}_123" / "debug_dumps"
            for inference in (1, 2):
                prefix = root / f"current_rank_0_inference_{inference}"
                manifest = Path(f"{prefix}_manifest.txt")
                manifest.write_text(manifest.read_text().replace(
                    "reconstructed_field_dtype=float64", "reconstructed_field_dtype=float32"))
                (np.arange(COUNT, dtype=np.float32) + inference).tofile(
                    f"{prefix}_denormalized_output.bin")
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_trailing_byte_fails(self):
        path = (self.scratch / "smartsim-24c1g-calib-transformer-t1ms_123" /
                "debug_dumps" / "current_rank_0_inference_1_denormalized_output.bin")
        with path.open("ab") as handle:
            handle.write(b"\x00")
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 1)
        self.assertIn("invalid size", result.stderr)

    def test_manifest_shape_mismatch_fails(self):
        path = (self.scratch / "smartsim-24c1g-calib-transformer-t1ms_123" /
                "debug_dumps" / "current_rank_0_inference_1_manifest.txt")
        path.write_text(path.read_text().replace("num_cubes=2", "num_cubes=3"))
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 1)
        self.assertIn("dimensions or count", result.stderr)

    def test_same_count_different_shape_fails(self):
        path = (self.scratch / "smartsim-24c1g-calib-transformer-t1ms_123" /
                "debug_dumps" / "current_rank_0_inference_1_manifest.txt")
        path.write_text(path.read_text().replace("num_cubes=2", "num_cubes=1")
                        .replace("forecast_window=2", "forecast_window=4"))
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 1)
        with (self.root / "analysis.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[1]["reason"], "output shape mismatch")

    def test_nonfinite_control_fails(self):
        path = (self.scratch / "smartsim-24c1g-calib-transformer-ctrl_123" /
                "debug_dumps" / "current_rank_0_inference_1_denormalized_output.bin")
        values = np.arange(COUNT, dtype=np.float64)
        values[0] = np.nan
        values.tofile(path)
        result = self.run_analyzer()
        self.assertEqual(result.returncode, 1)
        self.assertIn("non-finite values", result.stderr)


if __name__ == "__main__":
    unittest.main()
