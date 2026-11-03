import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "slurm/submit_phydll_cap50000_pilot.sh"
spec = importlib.util.spec_from_file_location("validator", ROOT / "scripts/validate_phydll_cap50000_pilot.py")
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class PilotTests(unittest.TestCase):
    def test_syntax_and_dry_run_are_offline(self):
        subprocess.run(["bash", "-n", str(SUBMIT)], check=True)
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PHYDLL_PILOT_ROOT=str(Path(tmp) / "absent"))
            result = subprocess.run(["bash", str(SUBMIT)], env=env, text=True, capture_output=True, check=True)
            self.assertIn("Dry run", result.stdout)
            self.assertIn("CASE_TIMEOUT_SECONDS=600", result.stdout)
            self.assertIn("phydll-blocking", result.stdout)
            self.assertFalse((Path(tmp) / "absent").exists())

    def test_submit_calls_only_mock_once_and_isolates_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mock = tmp / "sbatch"
            mock.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$HOME/args"\nprintf "%s\\n" "${CMI_NUMA_PIN-unset}" > "$HOME/inherited"\nprintf "12345\\n"\n')
            mock.chmod(0o755)
            env = dict(os.environ, HOME=str(tmp), PATH=f"{tmp}:{os.environ['PATH']}",
                       PHYDLL_PILOT_ROOT=str(tmp / "campaigns"), CMI_NUMA_PIN="1", RUN_STEPS="999")
            for _ in range(2):
                subprocess.run(["bash", str(SUBMIT), "--submit"], env=env, text=True, capture_output=True, check=True)
            campaigns = list((tmp / "campaigns").iterdir())
            self.assertEqual(len(campaigns), 2)
            for campaign in campaigns:
                self.assertIn("sbatch_response=12345", (campaign / "manifest.txt").read_text())
                self.assertTrue((campaign / "scratch").is_dir())
            args = (tmp / "args").read_text().splitlines()
            self.assertIn("--exclusive", args)
            self.assertIn("--ntasks=24", args)
            self.assertIn("--gres=gpu:1", args)
            self.assertIn("--time=00:20:00", args)
            self.assertEqual(args[-1], "phydll-blocking")
            exports = next(x for x in args if x.startswith("--export="))
            for value in ("MAIA_PHYDLL_BATCHSIZE=50000", "RUN_STEPS=200", "DEBUG_EXPORT=1", "CASE_TIMEOUT_SECONDS=600"):
                self.assertIn(value, exports)
            self.assertEqual((tmp / "inherited").read_text().strip(), "unset")

    def test_validator_requires_matching_workload_and_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot, canonical = Path(tmp) / "pilot", Path(tmp) / "canonical"
            record = {"CASE": "phydll-blocking", "RUN_STEPS": "200", "SOLVER_RANKS": "24", "DL_RANKS": "1", "MAIA_PHYDLL_BATCHSIZE": "50000", "DEBUG_EXPORT": "1", "ML_CUBE_OVERLAP": "0", "SOLUTION_INTERVAL": "10000", "EXIT_STATUS": "0", "SOLVER_EXIT_STATUS": "0", "MODEL": "./input/transformer_inference_scripted_fw2.pt", "GPU_VRAM_PEAK_MIB_host_gpu0": "100", "NODE_RAM_PEAK_MIB_host": "100"}
            work = "ML_WORKLOAD_LOG cubes_per_rank=1 samples_per_rank=3 cubes_aggregate=24 samples_aggregate=72 cubeD=8 overlap=0 solver_ranks=24\n"
            for path in (pilot, canonical):
                path.mkdir()
                (path / "run.info").write_text("".join(f"{k}={v}\n" for k, v in record.items()))
                (path / "properties.toml").write_text("timeSteps = 200\n")
                (path / "solver.log").write_text(work + "".join(f"ML_STEP_TIMING step={x} ml_ms=1\n" for x in (15, 44, 73, 102, 131, 160, 189)))
            (pilot / "debug_dumps").mkdir()
            (pilot / "debug_dumps/export.bin").write_bytes(b"x")
            validator.validate(pilot, canonical)
            (canonical / "solver.log").write_text(work.replace("samples_aggregate=72", "samples_aggregate=73"))
            with self.assertRaisesRegex(ValueError, "workload"):
                validator.validate(pilot, canonical)
            (canonical / "solver.log").write_text(work)
            (pilot / "run.info").write_text((pilot / "run.info").read_text().replace("MAIA_PHYDLL_BATCHSIZE=50000", "MAIA_PHYDLL_BATCHSIZE=2048"))
            with self.assertRaisesRegex(ValueError, "MAIA_PHYDLL_BATCHSIZE"):
                validator.validate(pilot, canonical)


if __name__ == "__main__":
    unittest.main()
