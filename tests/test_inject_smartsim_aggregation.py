import os
import subprocess
import tempfile
import unittest
from pathlib import Path


INJECT = Path(__file__).resolve().parents[1] / "slurm/inject_smartsim_aggregation.sh"
CONFIG = """[other]
batch_size = 123
[library] # target
batch_size = 8 # previous
min_batch_size = 4
min_batch_timeout = 5
[behavior]
batch_size = 456
"""


class InjectAggregationTest(unittest.TestCase):
    def run_inject(self, config, batch="10", minimum="8", timeout="1", command=None):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "config_smartsim.toml"
            if config is not None:
                path.write_text(config)
            info = directory / "run.info"
            info.write_text("original\n")
            env = {**os.environ, "SMARTSIM_CONFIG": str(path),
                   "SMARTSIM_BATCH_SIZE": batch, "SMARTSIM_MIN_BATCH_SIZE": minimum,
                   "SMARTSIM_MIN_BATCH_TIMEOUT_MS": timeout}
            result = subprocess.run(["bash", "-eu", "-c", command or f'source "{INJECT}"'],
                                    cwd=directory, env=env, text=True, capture_output=True)
            return result, path.read_text() if path.exists() else None, info.read_text()

    def test_rewrite_only_library_and_report_actual_values(self):
        result, config, info = self.run_inject(CONFIG, "010", "008", "01")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[other]\nbatch_size = 123", config)
        self.assertIn("[behavior]\nbatch_size = 456", config)
        self.assertIn("batch_size = 10\nmin_batch_size = 8\nmin_batch_timeout = 1", config)
        self.assertIn("SMARTSIM_BATCH_SIZE_EFFECTIVE=10\n", info)
        self.assertIn("SMARTSIM_MIN_BATCH_SIZE_EFFECTIVE=8\n", result.stdout)

    def test_zero_actually_disables_existing_settings(self):
        result, config, info = self.run_inject(CONFIG, "0", "4", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("batch_size = 0\nmin_batch_size = 0\nmin_batch_timeout = 0", config)
        self.assertIn("SMARTSIM_AGGREGATION_ENABLED=0\n", info)
        self.assertIn("SMARTSIM_MIN_BATCH_TIMEOUT_MS_EFFECTIVE=0\n", info)

    def test_missing_duplicate_or_malformed_fields_fail_without_reporting(self):
        for config in (CONFIG.replace("min_batch_size = 4\n", ""),
                       CONFIG.replace("min_batch_size = 4", "min_batch_size = 4\nmin_batch_size = 3"),
                       CONFIG.replace("min_batch_timeout = 5", 'min_batch_timeout = "5"'),
                       CONFIG.replace("[library]", "[other_library]"), None):
            with self.subTest(config=config):
                result, after, info = self.run_inject(config)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(after, config)
                self.assertEqual(info, "original\n")
                self.assertNotIn("EFFECTIVE", result.stdout)

    def test_invalid_values_fail_without_changing_config(self):
        for batch, minimum, timeout in (("10", "0", "1"), ("10", "11", "1"),
                                        ("10", "1", "0"), ("10", "1", "abc"),
                                        ("2147483648", "1", "1"), ("9" * 100, "1", "1")):
            with self.subTest(batch=batch, minimum=minimum, timeout=timeout):
                result, config, info = self.run_inject(CONFIG, batch, minimum, timeout)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(config, CONFIG)
                self.assertEqual(info, "original\n")

    def test_enable_then_disable(self):
        command = (f'source "{INJECT}"; SMARTSIM_BATCH_SIZE=0; '
                   f'source "{INJECT}"')
        result, config, info = self.run_inject(CONFIG, command=command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("batch_size = 0\nmin_batch_size = 0\nmin_batch_timeout = 0", config)
        self.assertIn("SMARTSIM_AGGREGATION_ENABLED=1\n", info)
        self.assertIn("SMARTSIM_AGGREGATION_ENABLED=0\n", info)


if __name__ == "__main__":
    unittest.main()
