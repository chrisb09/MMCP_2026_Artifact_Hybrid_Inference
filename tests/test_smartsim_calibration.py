"""Offline tests for SmartSim aggregation infrastructure.

No test invokes sbatch against a real Slurm cluster or requires CUDA.
A mock ``sbatch`` executable is placed on PATH for submission tests.
"""
import csv
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INJECT = ROOT / "slurm/inject_smartsim_aggregation.sh"
CALIB  = ROOT / "slurm/submit_smartsim_calibration.sh"
RERUNS = ROOT / "slurm/submit_smartsim_aggregation_reruns.sh"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _make_mock_sbatch(tmp: Path, job_id: str = "99999") -> Path:
    """Write a minimal sbatch stub that prints a job-id and records its args."""
    mock = tmp / "sbatch"
    mock.write_text(
        f'#!/bin/bash\nprintf "%s\\n" "$@" >> "$HOME/sbatch_calls"\nprintf "{job_id}\\n"\n'
    )
    mock.chmod(0o755)
    return mock


def _env_with_mock_sbatch(tmp: Path, extra: dict | None = None) -> dict:
    mock_sbatch_dir = str(tmp)
    env = {**os.environ, "HOME": str(tmp), "PATH": mock_sbatch_dir + ":" + os.environ["PATH"]}
    if extra:
        env.update(extra)
    return env


def _first_manifest(campaign_root: Path) -> Path:
    manifests = list(campaign_root.rglob("manifest.csv"))
    if not manifests:
        raise FileNotFoundError(f"No manifest.csv found under {campaign_root}")
    return manifests[0]


# ─────────────────────────────────────────────────────────────────────────────
# FILE 1: inject_smartsim_aggregation.sh – syntax & TOML rewrite tests
# ─────────────────────────────────────────────────────────────────────────────

class InjectHelperSyntaxTest(unittest.TestCase):
    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(INJECT)], check=True)


class InjectHelperTomlTest(unittest.TestCase):
    """Test the TOML rewrite logic of inject_smartsim_aggregation.sh."""

    _TEMPLATE = """\
[general]
coupling_type = "STATIC"

[library]
class = "Smartsim"
batch_size = 0
min_batch_size = 0
min_batch_timeout = 0
command_timeout = 60000
"""

    def _run_inject(self, toml_text: str, env_extra: dict, *, expect_fail: bool = False) -> tuple[str, str, str]:
        """Source inject_smartsim_aggregation.sh in a temp dir and return (stdout, stderr, toml_after)."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / "config_smartsim.toml"
            config.write_text(toml_text)
            env = {**os.environ, "SMARTSIM_CONFIG": str(config), **env_extra}
            cmd = f'source "{INJECT}"'
            result = subprocess.run(
                ["bash", "-euc", cmd],
                env=env, cwd=str(tmp),
                text=True, capture_output=True,
            )
            if not expect_fail and result.returncode != 0:
                self.fail(
                    f"inject failed (rc={result.returncode}):\nstdout={result.stdout}\nstderr={result.stderr}"
                )
            if expect_fail and result.returncode == 0:
                self.fail("inject succeeded but failure was expected")
            return result.stdout, result.stderr, config.read_text() if config.exists() else ""

    def test_correct_rewrite_enabled(self):
        _, _, toml = self._run_inject(
            self._TEMPLATE,
            {"SMARTSIM_BATCH_SIZE": "7000",
             "SMARTSIM_MIN_BATCH_SIZE": "7000",
             "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "5"},
        )
        def _field(name):
            m = re.search(rf"^{re.escape(name)}\s*=\s*(\d+)", toml, re.MULTILINE)
            self.assertIsNotNone(m, f"{name} not found in TOML:\n{toml}")
            return int(m.group(1))
        self.assertEqual(_field("batch_size"), 7000)
        self.assertEqual(_field("min_batch_size"), 7000)
        self.assertEqual(_field("min_batch_timeout"), 5)

    def test_disabled_skips_rewrite(self):
        _, _, toml = self._run_inject(
            self._TEMPLATE,
            {"SMARTSIM_BATCH_SIZE": "0",
             "SMARTSIM_MIN_BATCH_SIZE": "0",
             "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "0"},
        )
        # TOML must remain unchanged (all zeros)
        self.assertIn("batch_size = 0", toml)
        self.assertIn("min_batch_size = 0", toml)
        self.assertIn("min_batch_timeout = 0", toml)

    def test_rejects_min_batch_size_greater_than_batch_size(self):
        self._run_inject(
            self._TEMPLATE,
            {"SMARTSIM_BATCH_SIZE": "1000",
             "SMARTSIM_MIN_BATCH_SIZE": "2000",
             "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "1"},
            expect_fail=True,
        )

    def test_rejects_timeout_zero_when_batching_enabled(self):
        self._run_inject(
            self._TEMPLATE,
            {"SMARTSIM_BATCH_SIZE": "7000",
             "SMARTSIM_MIN_BATCH_SIZE": "7000",
             "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "0"},
            expect_fail=True,
        )

    def test_rejects_non_integer_batch_size(self):
        self._run_inject(
            self._TEMPLATE,
            {"SMARTSIM_BATCH_SIZE": "abc",
             "SMARTSIM_MIN_BATCH_SIZE": "0",
             "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "1"},
            expect_fail=True,
        )

    def test_rejects_missing_env_var(self):
        env_no_timeout = {
            "SMARTSIM_BATCH_SIZE": "7000",
            "SMARTSIM_MIN_BATCH_SIZE": "7000",
        }
        # Don't pass SMARTSIM_MIN_BATCH_TIMEOUT_MS at all; the script must detect it
        # by using "${VAR-}" (unset → empty string).
        # We can force an unset by popping it from the environment.
        env = {**os.environ, **env_no_timeout}
        env.pop("SMARTSIM_MIN_BATCH_TIMEOUT_MS", None)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / "config_smartsim.toml"
            config.write_text(self._TEMPLATE)
            env["SMARTSIM_CONFIG"] = str(config)
            result = subprocess.run(
                ["bash", "-c", f'source "{INJECT}"'],
                env=env, cwd=str(tmp), text=True, capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_idempotent_double_source(self):
        """Sourcing twice with the same values leaves the TOML identical."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / "config_smartsim.toml"
            config.write_text(self._TEMPLATE)
            env = {**os.environ,
                   "SMARTSIM_CONFIG": str(config),
                   "SMARTSIM_BATCH_SIZE": "7000",
                   "SMARTSIM_MIN_BATCH_SIZE": "7000",
                   "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "5"}
            cmd = f'source "{INJECT}" && source "{INJECT}"'
            result = subprocess.run(["bash", "-euc", cmd], env=env, cwd=str(tmp),
                                    text=True, capture_output=True, check=True)
            toml = config.read_text()
        # Values should still be correct after double rewrite
        self.assertIn("batch_size = 7000", toml)
        self.assertIn("min_batch_size = 7000", toml)
        self.assertIn("min_batch_timeout = 5", toml)

    def test_records_to_run_info_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            config = tmp / "config_smartsim.toml"
            config.write_text(self._TEMPLATE)
            run_info = tmp / "run.info"
            run_info.write_text("")
            env = {**os.environ,
                   "SMARTSIM_CONFIG": str(config),
                   "SMARTSIM_BATCH_SIZE": "7000",
                   "SMARTSIM_MIN_BATCH_SIZE": "7000",
                   "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "5"}
            subprocess.run(["bash", "-euc", f'source "{INJECT}"'],
                           env=env, cwd=str(tmp), check=True)
            content = run_info.read_text()
        self.assertIn("SMARTSIM_AGGREGATION_ENABLED=1", content)
        self.assertIn("SMARTSIM_BATCH_SIZE_EFFECTIVE=7000", content)
        self.assertIn("SMARTSIM_MIN_BATCH_TIMEOUT_MS_EFFECTIVE=5", content)


# ─────────────────────────────────────────────────────────────────────────────
# FILE 2: submit_smartsim_calibration.sh
# ─────────────────────────────────────────────────────────────────────────────

class CalibSyntaxTest(unittest.TestCase):
    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(CALIB)], check=True)


class CalibDryRunTest(unittest.TestCase):
    def test_dry_run_produces_no_dirs_outside_campaign(self):
        """Default (no --submit) must not create scratch dirs or real campaigns."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            calib_root = tmp / "results" / "smartsim_calibration"
            scratch_root = tmp / "scratch" / "smartsim_calibration"
            env = {**os.environ,
                   "SMARTSIM_CALIB_ROOT": str(calib_root),
                   "SMARTSIM_CALIB_SCRATCH": str(scratch_root)}
            # Call without --submit; should default to dry-run
            result = subprocess.run(["bash", str(CALIB)],
                                    env=env, text=True, capture_output=True, check=True)
            # scratch_root should NOT exist (nothing actually ran)
            self.assertFalse(scratch_root.exists(),
                             f"scratch dir was created during dry-run: {scratch_root}")
            # calib_root *is* created (for the campaign dir), but no sbatch calls
            self.assertIn("dry-run", result.stdout.lower())


class CalibManifestTest(unittest.TestCase):
    """Mock sbatch: manifest has correct columns and exactly 24 cells."""

    EXPECTED_COLUMNS = {"cell", "jobid", "placement", "model",
                        "batch_size", "min_batch_size", "timeout_ms",
                        "revision", "dry_run"}
    EXPECTED_CELLS = 24  # 2 models × 4 placements × 3 variants

    def test_manifest_structure_and_24_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_mock_sbatch(tmp)
            calib_root = tmp / "results"
            scratch_root = tmp / "scratch"
            env = _env_with_mock_sbatch(tmp, {
                "SMARTSIM_CALIB_ROOT": str(calib_root),
                "SMARTSIM_CALIB_SCRATCH": str(scratch_root),
            })
            result = subprocess.run(
                ["bash", str(CALIB), "--submit"],
                env=env, text=True, capture_output=True, check=True,
            )
            manifest = _first_manifest(calib_root)
            rows = _read_manifest(manifest)

        self.assertEqual(len(rows), self.EXPECTED_CELLS,
                         f"Expected {self.EXPECTED_CELLS} rows, got {len(rows)}.\n"
                         f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertEqual(set(rows[0].keys()), self.EXPECTED_COLUMNS)
        # All cells must be unique
        cells = [r["cell"] for r in rows]
        self.assertEqual(len(cells), len(set(cells)), "Duplicate cell names in manifest")

    def test_cell_naming_convention(self):
        """ctrl cells have timeout_ms=0; candidate cells encode timeout in name."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_mock_sbatch(tmp)
            calib_root = tmp / "results"
            env = _env_with_mock_sbatch(tmp, {
                "SMARTSIM_CALIB_ROOT": str(calib_root),
                "SMARTSIM_CALIB_SCRATCH": str(tmp / "scratch"),
            })
            subprocess.run(["bash", str(CALIB), "--submit"],
                           env=env, capture_output=True, check=True)
            rows = _read_manifest(_first_manifest(calib_root))

        ctrl_rows = [r for r in rows if r["cell"].endswith("-ctrl")]
        cand_rows = [r for r in rows if not r["cell"].endswith("-ctrl")]
        # 2 models × 4 placements = 8 control cells
        self.assertEqual(len(ctrl_rows), 8)
        # 16 candidate cells (2 timeouts × 8)
        self.assertEqual(len(cand_rows), 16)
        for row in ctrl_rows:
            self.assertEqual(row["batch_size"], "0")
            self.assertEqual(row["min_batch_size"], "0")
            self.assertEqual(row["timeout_ms"], "0")
        for row in cand_rows:
            self.assertGreater(int(row["batch_size"]), 0)
            self.assertGreater(int(row["timeout_ms"]), 0)
            # Cell name must contain t{N}ms
            self.assertRegex(row["cell"], r"-t\d+ms$")

    def test_model_batch_size_mapping(self):
        """transformer cells use batch_size=7000, mlp cells use batch_size=45000."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_mock_sbatch(tmp)
            calib_root = tmp / "results"
            env = _env_with_mock_sbatch(tmp, {
                "SMARTSIM_CALIB_ROOT": str(calib_root),
                "SMARTSIM_CALIB_SCRATCH": str(tmp / "scratch"),
            })
            subprocess.run(["bash", str(CALIB), "--submit"],
                           env=env, capture_output=True, check=True)
            rows = _read_manifest(_first_manifest(calib_root))

        for row in rows:
            if row["batch_size"] == "0":
                continue   # control cell
            if row["model"] == "transformer":
                self.assertEqual(row["batch_size"], "7000")
                self.assertEqual(row["min_batch_size"], "7000")
            elif row["model"] == "mlp":
                self.assertEqual(row["batch_size"], "45000")
                self.assertEqual(row["min_batch_size"], "45000")

    def test_all_placements_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_mock_sbatch(tmp)
            calib_root = tmp / "results"
            env = _env_with_mock_sbatch(tmp, {
                "SMARTSIM_CALIB_ROOT": str(calib_root),
                "SMARTSIM_CALIB_SCRATCH": str(tmp / "scratch"),
            })
            subprocess.run(["bash", str(CALIB), "--submit"],
                           env=env, capture_output=True, check=True)
            rows = _read_manifest(_first_manifest(calib_root))

        placements = {r["placement"] for r in rows}
        self.assertEqual(placements, {"24c1g", "24p24c1g", "96c4g", "96p96c4g"})


# ─────────────────────────────────────────────────────────────────────────────
# FILE 3: submit_smartsim_aggregation_reruns.sh
# ─────────────────────────────────────────────────────────────────────────────

class RerunsSyntaxTest(unittest.TestCase):
    def test_bash_syntax(self):
        subprocess.run(["bash", "-n", str(RERUNS)], check=True)


class RerunsValidationTest(unittest.TestCase):
    """Production submitter must error when required env vars are missing or invalid."""

    def _run_reruns(self, extra_env: dict | None = None, args: list[str] | None = None) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items()}
        # Scrub any accidental aggregation vars from the test environment
        for key in ("SMARTSIM_BATCH_SIZE", "SMARTSIM_MIN_BATCH_SIZE", "SMARTSIM_MIN_BATCH_TIMEOUT_MS"):
            env.pop(key, None)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(RERUNS)] + (args or []),
            env=env, text=True, capture_output=True,
        )

    def test_errors_without_batch_size(self):
        result = self._run_reruns({
            "SMARTSIM_MIN_BATCH_SIZE": "7000",
            "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "1",
        })
        self.assertNotEqual(result.returncode, 0)

    def test_errors_without_min_batch_size(self):
        result = self._run_reruns({
            "SMARTSIM_BATCH_SIZE": "7000",
            "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "1",
        })
        self.assertNotEqual(result.returncode, 0)

    def test_errors_without_timeout(self):
        result = self._run_reruns({
            "SMARTSIM_BATCH_SIZE": "7000",
            "SMARTSIM_MIN_BATCH_SIZE": "7000",
        })
        self.assertNotEqual(result.returncode, 0)

    def test_errors_when_batch_size_zero(self):
        result = self._run_reruns({
            "SMARTSIM_BATCH_SIZE": "0",
            "SMARTSIM_MIN_BATCH_SIZE": "0",
            "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "1",
        })
        self.assertNotEqual(result.returncode, 0)

    def test_errors_when_min_batch_size_exceeds_batch_size(self):
        result = self._run_reruns({
            "SMARTSIM_BATCH_SIZE": "1000",
            "SMARTSIM_MIN_BATCH_SIZE": "9999",
            "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "1",
        })
        self.assertNotEqual(result.returncode, 0)

    def test_errors_when_timeout_is_zero(self):
        result = self._run_reruns({
            "SMARTSIM_BATCH_SIZE": "7000",
            "SMARTSIM_MIN_BATCH_SIZE": "7000",
            "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "0",
        })
        self.assertNotEqual(result.returncode, 0)

    def test_errors_when_non_integer_value(self):
        result = self._run_reruns({
            "SMARTSIM_BATCH_SIZE": "notanumber",
            "SMARTSIM_MIN_BATCH_SIZE": "7000",
            "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "1",
        })
        self.assertNotEqual(result.returncode, 0)


class RerunsManifestTest(unittest.TestCase):
    """Mock sbatch: production reruns manifest has 24 rows with correct columns."""

    EXPECTED_COLUMNS = {"cell", "jobid", "placement", "model", "rep",
                        "batch_size", "min_batch_size", "timeout_ms",
                        "revision", "dry_run"}
    EXPECTED_CELLS = 24  # 2 models × 4 placements × 3 reps

    _BASE_AGG_ENV = {
        "SMARTSIM_BATCH_SIZE": "7000",
        "SMARTSIM_MIN_BATCH_SIZE": "7000",
        "SMARTSIM_MIN_BATCH_TIMEOUT_MS": "1",
    }

    def test_manifest_24_rows_correct_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_mock_sbatch(tmp)
            campaign_root = tmp / "results"
            scratch_root  = tmp / "scratch"
            env = _env_with_mock_sbatch(tmp, {
                **self._BASE_AGG_ENV,
                "SMARTSIM_AGG_RERUNS_ROOT": str(campaign_root),
                "SMARTSIM_AGG_SCRATCH":     str(scratch_root),
            })
            result = subprocess.run(
                ["bash", str(RERUNS), "--submit"],
                env=env, text=True, capture_output=True, check=True,
            )
            manifest = _first_manifest(campaign_root)
            rows = _read_manifest(manifest)

        self.assertEqual(len(rows), self.EXPECTED_CELLS,
                         f"Expected {self.EXPECTED_CELLS} rows, got {len(rows)}.\n"
                         f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertEqual(set(rows[0].keys()), self.EXPECTED_COLUMNS)

    def test_correct_column_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_mock_sbatch(tmp)
            campaign_root = tmp / "results"
            env = _env_with_mock_sbatch(tmp, {
                **self._BASE_AGG_ENV,
                "SMARTSIM_AGG_RERUNS_ROOT": str(campaign_root),
                "SMARTSIM_AGG_SCRATCH":     str(tmp / "scratch"),
            })
            subprocess.run(["bash", str(RERUNS), "--submit"],
                           env=env, capture_output=True, check=True)
            rows = _read_manifest(_first_manifest(campaign_root))

        for row in rows:
            self.assertEqual(row["batch_size"], "7000")
            self.assertEqual(row["min_batch_size"], "7000")
            self.assertEqual(row["timeout_ms"], "1")
            self.assertEqual(row["dry_run"], "0")
            self.assertIn(row["placement"], {"24c1g", "24p24c1g", "96c4g", "96p96c4g"})
            self.assertIn(row["model"], {"transformer", "mlp"})
            self.assertIn(row["rep"], {"1", "2", "3"})
            # Cell naming: smartsim-{placement}-agg-{model}-r{rep}
            expected_cell = f"smartsim-{row['placement']}-agg-{row['model']}-r{row['rep']}"
            self.assertEqual(row["cell"], expected_cell)

    def test_cell_names_are_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_mock_sbatch(tmp)
            campaign_root = tmp / "results"
            env = _env_with_mock_sbatch(tmp, {
                **self._BASE_AGG_ENV,
                "SMARTSIM_AGG_RERUNS_ROOT": str(campaign_root),
                "SMARTSIM_AGG_SCRATCH":     str(tmp / "scratch"),
            })
            subprocess.run(["bash", str(RERUNS), "--submit"],
                           env=env, capture_output=True, check=True)
            rows = _read_manifest(_first_manifest(campaign_root))
        cells = [r["cell"] for r in rows]
        self.assertEqual(len(cells), len(set(cells)), "Duplicate cell names")

    def test_all_placements_and_models_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _make_mock_sbatch(tmp)
            campaign_root = tmp / "results"
            env = _env_with_mock_sbatch(tmp, {
                **self._BASE_AGG_ENV,
                "SMARTSIM_AGG_RERUNS_ROOT": str(campaign_root),
                "SMARTSIM_AGG_SCRATCH":     str(tmp / "scratch"),
            })
            subprocess.run(["bash", str(RERUNS), "--submit"],
                           env=env, capture_output=True, check=True)
            rows = _read_manifest(_first_manifest(campaign_root))
        placements = {r["placement"] for r in rows}
        models = {r["model"] for r in rows}
        reps = {r["rep"] for r in rows}
        self.assertEqual(placements, {"24c1g", "24p24c1g", "96c4g", "96p96c4g"})
        self.assertEqual(models, {"transformer", "mlp"})
        self.assertEqual(reps, {"1", "2", "3"})

    def test_dry_run_by_default(self):
        """Without --submit the script defaults to dry-run and prints manifest path."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            campaign_root = tmp / "results"
            env = {**os.environ,
                   **self._BASE_AGG_ENV,
                   "SMARTSIM_AGG_RERUNS_ROOT": str(campaign_root),
                   "SMARTSIM_AGG_SCRATCH": str(tmp / "scratch")}
            result = subprocess.run(["bash", str(RERUNS)],
                                    env=env, text=True, capture_output=True, check=True)
            self.assertIn("dry-run", result.stdout.lower())
            # manifest is created even in dry-run
            manifest = _first_manifest(campaign_root)
            rows = _read_manifest(manifest)
            self.assertEqual(len(rows), self.EXPECTED_CELLS)
            self.assertTrue(all(r["dry_run"] == "1" for r in rows))


if __name__ == "__main__":
    unittest.main()


class TestRunnerWiring(unittest.TestCase):
    """Both benchmark runners must actually apply the aggregation env vars."""

    def test_both_runners_source_injection_helper(self):
        for name in ("run_cmi_smoke_gpu.sh", "cmi_bench_het_engine.sh"):
            text = (ROOT / "slurm" / name).read_text()
            self.assertIn("inject_smartsim_aggregation.sh", text, name)
            self.assertIn("SMARTSIM_BATCH_SIZE", text, name)
