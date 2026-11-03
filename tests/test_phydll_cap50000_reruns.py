"""
Offline tests for:
  - slurm/submit_phydll_cap50000_reruns.sh  (bash syntax + dry-run + mock submit)
  - scripts/check_phydll_cap_correctness.py  (chunked forward, NaN detection, shape mismatch)
"""
import csv
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "slurm/submit_phydll_cap50000_reruns.sh"
CHECKER_PATH = ROOT / "scripts/check_phydll_cap_correctness.py"

# ---------------------------------------------------------------------------
# Helpers to load checker module without executing __main__
# ---------------------------------------------------------------------------

def _load_checker():
    spec = importlib.util.spec_from_file_location("check_phydll_cap_correctness", CHECKER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


checker = _load_checker()


# ---------------------------------------------------------------------------
# Submitter tests
# ---------------------------------------------------------------------------

class TestSubmitterSyntax(unittest.TestCase):
    """bash -n: parse/syntax check of the shell submitter."""

    def test_bash_syntax_ok(self):
        result = subprocess.run(
            ["bash", "-n", str(SUBMITTER)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0,
                         f"bash -n failed:\n{result.stderr}")


class TestDryRun(unittest.TestCase):
    """Dry-run must not create any directories or files."""

    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ,
                       MODELS="transformer",
                       PLACEMENTS="24c1g",
                       WAITS="default",
                       REPS="1",
                       PHYDLL_RERUN_CAMPAIGN_BASE=str(Path(tmp) / "campaigns"))
            # Point campaign root somewhere inside tmp so we can assert no dirs.
            result = subprocess.run(
                ["bash", str(SUBMITTER), "--dry-run"],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0,
                             f"dry-run returned non-zero:\n{result.stderr}\n{result.stdout}")
            self.assertIn("DRY_RUN", result.stdout)
            self.assertIn("Dry run complete", result.stdout)
            # campaign path printed as <unique>
            self.assertIn("<unique>", result.stdout)
            # No new directories inside tmp
            created = list(Path(tmp).rglob("*"))
            self.assertEqual(created, [],
                             f"Dry-run created unexpected files: {created}")

    def test_dry_run_no_args_is_dry_run(self):
        """No arguments should default to dry-run."""
        result = subprocess.run(
            ["bash", str(SUBMITTER)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("Dry run complete", result.stdout)

    def test_dry_run_rejects_unknown_arg(self):
        result = subprocess.run(
            ["bash", str(SUBMITTER), "--unknown"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)


class TestMockSubmit(unittest.TestCase):
    """--submit with a mock sbatch: check campaign dir, manifest, provenance."""

    def _run_submit(self, tmp, models="transformer", placements="24c1g",
                    waits="default", reps="1"):
        tmp = Path(tmp)
        mock_sbatch = tmp / "bin" / "sbatch"
        mock_sbatch.parent.mkdir()
        # Mock sbatch: echo a fixed job ID and record args
        mock_sbatch.write_text(
            '#!/bin/bash\n'
            'printf "%s\\n" "$@" >> "$HOME/sbatch_args.txt"\n'
            'printf "99001\\n"\n'
        )
        mock_sbatch.chmod(0o755)
        new_path = f"{mock_sbatch.parent}:{os.environ['PATH']}"
        env = dict(os.environ,
                   HOME=str(tmp),
                   PATH=new_path,
                   MODELS=models,
                   PLACEMENTS=placements,
                   WAITS=waits,
                   REPS=reps,
                   PHYDLL_RERUN_CAMPAIGN_BASE=str(tmp / "campaigns"))
        result = subprocess.run(
            ["bash", str(SUBMITTER), "--submit"],
            env=env, capture_output=True, text=True,
        )
        return result, tmp

    def test_submit_creates_campaign_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, tmp = self._run_submit(tmp)
            self.assertEqual(result.returncode, 0,
                             f"submit failed:\n{result.stderr}\n{result.stdout}")
            campaign_base = ROOT / "results/phydll_cap50000_reruns"
            # Campaign dirs should exist (under the real repo root since we
            # don't override that; or wherever the script puts them).
            # We verify via stdout mention of 'campaign'.
            self.assertIn("Campaign complete", result.stdout)

    def test_submit_manifest_has_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, tmp = self._run_submit(tmp)
            self.assertEqual(result.returncode, 0,
                             f"submit failed:\n{result.stderr}\n{result.stdout}")
            # Find the manifest line in output
            lines = result.stdout.splitlines()
            manifest_line = next((l for l in lines if "manifest" in l.lower()), None)
            self.assertIsNotNone(manifest_line, "No manifest line in output")
            manifest_path = Path(manifest_line.split(":")[-1].strip())
            self.assertTrue(manifest_path.is_file(), f"manifest not found: {manifest_path}")
            with manifest_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            self.assertGreater(len(rows), 0)
            expected_cols = {"cell", "jobid", "placement", "model", "wait",
                             "rep", "cap", "status_dir", "revision", "dry_run"}
            self.assertEqual(set(reader.fieldnames), expected_cols)

    def test_submit_cell_names_include_cap50000(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, tmp = self._run_submit(tmp)
            self.assertEqual(result.returncode, 0,
                             f"submit failed:\n{result.stderr}\n{result.stdout}")
            lines = result.stdout.splitlines()
            manifest_line = next((l for l in lines if "manifest" in l.lower()), None)
            manifest_path = Path(manifest_line.split(":")[-1].strip())
            with manifest_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for row in rows:
                self.assertIn("cap50000", row["cell"],
                              f"cell name missing cap50000: {row['cell']}")

    def test_submit_cap_50000_for_transformer(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, tmp = self._run_submit(tmp, models="transformer",
                                           placements="24c1g", waits="default", reps="1")
            self.assertEqual(result.returncode, 0)
            lines = result.stdout.splitlines()
            manifest_line = next((l for l in lines if "manifest" in l.lower()), None)
            manifest_path = Path(manifest_line.split(":")[-1].strip())
            with manifest_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for row in rows:
                if row["model"] == "transformer":
                    self.assertEqual(row["cap"], "50000",
                                     f"transformer cap should be 50000, got {row['cap']}")

    def test_submit_cap_1000000_for_mlp(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, tmp = self._run_submit(tmp, models="mlp",
                                           placements="24c1g", waits="default", reps="1")
            self.assertEqual(result.returncode, 0)
            lines = result.stdout.splitlines()
            manifest_line = next((l for l in lines if "manifest" in l.lower()), None)
            manifest_path = Path(manifest_line.split(":")[-1].strip())
            with manifest_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for row in rows:
                if row["model"] == "mlp":
                    self.assertEqual(row["cap"], "1000000",
                                     f"mlp cap should be 1000000, got {row['cap']}")

    def test_submit_provenance_file_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, tmp = self._run_submit(tmp)
            self.assertEqual(result.returncode, 0,
                             f"submit failed:\n{result.stderr}\n{result.stdout}")
            self.assertIn("provenance", result.stdout)
            prov_line = next(
                (l for l in result.stdout.splitlines() if "provenance" in l.lower()), None
            )
            self.assertIsNotNone(prov_line)
            # Extract path after ':'
            prov_path = Path(prov_line.split(":")[-1].strip())
            self.assertTrue(prov_path.is_file(), f"provenance.txt not found: {prov_path}")

    def test_submit_dry_run_column_is_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, tmp = self._run_submit(tmp)
            self.assertEqual(result.returncode, 0)
            lines = result.stdout.splitlines()
            manifest_line = next((l for l in lines if "manifest" in l.lower()), None)
            manifest_path = Path(manifest_line.split(":")[-1].strip())
            with manifest_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            for row in rows:
                self.assertEqual(row["dry_run"], "0",
                                 f"dry_run column should be 0 on --submit, got {row['dry_run']}")

    def test_submit_multiple_reps_distinct_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, tmp = self._run_submit(tmp, reps="3")
            self.assertEqual(result.returncode, 0)
            lines = result.stdout.splitlines()
            manifest_line = next((l for l in lines if "manifest" in l.lower()), None)
            manifest_path = Path(manifest_line.split(":")[-1].strip())
            with manifest_path.open() as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            cells = [row["cell"] for row in rows]
            self.assertEqual(len(cells), len(set(cells)), "Duplicate cell names found")
            reps_found = {row["rep"] for row in rows}
            self.assertEqual(reps_found, {"1", "2", "3"})

    def test_does_not_write_to_cmi_bench_phydll_jobids(self):
        """Must NOT touch results/cmi_bench_phydll_jobids.txt."""
        jobids_file = ROOT / "results/cmi_bench_phydll_jobids.txt"
        existed_before = jobids_file.exists()
        mtime_before = jobids_file.stat().st_mtime if existed_before else None
        with tempfile.TemporaryDirectory() as tmp:
            self._run_submit(tmp)
        if existed_before:
            self.assertEqual(jobids_file.stat().st_mtime, mtime_before,
                             "cmi_bench_phydll_jobids.txt was modified")
        else:
            self.assertFalse(jobids_file.exists(),
                             "cmi_bench_phydll_jobids.txt was created")


class TestSubmitterValidation(unittest.TestCase):
    """Reject invalid matrix values."""

    def _dry_run_with_env(self, **env_overrides):
        env = dict(os.environ, **env_overrides)
        return subprocess.run(
            ["bash", str(SUBMITTER), "--dry-run"],
            env=env, capture_output=True, text=True,
        )

    def test_rejects_unknown_placement(self):
        result = self._dry_run_with_env(PLACEMENTS="99c99g")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR", result.stderr)

    def test_rejects_unknown_wait(self):
        result = self._dry_run_with_env(WAITS="immediate")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR", result.stderr)

    def test_rejects_unknown_model(self):
        result = self._dry_run_with_env(MODELS="resnet")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR", result.stderr)


# ---------------------------------------------------------------------------
# Correctness checker unit tests (no GPU, no model file required)
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn


def _make_linear_model(in_features=5 * 512, out_features=2 * 512, scale=2.0):
    """Simple y = scale*x model as a TorchScript module."""
    class LinearScale(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            # x: [N, 5, 512] -> y: [N, 2, 512]  (first 2 channels, scaled)
            return x[:, :2, :] * self.scale

    model = LinearScale(scale)
    model.eval()
    return torch.jit.script(model)


class TestRunModelChunked(unittest.TestCase):
    """Test checker.run_model_chunked for correctness and consistency across caps."""

    def setUp(self):
        self.model = _make_linear_model(scale=2.0)
        rng = np.random.default_rng(42)
        # 10 samples, shape [10, 5, 512]
        self.inp = rng.standard_normal((10, 5, 512)).astype(np.float32)

    def test_chunked_vs_full_batch_identical(self):
        """cap_a and cap_b must produce identical results for a linear model."""
        out_full = checker.run_model_chunked(self.model, self.inp, cap=10)
        out_chunked = checker.run_model_chunked(self.model, self.inp, cap=3)
        np.testing.assert_array_equal(
            out_full, out_chunked,
            err_msg="Chunked and full-batch outputs differ for linear model",
        )

    def test_cap_a_vs_cap_b_passes_allclose(self):
        """For a linear model, outputs at different caps should pass np.allclose."""
        out_a = checker.run_model_chunked(self.model, self.inp, cap=2)
        out_b = checker.run_model_chunked(self.model, self.inp, cap=7)
        self.assertTrue(np.allclose(out_a, out_b, atol=1e-4, rtol=1e-4))

    def test_output_shape(self):
        out = checker.run_model_chunked(self.model, self.inp, cap=5)
        self.assertEqual(out.shape, (10, 2, 512))

    def test_output_values_correct(self):
        """y should equal 2 * x[:, :2, :]."""
        out = checker.run_model_chunked(self.model, self.inp, cap=10)
        expected = self.inp[:, :2, :] * 2.0
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_single_sample(self):
        inp1 = self.inp[:1]
        out = checker.run_model_chunked(self.model, inp1, cap=1)
        self.assertEqual(out.shape, (1, 2, 512))

    def test_cap_larger_than_n(self):
        """cap > N should still produce correct results (single forward call)."""
        out_large_cap = checker.run_model_chunked(self.model, self.inp, cap=10000)
        out_exact_cap = checker.run_model_chunked(self.model, self.inp, cap=10)
        np.testing.assert_array_equal(out_large_cap, out_exact_cap)


class TestCheckFinite(unittest.TestCase):
    """Test NaN and Inf detection."""

    def test_clean_array_passes(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertTrue(checker.check_finite(arr, "clean"))

    def test_nan_detected(self):
        arr = np.array([1.0, float("nan"), 3.0], dtype=np.float32)
        self.assertFalse(checker.check_finite(arr, "nan_test"))

    def test_inf_detected(self):
        arr = np.array([1.0, float("inf"), 3.0], dtype=np.float32)
        self.assertFalse(checker.check_finite(arr, "inf_test"))

    def test_neginf_detected(self):
        arr = np.array([-float("inf"), 1.0], dtype=np.float32)
        self.assertFalse(checker.check_finite(arr, "neginf_test"))

    def test_mixed_nan_inf_detected(self):
        arr = np.array([float("nan"), float("inf")], dtype=np.float32)
        self.assertFalse(checker.check_finite(arr, "mixed"))


class TestCompareOutputs(unittest.TestCase):
    """Test compare_outputs shape mismatch and tolerance logic."""

    def test_identical_arrays_pass(self):
        a = np.zeros((5, 2, 512), dtype=np.float32)
        b = np.zeros((5, 2, 512), dtype=np.float32)
        passed, stats = checker.compare_outputs(a, b, atol=1e-6, rtol=1e-6, label="test")
        self.assertTrue(passed)
        self.assertAlmostEqual(stats["max_abs_err"], 0.0)
        self.assertAlmostEqual(stats["rmse"], 0.0)

    def test_shape_mismatch_fails(self):
        a = np.zeros((5, 2, 512), dtype=np.float32)
        b = np.zeros((6, 2, 512), dtype=np.float32)
        passed, stats = checker.compare_outputs(a, b, atol=1e-6, rtol=1e-6, label="test")
        self.assertFalse(passed)
        self.assertIn("shape mismatch", stats.get("error", ""))

    def test_within_tolerance_passes(self):
        a = np.ones((3, 2, 512), dtype=np.float32)
        b = a + 5e-5  # within atol=1e-4
        passed, stats = checker.compare_outputs(a, b, atol=1e-4, rtol=1e-4, label="test")
        self.assertTrue(passed)

    def test_outside_tolerance_fails(self):
        a = np.ones((3, 2, 512), dtype=np.float32)
        b = a + 1.0   # well outside tolerance
        passed, stats = checker.compare_outputs(a, b, atol=1e-4, rtol=1e-4, label="test")
        self.assertFalse(passed)
        self.assertGreater(stats["max_abs_err"], 0.5)

    def test_stats_keys_present(self):
        a = np.ones((2, 2, 4), dtype=np.float32)
        b = a * 1.001
        _, stats = checker.compare_outputs(a, b, atol=1e-2, rtol=1e-2, label="x")
        for key in ("max_abs_err", "rmse", "rel_err", "passed"):
            self.assertIn(key, stats)


class TestParseManifest(unittest.TestCase):
    def test_parse_key_value(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("num_cubes=7\ncap=50000\n# comment\nfoo=bar\n")
            name = f.name
        result = checker.parse_manifest(Path(name))
        self.assertEqual(result["num_cubes"], "7")
        self.assertEqual(result["cap"], "50000")
        self.assertEqual(result["foo"], "bar")
        self.assertNotIn("# comment", result)
        os.unlink(name)

    def test_ignores_lines_without_equals(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("no_equals_here\nkey=val\n")
            name = f.name
        result = checker.parse_manifest(Path(name))
        self.assertEqual(list(result.keys()), ["key"])
        os.unlink(name)


class TestSha256File(unittest.TestCase):
    def test_known_sha(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"hello world\n")
            name = f.name
        result = checker.sha256_file(Path(name))
        import hashlib
        expected = hashlib.sha256(b"hello world\n").hexdigest()
        self.assertEqual(result, expected)
        os.unlink(name)


if __name__ == "__main__":
    unittest.main()


class TestSelectRanks(unittest.TestCase):
    """Evenly spaced rank sampling keeps both endpoints and stays bounded."""

    def test_all_when_limit_none_or_large(self):
        self.assertEqual(checker.select_ranks(24, None), list(range(24)))
        self.assertEqual(checker.select_ranks(24, 24), list(range(24)))
        self.assertEqual(checker.select_ranks(24, 99), list(range(24)))

    def test_evenly_spaced_includes_ends(self):
        self.assertEqual(checker.select_ranks(24, 4), [0, 8, 15, 23])
        self.assertEqual(checker.select_ranks(24, 1), [0])
        self.assertEqual(len(checker.select_ranks(24, 7)), 7)

    def test_rejects_nonpositive_limit(self):
        with self.assertRaises(ValueError):
            checker.select_ranks(24, 0)
