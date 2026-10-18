"""Allocation-only tests: python3 -m unittest discover -s scripts -p test_smartsim_controller_gpu.py."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


CONTROLLER = Path(__file__).resolve().parents[1] / "CPP-ML-Interface/dl_clients/smartsim_controller.py"


def load_controller():
    modules = {name: ModuleType(name) for name in (
        "smartsim", "smartsim.experiment", "smartsim.database",
        "smartsim.database.orchestrator",
    )}
    modules["smartsim.experiment"].Experiment = object
    modules["smartsim.database.orchestrator"].Orchestrator = type("Orchestrator", (), {})
    spec = importlib.util.spec_from_file_location("controller_gpu_test", CONTROLLER)
    controller = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(controller)
    return controller


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = load_controller()

    def run_controller(self, env, extra=(), address_plan=None, timeout=False):
        controller = self.controller
        with tempfile.TemporaryDirectory() as directory:
            endpoint = Path(directory) / "endpoint"
            done = Path(directory) / "done"
            dbs = []
            polls = []
            stops = []
            clock = [0.0]

            class Database:
                def __init__(self, kwargs):
                    self.kwargs = kwargs
                    self.run_args = {}
                    self.cpus = None
                    self.poll_count = 0
                    self.entities = [SimpleNamespace(
                        ports=[kwargs["port"]],
                        run_settings=SimpleNamespace(
                            env_vars={"KEEP": "value", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                            exe=["/fake/redis-server"],
                            exe_args=[f"+port={kwargs['port']}", "+ifname=lo"],
                        ),
                    ) for _ in range(kwargs["db_nodes"])]

                def set_run_arg(self, key, value):
                    self.run_args[key] = value

                def set_cpus(self, value):
                    self.cpus = value

                def set_db_conf(self, *args):
                    pass

                def get_address(self):
                    self.poll_count += 1
                    polls.append(endpoint.exists())
                    if address_plan:
                        return address_plan(dbs.index(self), self.poll_count)
                    return [f"host:{self.kwargs['port'] + i}"
                            for i in range(self.kwargs["db_nodes"])]

            class Experiment:
                def __init__(self, **kwargs):
                    pass

                def create_database(self, **kwargs):
                    db = Database(kwargs)
                    dbs.append(db)
                    return db

                def start(self, *databases, **kwargs):
                    pass

                def stop(self, *databases):
                    stops.extend(databases)

            def sleep(seconds):
                clock[0] += seconds
                if endpoint.exists():
                    done.touch()

            argv = [str(CONTROLLER), "--launcher", "local", "--db-layout", "per-ml-node",
                    "--db-nodes", "4", "--intra-op-threads", "2", "--silent",
                    "--endpoint-file", str(endpoint), "--done-file", str(done),
                    "--exp-dir", directory, "--timeout-s", "2", *extra]
            output = io.StringIO()
            with patch.dict(os.environ, env, clear=True), \
                    patch.object(sys, "argv", argv), \
                    patch.object(controller, "Experiment", Experiment), \
                    patch.object(controller.time, "time", side_effect=lambda: clock[0]), \
                    patch.object(controller.time, "sleep", side_effect=sleep), \
                    contextlib.redirect_stdout(output):
                if timeout:
                    with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                        controller.main()
                    self.assertFalse(endpoint.exists())
                else:
                    self.assertEqual(controller.main(), 0)
                    self.assertFalse(done.exists())
            return SimpleNamespace(dbs=dbs, stops=stops, polls=polls,
                                   endpoint=endpoint.read_text() if endpoint.exists() else None,
                                   output=output.getvalue())

    def test_explicit_noncontiguous_and_uuid_mapping(self):
        for ids in ("7,2,9,4", "GPU-a,GPU-b,MIG-GPU-c/1/0,MIG-d"):
            with self.subTest(ids=ids):
                env = {"SMARTSIM_DB_GPU_IDS": ids, "SMARTSIM_DB_GPU_COUNT": "4",
                       "CUDA_VISIBLE_DEVICES": "wrong-cpu-side", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
                result = self.run_controller(env, ["--use-gpu", "--launcher", "slurm",
                                                   "--het-group", "1", "--db-step-mem", "0"])
                records = [json.loads(line.split(" ", 1)[1])
                           for line in result.output.splitlines()
                           if line.startswith("SMARTSIM_DB_GPU_MAP ")]
                self.assertEqual(len(records), 4)
                for i, (db, mask) in enumerate(zip(result.dbs, ids.split(","))):
                    self.assertEqual(db.kwargs["db_nodes"], 1)
                    self.assertEqual(db.kwargs["db_identifier"], f"shard_{i}")
                    self.assertEqual(db.kwargs["port"], 6780 + i)
                    settings = db.entities[0].run_settings
                    self.assertEqual(settings.env_vars, {"KEEP": "value", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                                                         "CUDA_VISIBLE_DEVICES": mask})
                    self.assertEqual(settings.exe[1:], [f"CUDA_VISIBLE_DEVICES={mask}", "/fake/redis-server"])
                    self.assertEqual(db.run_args["mem"], "0")
                    self.assertIsNone(db.cpus)
                    self.assertEqual(records[i]["shard"], i)
                    self.assertEqual(records[i]["port"], 6780 + i)
                    self.assertEqual(records[i]["mask"], mask)
                    self.assertEqual(records[i]["logical_device"], 0)
                self.assertEqual(result.stops, result.dbs)

    def test_invalid_allocations_rejected_before_experiment(self):
        invalid = [
            {}, {"SMARTSIM_DB_GPU_COUNT": "4"},
            {"SMARTSIM_DB_GPU_IDS": ""}, {"SMARTSIM_DB_GPU_IDS": "0,1,2"},
            {"SMARTSIM_DB_GPU_IDS": "GPU-a,GPU-a,GPU-b,GPU-c"},
            {"SMARTSIM_DB_GPU_IDS": "0,1,,3"}, {"SMARTSIM_DB_GPU_IDS": "0,1,2,-1"},
            {"SMARTSIM_DB_GPU_IDS": "0,1,2,3", "SMARTSIM_DB_GPU_COUNT": "5"},
            {"SMARTSIM_DB_GPU_IDS": "0,1,2,3", "SMARTSIM_DB_GPU_COUNT": "bad"},
        ]
        for env in invalid:
            with self.subTest(env=env), self.assertRaises(ValueError):
                self.run_controller(env, ["--use-gpu"])

    def test_inherited_visibility_only_local_or_proven_same_node(self):
        controller = self.controller
        env = {"CUDA_VISIBLE_DEVICES": "6,3,8,1"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(controller.database_gpu_ids(4, "local", None), ["6", "3", "8", "1"])
            with self.assertRaises(ValueError):
                controller.database_gpu_ids(4, "local", "1")
            with self.assertRaises(ValueError):
                controller.database_gpu_ids(4, "slurm", None)
            os.environ.update(SLURM_JOB_NUM_NODES="1", SLURMD_NODENAME="gpu-node")
            with patch.object(controller.socket, "gethostname", return_value="gpu-node.example"):
                self.assertEqual(controller.database_gpu_ids(4, "slurm", None), ["6", "3", "8", "1"])
                os.environ["SLURM_HET_SIZE"] = "2"
                with self.assertRaises(ValueError):
                    controller.database_gpu_ids(4, "slurm", None)

    def test_local_mask_does_not_wrap_executable(self):
        result = self.run_controller({"CUDA_VISIBLE_DEVICES": "7,2,9,4"}, ["--use-gpu"])
        for db, mask in zip(result.dbs, ["7", "2", "9", "4"]):
            settings = db.entities[0].run_settings
            self.assertEqual(settings.env_vars["CUDA_VISIBLE_DEVICES"], mask)
            self.assertEqual(settings.exe, ["/fake/redis-server"])

    def test_slurm_prefix_applies_mask_after_inherited_environment(self):
        result = self.run_controller({"SMARTSIM_DB_GPU_IDS": "7,2,9,4"},
                                     ["--use-gpu", "--launcher", "slurm"])
        settings = result.dbs[0].entities[0].run_settings
        # Execute only env + a harmless Python stand-in, never a DB or Slurm job.
        command = settings.exe[:-1] + [sys.executable, "-c",
                    "import os; print(os.environ['CUDA_VISIBLE_DEVICES']); "
                    "print(os.environ['CUDA_DEVICE_ORDER'])"]
        process = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                                              "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
                                 check=True, capture_output=True, text=True)
        self.assertEqual(process.stdout.splitlines(), ["7", "PCI_BUS_ID"])

    def test_all_standalone_addresses_required(self):
        def delayed(index, attempt):
            return [] if index == 3 and attempt == 1 else [f"host:{6780 + index}"]
        result = self.run_controller({}, address_plan=delayed)
        self.assertEqual([db.poll_count for db in result.dbs], [2, 2, 2, 2])
        self.assertFalse(any(result.polls))
        self.assertEqual(result.endpoint, "host:6780,host:6781,host:6782,host:6783\n")

    def test_bad_standalone_addresses_timeout_and_cleanup(self):
        for bad in ([], [""], ["host:6783", "host:6784"]):
            with self.subTest(bad=bad):
                result = self.run_controller({}, address_plan=lambda i, n: bad if i == 3 else [f"host:{6780+i}"],
                                             timeout=True)
                self.assertEqual(result.stops, result.dbs)

    def test_shared_gpu_mapping_and_readiness_unchanged(self):
        result = self.run_controller({"SMARTSIM_DB_GPU_IDS": "invalid,invalid"},
                                     ["--use-gpu", "--db-layout", "shared"],
                                     address_plan=lambda i, n: ["host:6780"])
        self.assertEqual(len(result.dbs), 1)
        self.assertEqual(result.dbs[0].poll_count, 1)
        self.assertEqual(result.endpoint, "host:6780\n")
        self.assertNotIn("SMARTSIM_DB_GPU_MAP", result.output)
        for entity in result.dbs[0].entities:
            self.assertNotIn("CUDA_VISIBLE_DEVICES", entity.run_settings.env_vars)
            self.assertEqual(entity.run_settings.exe, ["/fake/redis-server"])

    def test_single_standalone_and_extra_allocated_ids(self):
        result = self.run_controller({"SMARTSIM_DB_GPU_IDS": "GPU-a,GPU-b",
                                      "SMARTSIM_DB_GPU_COUNT": "2"},
                                     ["--use-gpu", "--db-nodes", "1"])
        self.assertEqual(len(result.dbs), 1)
        self.assertEqual(result.dbs[0].entities[0].run_settings.env_vars["CUDA_VISIBLE_DEVICES"], "GPU-a")
        self.assertEqual(result.endpoint, "host:6780\n")

    def test_per_db_cpu_settings_and_memory_preserved(self):
        for extra, cpus in (([], 6), (["--no-cpu-bind"], None),
                            (["--use-default-cpu-settings"], None)):
            result = self.run_controller({"SMARTSIM_DB_GPU_IDS": "invalid,invalid"},
                                         ["--launcher", "slurm", "--db-step-mem", "0",
                                          "--cpu-cores-per-node", "6", *extra])
            for db in result.dbs:
                self.assertEqual(db.cpus, cpus)
                self.assertEqual(db.run_args["mem"], "0")
                self.assertEqual("intra_op_threads" in db.kwargs,
                                 "--use-default-cpu-settings" not in extra)
                self.assertNotIn("CUDA_VISIBLE_DEVICES", db.entities[0].run_settings.env_vars)
            self.assertNotIn("SMARTSIM_DB_GPU_MAP", result.output)


if __name__ == "__main__":
    unittest.main()
