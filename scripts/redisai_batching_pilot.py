#!/usr/bin/env python3
"""Standalone RedisAI batching experiment; requires an isolated SSDB."""
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import queue
import time
import traceback
from pathlib import Path

import numpy as np
import torch


class BatchWitness(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x * 2.0 + 1.0, torch.ones_like(x) * x.size(0)), dim=1)


def expected(x, batch_size):
    return np.concatenate((x * 2 + 1, np.full_like(x, batch_size)), axis=1)


def check_output(x, output, batch_size):
    np.testing.assert_array_equal(output, expected(x, batch_size))


def worker(index, size, model, gate, ready, results):
    try:
        from smartredis import Client
        client = Client(cluster=False, logger_name="batch-pilot-worker")
        x = (np.arange(size, dtype=np.float32) + 10 * index).reshape(-1, 1)
        inp, out = f"{model}:in:{index}", f"{model}:out:{index}"
        client.put_tensor(inp, x)
        ready.put(index)
        if not gate.wait(30):
            raise RuntimeError("request gate timed out")
        client.run_model(model, [inp], [out])
        results.put((index, x, client.get_tensor(out), None))
    except BaseException:
        results.put((index, None, None, traceback.format_exc()))


def run_case(model, sizes, forward_size):
    ctx = mp.get_context("spawn")
    gate, ready, results = ctx.Event(), ctx.Queue(), ctx.Queue()
    processes = [ctx.Process(target=worker, args=(i, n, model, gate, ready, results))
                 for i, n in enumerate(sizes)]
    try:
        for p in processes:
            p.start()
        deadline = time.monotonic() + 60
        for _ in processes:
            ready.get(timeout=max(0.01, deadline - time.monotonic()))
        gate.set()
        seen = set()
        for _ in processes:
            i, x, output, error = results.get(timeout=max(0.01, deadline - time.monotonic()))
            if error:
                raise RuntimeError(error)
            if i in seen or i not in range(len(sizes)):
                raise RuntimeError(f"Unexpected result index {i}")
            seen.add(i)
            check_output(x, output, forward_size if forward_size else sizes[i])
            print(json.dumps({"model": model, "request": i, "input": x.tolist(),
                              "output": output.tolist()}), flush=True)
    except queue.Empty as exc:
        raise RuntimeError(f"Timed out waiting for workers in {model}") from exc
    finally:
        gate.set()
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            if p.pid is not None:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5)
        ready.close()
        results.close()


def main():
    from smartredis import Client
    sizes = (2, 3)
    path = Path("batch_witness.pt")
    torch.jit.script(BatchWitness()).save(str(path))
    provenance = {"python": platform.python_version(), "torch": torch.__version__,
                  "smartredis": importlib.metadata.version("smartredis"),
                  "smartsim": importlib.metadata.version("smartsim"),
                  "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "ssdb": os.environ["SSDB"], "sizes": sizes,
                  "batch_size": 5, "min_batch_size": 5, "min_batch_timeout_ms": 5000}
    print(json.dumps(provenance, sort_keys=True), flush=True)
    client = Client(cluster=False, logger_name="batch-pilot-main")
    model = path.read_bytes()
    client.set_model("pilot_off", model, "TORCH", device="GPU:0")
    run_case("pilot_off", sizes, 0)
    client.set_model("pilot_on", model, "TORCH", device="GPU:0",
                     batch_size=5, min_batch_size=5, min_batch_timeout=5000)
    run_case("pilot_on", sizes, 5)
    print("PASS: unbatched controls and merged GPU forward verified", flush=True)


if __name__ == "__main__":
    main()
