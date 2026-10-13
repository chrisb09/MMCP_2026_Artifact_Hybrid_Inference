#!/usr/bin/env python3
"""Pin one MAIA task to a distinct physical core within its inherited affinity.

Opt-in via CMI_NUMA_PIN=1. Keeps one logical CPU per physical core and maps the
task's local rank to a physical core in CPU-ID order (NUMA-ordered on c23g),
mirroring the terrain-solver ``phydll_pin_exec.py`` approach. Then execs argv.
"""

import argparse
import os
import socket
import sys
from pathlib import Path

_THREAD_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS",
    "MLCOUPLING_INTRA_OP_THREADS", "MLCOUPLING_INTER_OP_THREADS",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=0,
                        help="number of physical cores to use (0 = all allowed)")
    parser.add_argument("--rank-env", default=None,
                        help="env var holding the local rank (default: auto)")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command

    try:
        rank_env = args.rank_env
        if not rank_env:
            rank_env = "SLURM_LOCALID" if "SLURM_LOCALID" in os.environ else "OMPI_COMM_WORLD_LOCAL_RANK"
        local_rank = int(os.environ.get(rank_env, "0"))

        allowed = os.sched_getaffinity(0)
        cores = {}
        for cpu in sorted(allowed):
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            key = (int((topology / "physical_package_id").read_text()),
                   int((topology / "core_id").read_text()))
            cores.setdefault(key, cpu)
        pool = list(cores.values())
        budget = args.budget if args.budget > 0 else len(pool)
        budget = min(budget, len(pool))
        if budget < 1:
            raise ValueError("no distinct physical cores in the allowed set")
        chosen = {pool[local_rank % budget]}
        os.sched_setaffinity(0, chosen)
        for name in _THREAD_ENV:
            os.environ[name] = "1"
        if args.log or os.environ.get("CMI_NUMA_PIN_LOG", "0") == "1":
            print(f"CMI_NUMA_PIN host={socket.gethostname()} rank_env={rank_env} "
                  f"local_rank={local_rank} budget={budget} "
                  f"mask={sorted(os.sched_getaffinity(0))}", file=sys.stderr, flush=True)
        if command:
            os.execvp(command[0], command)
    except (KeyError, ValueError, OSError) as exc:
        parser.exit(2, f"CMI_NUMA_PIN error: host={socket.gethostname()}: {exc}\n")


if __name__ == "__main__":
    main()
