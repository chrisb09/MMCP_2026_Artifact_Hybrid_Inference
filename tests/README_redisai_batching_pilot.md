# RedisAI GPU batching pilot

From the repository root, submit only when desired: `sbatch slurm/redisai_batching_pilot.sbatch`. This is independent of existing campaigns. It requests one c23g GPU and uses the existing CUDA SmartSim environment (override its activation script with `SMARTSIM_PYTHON_ENV`). The job requires Slurm to expose exactly one GPU via `CUDA_VISIBLE_DEVICES`. Results, model, controller logs and provenance are stored in a unique `results/redisai_batching_pilot/<job>.*` directory. Slurm's standard output remains in the submission directory.

Job 5251528 failed before the unbatched control model loaded: the Redis server
could not find `libnvrtc.so.12` when loading its Torch backend. The pilot now
loads the project's CUDA environment and adds the same CUDA library path as the
GPU benchmark runner. Database readiness alone does not establish backend readiness.

Two independently connected processes issue requests of 2 and 3 samples to the same model after a readiness gate. The control model has batching disabled. The enabled model has maximum and minimum batch size 5 and a 5000 ms minimum-batch timeout. Its scripted forward returns both `2*x+1` and its actual leading dimension for every sample. Exact checks require the control to report 2 and 3, and both enabled outputs to report 5 with the correct values and split sizes. A timeout or failure to merge is a failure, not a performance result. The finite timeout may flush a partial batch if scheduling is delayed; rerunning can distinguish a scheduling failure from a persistent compatibility issue. GPU execution itself requires the job and cannot be verified offline.

Offline CPU test in the same Python environment: `python3 -m unittest discover -s tests -p test_redisai_batching_pilot.py`. This checks scripted dynamic dimensions and the output oracle, not RedisAI, GPU execution, process synchronization, or Slurm cleanup. The job has readiness, client, controller and Slurm time bounds; cleanup asks the controller to stop its isolated database. Abrupt node failure or forced termination can prevent graceful cleanup, in which case Slurm allocation teardown is the final boundary.
