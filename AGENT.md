You are supposed to solve problems or follow a plan on your own. If you fail to fix the problem after 3 independent attempts with different approaches, are essentially stuck — out of ideas or your attempts repeatedly fail — or if the problem is too hard for you, you may escalate the problem.

Assuming you are opencode-go/deepseek-v4-flash, you may normally escalate to opencode-go/deepseek-v4-pro. Specifically, you can use `opencode run --model opencode-go/deepseek-v4-pro <Query>` as a console command to use the model. Ideally, describe the problem extensively, be specific with the query/prompt you work with, and link the files and/or markdown files relevant to the problem/task that are important or relevant. You can also not just use the model for research; you may at times also let it apply fixes in your stead. Still, you are to fairly and accurately evaluate if the information/diagnosis/fixes/ideas that the better model proposes are correct or valuable, ideally — if applicable and sensible — test them.

If opencode-go/deepseek-v4-pro is incapable of solving the given task, or if you judge the task initially to be extremely hard, then you may in these cases use opencode-go/glm-5.2 instead. Be advised though that glm-5.2 is significantly more expensive, so only use it if necessary. However, it is better to use it and solve the problem than to not use it and repeatedly fail with v4 pro and v4 flash a large amount of times. A few failures that lead to a success however are acceptable.

V4 Pro is roughly 10 times as expensive as our normal v4 flash usage.
GLM 5.2 is roughly 50 times as expensive as our normal v4 flash usage.

Still, correctness is in the end more important than cost.

Especially if a situation requires a proper, detailed and non-trivial plan, simply using glm-5.2 might be worthwhile. We can then use our normal v4 flash (no need for an opencode command) to start following the plan — and use a more complex model for the harder tasks/substeps of the plan.

## HPC Cluster Usage Rules

### 1. Devel Node
Use the `devel` node for interactive/development work that requires more resources than the login node. Always use the **default account** when submitting to or requesting the devel node (i.e., do not specify a custom `--account` unless explicitly instructed).

### 2. No Long or Compute-Intensive Tasks on the Login Node
The login node is a shared resource. You only have access to roughly **4–6 cores worth of CPU** there, and the system **will automatically cancel** jobs or processes that hog too much CPU, memory, or GPU resources. Therefore:
- Do **not** run non-trivial compilations on the login node.
- Do **not** run GPU-intensive tasks on the login node.
- Do **not** run long-running compute-intensive processes on the login node.
- Short tasks (a few minutes of light work) are acceptable.
- For anything heavier, submit a job or use the devel node.

### 3. Limit Compile Parallelism to `-j 4` on the Login Node
Even though tools like `nproc` will report **96 cores**, the login node only permits ~4–6 cores of actual CPU usage. When compiling on the login node, always cap parallelism explicitly:
```bash
make -j 4   # or cmake --build . -- -j 4, ninja -j 4, etc.
```
Do **not** rely on `$(nproc)` or similar auto-detection on the login node — it will produce an inaccurate value and the resulting overload will get the process killed.
