# Stage 1 baseline smoke-test results

These files belong to the initial Pi0 / LIBERO baseline reproduction stage of the
inference-time action-decoupling and temporal-smoothing project.

No proposed smoothing module was applied in these runs. They do not demonstrate
jitter reduction, a baseline-versus-module comparison, or a full benchmark score.

- `pi0_smoke_authorized_20261122/`: Spatial task 0, one episode, no success.
- `pi0_four_suite_smoke_20261122/`: task 0 of each of the four standard suites,
  one episode per suite. Object and Goal succeeded; Spatial and Long did not.

Each directory contains the original evaluator JSON and recorded videos.
Full baseline results and smoothing-module results remain to be added by the author.
