# Robustness runner mode

Added `robust` to `tools/kaggle/run.py` on HEAD
`c3781d468b7412a7aa4bf91caf9ca9f53dae696b`. It creates one private T4
notebook run with two robustness cases, each executed in one official
`harness/run_task.py --repeats 2` process: base, solution, solution, base.
`harness/check_rules.py` runs immediately after each case.

The official harness already exposes `--seed`, `--train-file` and
`--repeats`; `task/prepare_data.py` only exposes an output directory and fixes
its first 256 rows. The first case trains on the unchanged official train file
with the task seed incremented by one. The second uses the official pinned
public dataset revision and a deterministic 256-row sample with seed 15015
from indices `[256, 40000)`. This excludes the original train rows `[0, 256)`
and fixed public evaluation rows `[40000, 40256)`. The generated JSONL is
written to `/kaggle/working/out/robust_train.jsonl`; eval file, task config and
harness implementation stay unchanged.

For each case the notebook records speedup, base/solution peak-memory ratio,
evaluation loss delta, and all ten check results in `out/robust.json`; it also
prints a short row. Full per-case `result.json` and `checks.json` files and logs
remain under `out/robust/{seed,subset}/`. Adapter weights and compiler caches
are removed in notebook cleanup. Existing runner modes use their original
notebook execution cells.

Local validation (no Kaggle invocation):

```sh
python3 -m py_compile tools/kaggle/run.py
python3 tools/kaggle/run.py run --mode robust --dry-run --tag exp015robust
git diff --check
```

All passed. The rendered notebook is T4-enabled; its code cell compiles, and
inspection confirms both repeats, the alternate data output path, JSONL
newlines, rules checks and base/solution/solution/base ordering. No GPU run,
Kaggle submission or commit was made.

After review and commit, run:

```sh
python3 tools/kaggle/run.py run --mode robust --tag robust
```
