#!/usr/bin/env python3
"""Sequential cold-cache A/B runs through the unmodified official harness.

Standard library only; importing/aggregating evidence needs neither torch nor
Kaggle. Negative paired differences (variant - A) mean an improvement in time.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
METRICS = ("wall_s", "loop_s", "first_step_s", "median_step_s", "peak_mib", "eval_loss", "own_share")
CSV_FIELDS = (
    "experiment", "commit", "owner", "family", "params", "wall_s", "loop_s", "first_step_s",
    "peak_mib", "speedup", "memory_ratio", "eval_delta_pct", "checks", "decision",
    "session", "run_id", "variant", "round", "order", "median_step_s", "eval_loss", "own_share",
    "timed_loop_s", "ok", "harness_exit_code", "train_exit_code", "eval_exit_code", "error",
    "result_file", "metrics_file", "local_artifact_root",
)


def parse_variants(spec):
    """A=;B=SOUP_EXP_X=1,SOUP_EXP_Y=1;C= (C can be a no-op control)."""
    variants = []
    for part in spec.split(";"):
        name, separator, settings = part.strip().partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,31}", name):
            raise ValueError("--variants: use A=;B=SOUP_EXP_NAME=1 (unique ASCII labels)")
        if name.lower() == "base" or any(v["name"] == name for v in variants):
            raise ValueError("variant names must be unique; base is reserved")
        environment = {}
        for setting in settings.split(",") if settings else ():
            key, equals, value = setting.strip().partition("=")
            if (not equals or not re.fullmatch(r"SOUP_EXP_[A-Z][A-Z0-9_]*", key)
                    or value != "1" or key in environment):
                raise ValueError("variants accept only unique opt-in SOUP_EXP_NAME=1 flags")
            environment[key] = value
        variants.append({"name": name, "env": environment})
    if len(variants) < 2 or variants[0] != {"name": "A", "env": {}}:
        raise ValueError("provide at least two variants, starting with the unchanged control A=")
    return variants


def schedule(variants, rounds):
    if rounds < 1:
        raise ValueError("--rounds must be positive")
    yield {"name": "base", "env": {}, "side": "base", "round": 0, "order": 0}
    order = 0
    for number in range(1, rounds + 1):
        for variant in variants if number % 2 else reversed(variants):
            order += 1
            yield dict(variant, side="solution", round=number, order=order)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_rows(path):
    if not path.exists():
        return []
    rows = []
    seen = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            key = row["run_id"]
            assert row["schema"] == 1 and isinstance(key, str)
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            raise ValueError(f"invalid registry row {path}:{number}") from exc
        if key in seen:
            if seen[key] != row:
                raise ValueError(f"conflicting registry run_id: {key}")
            continue
        seen[key] = row
        rows.append(row)
    return rows


def append_rows(path, incoming):
    rows = read_rows(path)
    seen = {row["run_id"]: row for row in rows}
    additions = []
    for row in incoming:
        previous = seen.get(row["run_id"])
        if previous is not None:
            if previous != row:
                raise ValueError(f"conflicting registry run_id: {row['run_id']}")
            continue
        seen[row["run_id"]] = row
        additions.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in additions:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return rows + additions


def paired_interval(values, *, samples=10000, seed=303):
    """Resample ROUND pairs, never individual A/B runs or different machines."""
    if not values:
        return {"pairs": 0, "median": None, "ci95": None}
    estimate = statistics.median(values)
    if len(values) < 2:
        return {"pairs": len(values), "median": estimate, "ci95": None}
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    rng = random.Random(seed)
    boot = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(samples))
    return {"pairs": len(values), "median": estimate,
            "ci95": [boot[int((samples - 1) * 0.025)], boot[int((samples - 1) * 0.975)]]}


def summarize(rows, *, samples=10000):
    groups = {}
    for row in rows:
        # Config digest prevents accidental pooling of different variant sets.
        key = (row["session"], row["commit"], row["experiment"], row["config_hash"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        control = {r["round"]: r for r in group if r["variant"] == "A" and r["ok"]}
        names = sorted({r["variant"] for r in group})
        variants = {}
        for name in names:
            all_runs = [r for r in group if r["variant"] == name]
            good = [r for r in all_runs if r["ok"]]
            medians = {metric: statistics.median([r[metric] for r in good if finite(r.get(metric))])
                       if any(finite(r.get(metric)) for r in good) else None for metric in METRICS}
            paired = {}
            if name not in {"A", "base"}:
                for metric in METRICS:
                    pairs = [(r, control[r["round"]]) for r in good if r["round"] in control
                             and finite(r.get(metric)) and finite(control[r["round"]].get(metric))]
                    paired[metric] = paired_interval([v[metric] - a[metric] for v, a in pairs], samples=samples)
                    if metric == "wall_s":
                        paired["wall_delta_pct"] = paired_interval(
                            [100 * (v[metric] / a[metric] - 1) for v, a in pairs if a[metric] > 0], samples=samples)
            variants[name] = {"attempts": len(all_runs), "ok": len(good), "medians": medians,
                              "paired_vs_A": paired}
        summaries.append({"session": key[0], "commit": key[1], "experiment": key[2],
                          "config_hash": key[3], "variants": variants})
    return summaries


def write_results(rows, path):
    # Preserve one CSV row per attempted run, including failures. The singleton
    # base supplies descriptive ratios, not an uncertainty estimate.
    bases = {(r["session"], r["config_hash"]): r for r in rows if r["variant"] == "base" and r["ok"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            data = {field: row.get(field) for field in CSV_FIELDS}
            data["params"] = json.dumps(row["env"], sort_keys=True)
            data.update(checks="not_run", decision="measurement_only")
            base = bases.get((row["session"], row["config_hash"]))
            if row["ok"] and base:
                for field, numerator, denominator in (("speedup", base.get("wall_s"), row.get("wall_s")),
                                                       ("memory_ratio", base.get("peak_mib"), row.get("peak_mib"))):
                    if finite(numerator) and finite(denominator) and denominator > 0:
                        data[field] = numerator / denominator
                if finite(row.get("eval_loss")) and finite(base.get("eval_loss")) and base["eval_loss"] > 0:
                    data["eval_delta_pct"] = 100 * (row["eval_loss"] / base["eval_loss"] - 1)
            writer.writerow(data)
    temporary.replace(path)


def aggregate(registry, results, *, samples=10000):
    rows = read_rows(registry)
    write_results(rows, results)
    summaries = summarize(rows, samples=samples)
    for summary in summaries:
        print(f"\nSession {summary['session']} commit {summary['commit']} ({summary['experiment']})")
        for name, values in summary["variants"].items():
            print(name, f"ok={values['ok']}/{values['attempts']}", "medians", json.dumps(values["medians"], sort_keys=True))
            if values["paired_vs_A"]:
                print("  paired differences vs A (negative time = faster):", json.dumps(values["paired_vs_A"], sort_keys=True))
    print("95% intervals: percentile bootstrap of paired round differences; few rounds give weak evidence.")
    print("A/B measurements do not certify the jury checks; finalists still need full2 + check_rules.")
    return summaries


def extract_record(root, item, *, session, commit, experiment, config_hash, returncode, command, elapsed):
    directory = root / "runs" / f"{item['order']:03d}_r{item['round']:02d}_{item['name']}"
    record = dict(schema=1, experiment=experiment, session=session, commit=commit, config_hash=config_hash,
                  run_id=f"{session}/{config_hash}/{item['order']:03d}", owner="Codex-3", family="infra",
                  variant=item["name"], env=item["env"], round=item["round"], order=item["order"],
                  side=item["side"], timestamp=datetime.now(timezone.utc).isoformat(),
                  harness_exit_code=returncode, harness_wall_s=elapsed, command=command,
                  result_file=str(directory.relative_to(root) / "result.json"),
                  metrics_file=str(directory.relative_to(root) / f"{item['side']}_1/metrics.json"),
                  artifact_root=str(root), cold_cache=True, ok=False, error=None,
                  **{metric: None for metric in METRICS})
    try:
        result = json.loads((directory / "result.json").read_text())
        runs = result["runs"][item["side"]]
        if len(runs) != 1:
            raise ValueError("expected exactly one official training run")
        train = runs[0]
        metrics = json.loads((directory / f"{item['side']}_1/metrics.json").read_text())
        evaluation = result.get("evals", {}).get(item["side"], {})
        steps = metrics.get("step_s", [])
        timed = metrics.get("timed_loop_s")
        own = {name: ms for name, ms in metrics.get("own_kernel_ms", {}).items()
               if not name.startswith(("example_", "triton_poi_", "triton_red_", "triton_per_", "triton_tem_",
                                       "triton_for_", "triton_spl_", "triton_unk_"))}
        record.update(wall_s=train.get("wall_s"), loop_s=metrics.get("train_loop_s"),
                      first_step_s=steps[0] if steps else None, median_step_s=metrics.get("median_step_s"),
                      peak_mib=train.get("gpu_peak_mib"), eval_loss=evaluation.get("mean_loss"),
                      own_share=sum(own.values()) / (1000 * timed) if finite(timed) and timed > 0 else None,
                      own_kernel_ms=own, timed_loop_s=timed, train_exit_code=train.get("exit_code"),
                      eval_exit_code=evaluation.get("exit_code"), task=result.get("task"),
                      versions=result.get("versions"), steps=metrics.get("steps"), cuda_device=metrics.get("cuda_device"))
        record["ok"] = (returncode == 0 and train.get("exit_code") == 0 and metrics.get("exit_code") == 0
                        and evaluation.get("exit_code") == 0 and all(finite(record.get(m)) for m in METRICS)
                        and not result.get("packages_changed") and not metrics.get("triton_watch_error"))
        if not record["ok"]:
            record["error"] = "training/eval/metrics/package/watch validation failed; inspect official artifacts"
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def run_ab(args):
    import time

    variants = parse_variants(args.variants)
    plan = list(schedule(variants, args.rounds))
    out = args.out.resolve()
    # Never reuse an output directory or leave stale result.json available.
    out.mkdir(parents=True, exist_ok=False)
    registry_root = args.registry_root.resolve() if args.registry_root else out / "experiments"
    registry = registry_root / "registry.jsonl"
    config = {"variants": variants, "rounds": args.rounds, "order": plan,
              "commit": args.commit, "session": args.session, "experiment": args.experiment}
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    config["config_hash"] = digest
    (out / "plan.json").write_text(json.dumps(config, indent=2) + "\n")
    # An inherited diagnostic switch must not silently change A or the base.
    environment = {k: v for k, v in os.environ.items() if not k.startswith("SOUP_")
                   and k not in {"TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"}}
    failed = False
    for item in plan:
        directory = out / "runs" / f"{item['order']:03d}_r{item['round']:02d}_{item['name']}"
        if directory.exists():
            raise ValueError(f"run directory must be new: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "harness/run_task.py", "--only", item["side"], "--out", str(directory)]
        # run_task.py removes side_1 before train and sets its TRITON_CACHE_DIR
        # and TORCHINDUCTOR_CACHE_DIR. No profiling or second training is added.
        print(f"round={item['round']} order={item['order']} {item['name']} env={item['env']}", flush=True)
        started = time.monotonic()
        with (directory.parent / (directory.name + ".log")).open("w") as handle:
            try:
                process = subprocess.run(command, cwd=ROOT, env=dict(environment, **item["env"]),
                                         stdout=handle, stderr=subprocess.STDOUT, check=False)
                returncode = process.returncode
            except OSError as exc:
                handle.write(f"{type(exc).__name__}: {exc}\n")
                returncode = 127
        try:
            record = extract_record(out, item, session=args.session, commit=args.commit, experiment=args.experiment,
                                    config_hash=digest, returncode=returncode, command=command,
                                    elapsed=time.monotonic() - started)
            record["artifact_root_relative_to_registry"] = os.path.relpath(out, registry.parent)
            rows = append_rows(registry, [record])
            write_results(rows, registry_root / "results.csv")
        finally:
            # Keep official logs/JSON, but do not export trained adapters or
            # compiler caches as notebook artifacts. Each next run is cold.
            for name in ("adapter", "triton_cache", "inductor_cache"):
                shutil.rmtree(directory / f"{item['side']}_1" / name, ignore_errors=True)
        failed |= not record["ok"]
        print(json.dumps(record, sort_keys=True), flush=True)
    summaries = aggregate(registry, registry_root / "results.csv")
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return int(failed)


def import_registry(source, destination):
    rows = read_rows(source)
    # Preserve remote provenance; add a local root for the downloaded evidence.
    for row in rows:
        row["local_artifact_root"] = str((source.resolve().parent / row["artifact_root_relative_to_registry"]).resolve())
    append_rows(destination, rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-ab", help="T4-side sequential official runs (no submission)")
    run.add_argument("--variants", required=True)
    run.add_argument("--rounds", type=int, default=4)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--registry-root", type=Path)
    run.add_argument("--commit", required=True)
    run.add_argument("--session", required=True)
    run.add_argument("--experiment", default="EXP-003")
    run.set_defaults(handler=run_ab)
    for name in ("aggregate", "import"):
        sub = commands.add_parser(name)
        sub.add_argument("--registry", type=Path, default=ROOT / "experiments/registry.jsonl")
        sub.add_argument("--results", type=Path, default=ROOT / "experiments/results.csv")
        sub.add_argument("--bootstrap-samples", type=int, default=10000)
        if name == "import":
            sub.add_argument("source", type=Path)
        def handler(args):
            if args.command == "import":
                print("Imported rows:", import_registry(args.source, args.registry))
            aggregate(args.registry, args.results, samples=args.bootstrap_samples)
        sub.set_defaults(handler=handler)
    args = parser.parse_args()
    try:
        return args.handler(args) or 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
