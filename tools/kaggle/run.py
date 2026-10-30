#!/usr/bin/env python3
"""Ship committed code in a private dataset and run the jury harness on T4.

Only Python's standard library is used. Kaggle CLI 2.2.5 command contracts:
docs/{kernels,kernels_metadata,datasets,datasets_metadata,configuration}.md.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATASET_SLUG = "soup-kernel-code"
POLL_SECONDS = 30
TIMEOUT_SECONDS = 90 * 60


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def kaggle(*args, check=True):
    # config view can contain credentials in proxy URLs; NEVER echo raw CLI
    # output or attach it to CalledProcessError, including on authentication errors.
    command = [os.environ.get("KAGGLE_BIN", "kaggle"), *map(str, args)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if check and result.returncode:
        raise RuntimeError(f"Kaggle {' '.join(args[:2])} failed (exit {result.returncode}); "
                           "check CLI authentication/access separately")
    return result


def username(dry_run=False):
    user = os.environ.get("KAGGLE_USERNAME", "")
    if not user and not dry_run:
        output = kaggle("config", "view").stdout
        match = re.search(r"(?im)^\s*-?\s*username\s*:\s*([a-zA-Z0-9_-]+)\s*$", output)
        user = match.group(1) if match else ""
    if not user and dry_run:
        return "dry-run-user"
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", user) or user.lower() == "none":
        raise RuntimeError("Set KAGGLE_USERNAME or configure the Kaggle CLI username")
    return user


def commit():
    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        print("WARNING: working tree is dirty; dataset contains committed HEAD only", file=sys.stderr)
    print("Commit:", sha)
    return sha


def checked_archive(directory, sha):
    """git archive cannot read home credentials; also reject tracked credentials/links."""
    archive = directory / "source.zip"
    subprocess.run(["git", "-C", str(ROOT), "archive", "--format=zip",
                    f"--output={archive}", sha], check=True)
    hashes = {}
    with zipfile.ZipFile(archive) as source:
        if source.comment.decode("ascii").strip() != sha:
            raise RuntimeError("git archive ZIP commit comment does not match HEAD")
        for item in source.infolist():
            path = PurePosixPath(item.filename)
            if (path.is_absolute() or ".." in path.parts or ".kaggle" in path.parts
                    or path.name in {"kaggle.json", "auth.json"}
                    or stat.S_ISLNK(item.external_attr >> 16)):
                raise RuntimeError(f"Archive contains forbidden credential path or link: {path}")
            if not item.is_dir():
                hashes[str(path)] = hashlib.sha256(source.read(item)).hexdigest()
    manifest = {"commit": sha, "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "files": hashes}
    write_json(directory / "source-manifest.json", manifest)
    return manifest


def push_code(args):
    user, sha = username(), commit()
    dataset = f"{user}/{DATASET_SLUG}"
    with tempfile.TemporaryDirectory(prefix="soup-kaggle-code-") as tmp:
        directory = Path(tmp)
        checked_archive(directory, sha)
        with tempfile.TemporaryDirectory(prefix="soup-kaggle-metadata-") as probe:
            exists = kaggle("datasets", "metadata", dataset, "-p", probe, check=False)
            if exists.returncode == 0:
                metadata = json.loads((Path(probe) / "dataset-metadata.json").read_text())
                if metadata.get("isPrivate") is not True:
                    raise RuntimeError("Existing dataset is not confirmed private; set its visibility to private")
                update = True
            elif re.search(r"\b404\b|\bnot found\b", exists.stdout + exists.stderr, re.I):
                update = False
            else:
                raise RuntimeError("Cannot determine dataset existence; check Kaggle authentication/access")
        write_json(directory / "dataset-metadata.json", {
            "title": "Soup Kernel Code", "id": dataset,
            "licenses": [{"name": "apache-2.0"}],
            "description": f"Committed source archive, git HEAD {sha}.",
        })
        if update:
            kaggle("datasets", "version", "-p", directory, "-m", f"git {sha}", "--keep-tabular")
        else:
            # Default private. Never pass --public.
            kaggle("datasets", "create", "-p", directory, "--keep-tabular")
    print(f"Private dataset submitted: {dataset}; commit {sha}")


def build_kernel(mode, user, sha):
    directory = Path(tempfile.mkdtemp(prefix="soup-kaggle-kernel-"))
    # Unique slug prevents an overlapping/latest-version run from supplying
    # someone else's output. Title follows the slug rule in CLI docs.
    stamp = datetime.now(timezone.utc).strftime("%y%m%d-%H%M%S-%f")
    slug = f"soup-{mode}-{sha[:7]}-{stamp}"
    metadata = {
        "id": f"{user}/{slug}", "title": slug.replace("-", " "),
        "code_file": "iterate.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4", "dataset_sources": [f"{user}/{DATASET_SLUG}"],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }
    notebook = json.loads((Path(__file__).parent / "iterate.ipynb").read_text())
    notebook["cells"][1]["source"] = [
        f"MODE = {mode!r}\n", f"EXPECTED_SHA = {sha!r}\n",
        f"DATASET = {metadata['dataset_sources'][0]!r}\n",
    ]
    write_json(directory / "iterate.ipynb", notebook)
    write_json(directory / "kernel-metadata.json", metadata)
    return directory, metadata


def terminal_status(output):
    # CLI emits e.g. has status "KernelWorkerStatus.COMPLETE".
    match = re.search(r"\bstatus\s*[:=]?\s*[\"']?(?:KernelWorkerStatus\.)?([a-z_]+)",
                      output, re.I)
    if not match:
        raise RuntimeError("Unrecognized Kaggle status response; kernel was not re-submitted")
    return match.group(1).lower()


def summarize(directory):
    for path in sorted(directory.rglob("result.json")):
        value = json.loads(path.read_text())
        summary = value.get("summary", {})
        print(f"\n{path.relative_to(directory)}")
        for side in ("base", "solution"):
            values = summary.get(side, {})
            if values:
                print(side, json.dumps({k: values.get(k) for k in
                                       ("ok", "wall_s", "gpu_peak_mib", "eval_loss")}))
        print("ratios:", json.dumps(summary.get("ratios", {})))
    for path in sorted(directory.rglob("checks.json")):
        for check in json.loads(path.read_text()).get("checks", []):
            print("PASS" if check.get("ok") else "FAIL", check.get("check"), check.get("detail", ""))
    for path in sorted(directory.rglob("metrics.json")):
        if "solution" not in str(path.relative_to(directory)):
            continue
        metrics = json.loads(path.read_text())
        own = metrics.get("own_kernel_ms", {})
        elapsed = metrics.get("timed_loop_s")
        share = sum(own.values()) / (1000 * elapsed) if elapsed else None
        print(path.relative_to(directory), "own_kernel_ms:", json.dumps(own),
              "share_of_timed_loop:", f"{share:.2%}" if share is not None else "unavailable")
    for path in sorted(directory.rglob("pytest*.log")):
        print(f"\n{path.relative_to(directory)} (tail)")
        print("\n".join(path.read_text(errors="replace").splitlines()[-12:]))
    statuses = list(directory.rglob("status.json"))
    for path in statuses:
        print("Notebook status:", path.read_text())
    return bool(statuses) and all(json.loads(p.read_text()).get("ok") for p in statuses)


def run(args):
    user, sha = username(args.dry_run), commit()
    directory, metadata = build_kernel(args.mode, user, sha)
    print("Kernel directory:", directory)
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        print("Dry run: Kaggle CLI was not invoked; kernel directory is retained")
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "runs_kaggle" / f"{stamp}_{args.mode}_{sha[:7]}"
    output.mkdir(parents=True)
    shutil.copytree(directory, output / "submitted")
    write_json(output / "launch.json", {"commit": sha, "kernel": metadata["id"], "mode": args.mode})
    kaggle("kernels", "push", "-p", directory, "--accelerator", "NvidiaTeslaT4",
           "--timeout", str(TIMEOUT_SECONDS))
    deadline = time.monotonic() + TIMEOUT_SECONDS
    status = "unknown"
    try:
        while time.monotonic() < deadline:
            response = kaggle("kernels", "status", metadata["id"])
            status = terminal_status(response.stdout)
            print(f"{datetime.now(timezone.utc).isoformat()} {metadata['id']}: {status}", flush=True)
            write_json(output / "remote-status.json", {"status": status})
            if status in {"complete", "completed", "error", "failed", "cancelled", "canceled"}:
                break
            time.sleep(POLL_SECONDS)
        else:
            status = "timeout"
    finally:
        # Also attempt collection after a remote failure/timeout; never resubmit.
        downloaded = kaggle("kernels", "output", metadata["id"], "-p", output, check=False)
        write_json(output / "download-status.json", {"exit_code": downloaded.returncode,
                                                   "remote_status": status})
        ok = summarize(output)
        print("Artifacts:", output)
    return 0 if status in {"complete", "completed"} and downloaded.returncode == 0 and ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    push = commands.add_parser("push-code", help="Publish committed HEAD as a private Kaggle dataset")
    push.set_defaults(handler=push_code)
    launch = commands.add_parser("run", help="Run a private T4 notebook and download its outputs")
    launch.add_argument("--mode", choices=("unit", "solution", "full", "full2"), required=True)
    launch.add_argument("--dry-run", action="store_true", help="Build notebook/metadata without invoking Kaggle")
    launch.set_defaults(handler=run)
    args = parser.parse_args()
    try:
        return args.handler(args) or 0
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
        # Avoid subprocess exceptions that include captured config output.
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
