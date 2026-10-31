# Coordinator handoff (Claude → Codex)

Use when Claude Code hits its usage limit. The Codex coordinator takes over the
same Orca Run with the same rules (AGENTS.md, CLAUDE.md, experiments/best_config.json).

## Launch (account provided by the user; never Islam Anuarbek's account)

```bash
CODEX_HOME='<coordinator account home>' codex --no-daemon --no-alt-screen \
  --ask-for-approval never --sandbox workspace-write \
  -c sandbox_workspace_write.network_access=true \
  -m gpt-6.1-sol -c model_reasoning_effort=xhigh
```

First prompt: "Ты координатор. Прочитай orchestration/COORDINATOR_HANDOFF.md,
AGENTS.md, experiments/best_config.json и продолжай Run run_46ce7c62a997."

## State to resume

- Run: `run_46ce7c62a997`. Workers: Codex-1 `term_6e9a0f2a…` (EXP-013 startup),
  Codex-3 (closed; relaunch with its CODEX_HOME if needed). Codex-2 forbidden.
- Branch `cloud-orchestrator-codex-workers`; remote origin = GitHub soupsuck (no direct push to main).
- BEST_VALIDATED: `f4de4e4` (full2 70.53 s, peak 3769 MiB, 10/10).
- Candidate `96a890c`: EXP-010 CPU launchers default (A/B −3.33 s, CI [−3.66, −2.86]);
  needs unit + full2 + check_rules 10/10, then update best_config.json.
- Queued after that: `run --mode bench --tag exp011` (launch-config sweep),
  `run --mode tokstats --tag exp012` (padding stats, CPU only).
- Rejected: EXP-002, EXP-006 (no gain), EXP-008 (no gain). Memory-only: EXP-009 (−82 MiB, +2.06 s).

## Kaggle

Runner `tools/kaggle/run.py` from a clean worktree (`git worktree add <dir> HEAD`);
`push-code` then `run --mode …`; never commit between push-code and run; one GPU job at a time.
Kaggle token lives only in `~/.kaggle/kaggle.json`.

## Rules

Zero speed regression; peak memory must not grow; do not edit harness/, tests/, task/;
no torch.compile or banned libraries; never claim results without SHA and logs.
