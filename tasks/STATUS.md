# Status и evidence

Статусы: VERIFIED (есть лог/вывод), UNVERIFIED (не запускалось), FAILED, BLOCKED.

| ID | Исполнитель | Статус | Evidence |
|---|---|---|---|
| S-000 | Claude | VERIFIED | стартовый kit перенесён из `soup-kernel-challenge-main.zip` (c7c2fcd) |
| S-001 | Codex-2 | VERIFIED (статически) | `tasks/RUNTIME_AUDIT.md`, коммит 712e567; worker_done не доставлен (EPERM песочницы), dispatch abandoned |
| S-002 | Codex-1 | in progress | повтор ctx_9a7cf30d4ab5 после перезапуска с доступом к Orca; уточнение контракта msg_3664d4341b76 |
| S-003 | Codex-1 | planned | |
| S-004 | Codex-2 | in progress | ctx_5128519ffb1a |
| S-005 | Claude | BLOCKED | kaggle CLI 2.2.5 установлен (venv в scratchpad), ждём `~/.kaggle/kaggle.json` |
