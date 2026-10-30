# Status и evidence

Статусы: VERIFIED (есть лог/вывод), UNVERIFIED (не запускалось), FAILED, BLOCKED.

| ID | Исполнитель | Статус | Evidence |
|---|---|---|---|
| S-000 | Claude | VERIFIED | стартовый kit перенесён из `soup-kernel-challenge-main.zip` (c7c2fcd) |
| S-001 | Codex-2 | VERIFIED (статически) | `tasks/RUNTIME_AUDIT.md`, коммит 712e567; worker_done не доставлен (EPERM песочницы), dispatch abandoned |
| S-002 | Codex-1 | in progress | повтор ctx_9a7cf30d4ab5 после перезапуска с доступом к Orca; уточнение контракта msg_3664d4341b76 |
| S-003 | Codex-1 | planned | |
| S-004 | Codex-2 | in progress | ctx_5128519ffb1a |
| S-005 | Claude | in progress | Kaggle T4 раннер работает (tools/kaggle), см. замеры ниже |

## Замеры на Kaggle T4 (VERIFIED, runs_kaggle/, не в Git)

| Прогон | Коммит | База wall / peak / eval | Solution wall / peak / eval | Checks |
|---|---|---|---|---|
| full ×1 | 9bf579d | 79.29 s / 14345 MiB / 2.04962 | 80.71 s / 12383 MiB / 2.04964 | 9/10; FAIL 4 (1.2%, fused CE не активен — исправлено в 3d103ac) |

Тесты на T4 (9bf579d): `pytest solution` 82 passed; Soup tests 79 passed с патчами.
