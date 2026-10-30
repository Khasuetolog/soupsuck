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
| solution (профиль) | 25dc62d | — | 81.27 s / 7983 MiB / 2.04962, share 3.35% | fused head active (S-009) |
| solution default | 0e12e01 | — | 89.81 s / 8003 MiB / 2.04967, share 3.63%, median 0.82 s | другая машина |
| solution `SOUP_FUSED_NORM_FP16=1` | 0e12e01 | — | 76.64 s / 5843 MiB / 2.04966, share 3.96%, median 0.72 s | → сделано по умолчанию в b3a3432 |
| **full ×1** | **de836e0** | 77.47 s / 14345 MiB / 2.04964 | **70.05 s / 5843 MiB / 2.04962** | **10/10 PASS**; speedup 1.106, memory 2.455, share 8.6% |
| **full ×1** | **ce1ff09** | 82.25 s / 14345 MiB / 2.0496 | **73.54 s / 5863 MiB / 2.04969** | **10/10 PASS**; speedup 1.118, memory 2.447, share 12.1% (NF4 включён); pytest solution 214 passed / 4 failed (тесты устарели, см. S-022) |
| solution MLP store | 8f8ebbc | — | 79.57 s / 5843 MiB / 2.04966, loop 47.05 s | |
| solution MLP recompute | 8f8ebbc | — | 86.39 s / 4143 MiB / 2.04966, loop 53.63 s | |
| full ×1 `SOUP_MLP_RECOMPUTE=1` | ce1ff09 | 97.75 s / 14345 MiB / 2.04964 | 93.95 s / 4123 MiB / 2.04968 | 10/10 PASS; speedup **1.040 (<1.05 = 0 баллов скорости)**, memory 3.479, share 10.6% → дефолт остаётся store |
| **full ×1** | **585de28** | 87.56 s / 14345 MiB / 2.04965 | **71.30 s / 5403 MiB / 2.04971** | **10/10 PASS**; speedup 1.228, memory 2.655, share 14.8%; **pytest solution 247 passed / 0 failed**; Soup 79 passed |
| **full ×1 (unpad + packed head)** | **42423bf** | 78.32 s / 14345 MiB / 2.04968 | **63.88 s / 4775 MiB / 2.04967** | **10/10 PASS**; speedup 1.226, memory 3.004, share 16.6%; pytest solution 300 passed / 1 failed (test_unpad, разбирается S-032) |
