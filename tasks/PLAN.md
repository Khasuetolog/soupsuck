# План: Soup Kernel Challenge — волна 1 (память)

## Контекст
Кейс Decentrathon 6.0 (ТЗ: `~/Downloads/Decentrathon 6.0_ Soup Kernel Challenge (1).docx`,
стартовый kit: `~/Downloads/soup-kernel-challenge-main.zip`). Нужен пакет `solution/` с
`apply()`, который harness вызывает до построения модели Soup; свои `@triton.jit` кернелы
должны давать ≥5% времени обучения (шаги 3+), не ухудшать loss >0.5%, время/память >10%,
проходить 10 проверок `harness/check_rules.py` (включая 79 тестов Soup с `SOUP_APPLY_SOLUTION=1`).
Баллы: 35 скорость, 30 память (тай-брейк — память), 15 README. Дедлайн 31.10 18:59 UTC.
Решение пользователя: **сначала память**. Kaggle-токен пользователь даст; репозитория
команды на GitHub пока нет.

## Проверенные факты из kit (влияют на дизайн)
- База: ~80 с, пик ~14 350 MiB (nvidia-smi, `expandable_segments`), eval 2.050.
- `prepare_model_for_kbit_training` держит norms, biases и **tied embedding/lm_head в fp32**;
  LoRA r16/α32/dropout 0.05 на q_proj,v_proj; без grad checkpointing; SDPA; TRL `SFTTrainer`
  (Soup `RewindSFTTrainer`), loss только по ответу.
- **TRL каждый шаг считает entropy и mean_token_accuracy по полным логитам, Soup пишет
  per-row loss в `rewind.jsonl`** — check 5 сравнивает их (±3%). Значит fused CE обязан
  выдавать per-token loss, entropy и argmax, иначе выходы Soup пропадут.
- `torch.compile` по правилам разрешён, но не засчитывается; не используем (строже промпта пользователя).
- Кернелы примера (`example_*`, похожесть ≥0.95) не считаются; свой SwiGLU — только если
  структурно иной (например, сливается с чем-то ещё), иначе риск «lookalike».
- Локально: macOS без GPU/torch/triton → вся GPU-проверка только на Kaggle T4.

## Шаг 0 — подготовка (я, Claude)
1. Перенести стартовый kit в featherduster (распаковать zip поверх, коммит на текущей ветке
   `cloud-orchestrator-codex-workers`). Наши `CLAUDE.md`/`AGENTS.md`/`orchestration/` не трогать;
   orca-soup-agent-kit не устанавливать — его skills использовать как текст в заданиях.
2. `tasks/PLAN.md` (этот план + владение файлами), `tasks/STATUS.md` для evidence.
3. Локальный venv в scratchpad (не в репо) с CPU torch + `soup-cli[train]==0.77.0`,
   `transformers==5.16.1 peft==0.20.0 trl==0.29.1 accelerate==1.14.0` — для чтения исходников
   и CPU-прогона 79 тестов (bitsandbytes/triton на macOS могут не встать — фиксируем как есть).
4. Kaggle-раннер (после получения токена): установить `kaggle` CLI в venv, токен только в
   `~/.kaggle/kaggle.json`. Код на Kaggle доставлять приватным Kaggle Dataset
   (`git archive` → `kaggle datasets version`), т.к. GitHub-репозитория пока нет; свой
   итерационный ноутбук `tools/kaggle/iterate.ipynb` (не `templates/`) копирует dataset в
   `/kaggle/working/repo` и выполняет те же команды, что шаблон: prepare_data, pytest solution,
   run_task.py (`--only solution` или полный), check_rules.py, score.py. `kaggle kernels push`
   → poll status → `kaggle kernels output`. Выбор T4 в metadata проверить по `--help`; если
   CLI не позволяет выбрать T4 — сообщить пользователю, это блокер автозапуска.
5. Финальная сдача (позже): нужен GitHub-репозиторий платформы и публичный ноутбук из шаблона.

## Orca
Загрузить версионный навык orchestration через Orca, проверить liveness handles из
`orchestration/runtime.local.json` и `orca account list --json`, создать Run, отдельные Orca
worktrees от текущей ветки после шага 0. Назначать через
`worker-start --terminal <handle> --worktree <selector>` по `orchestration/task-template.md`.

## Контракт интерфейса (фиксирую до параллельной работы)
`solution/kernels/cross_entropy.py` экспортирует:
```
fused_linear_cross_entropy(hidden[N,H] fp16 cuda, weight[V,H] fp32|fp16 (frozen),
                           targets[N] int64, ignore_index=-100)
  -> per_token_loss[N] fp32 (0 для ignored, differentiable по hidden),
     entropy[N] fp32, argmax[N] int64, lse[N] fp32   (последние три — без градиента)
```
Таргеты уже сдвинуты вызывающим (shift делает интеграция по результатам аудита). Вес
грузится тайлами и приводится к fp16 внутри кернела (без копии 467 MB), dot с fp32
аккумулятором; backward: dX = g·(softmax − onehot) @ W тайлами, dW не считается (assert
`not weight.requires_grad`). `solution/kernels/rmsnorm.py`: `rms_norm(x, weight, eps)` с
autograd fwd/bwd, dtype-поведение = `Qwen2RMSNorm` (уточняется аудитом).

## Задания волны 1
| ID | Исполнитель | Суть | Владение файлами | Зависит |
|---|---|---|---|---|
| S-001 | Codex-2 | Аудит рантайма по установленным исходникам: TRL compute_loss (shift, mask, entropy, accuracy, нормировка), Soup Rewind per-row loss, `Qwen2ForCausalLM.forward` (logits_to_keep, loss_function), dtypes после kbit-prep, Qwen2RMSNorm, PEFT Linear4bit+LoRA forward, что вызывают фейки в `tests/`. Точка перехвата до материализации логитов | `tasks/RUNTIME_AUDIT.md` | шаг 0 |
| S-002 | Codex-1 | Fused linear CE fwd/bwd + entropy/argmax по контракту, torch-эталон (чанковый), тесты (малый V, V=151936, хвосты, всё замаскировано, большие логиты, fp32 W) | `solution/kernels/__init__.py`, `solution/kernels/cross_entropy.py`, `solution/tests/test_cross_entropy.py` | контракт |
| S-003 | Codex-1 | RMSNorm fwd/bwd (fp32 статистики, dgamma поэтапной редукцией, dgamma только если нужен) | `solution/kernels/rmsnorm.py`, `solution/tests/test_rmsnorm.py` | S-001 (dtype), после S-002 |
| S-004 | Codex-2 | Интеграция: `apply()` идемпотентный, guards для фейков тестов, перехват loss/метрик TRL+Soup без полных логитов, патч RMSNorm, fallback на оригинал вне поддерживаемых условий; integration-тест; 79 тестов локально если возможно | `solution/__init__.py`, `solution/patch.py`, `solution/autograd.py`, `solution/utils.py`, `solution/tests/test_integration.py` | S-001, интерфейсы S-002/S-003 |
| S-005 | Claude | Ревью диффов, интеграция веток, Kaggle T4: pytest solution, run_task `--only solution --profile`, затем полный run + check_rules | `tasks/STATUS.md`, `tools/kaggle/` | S-002..S-004 |

Пример `solution/swiglu.py` оставить работать (−2 GB), но в зачёт не идёт.

## Волна 2 (после первого T4-профиля, решаю по цифрам)
Слияние LoRA-пути q/v (dropout+A+B+scale+residual), удаление pad внутри батча, свой
fused-кернел для fp32↔fp16 кастов/RMSNorm+residual, NF4 dequant. Порядок — по profile.json.
README по структуре 11.1 ТЗ и SUBMISSION.md — в конце, только с измеренными числами.

## Проверка
- Локально: юнит-тесты не запускаются (нет GPU) — только статическое ревью и CPU-тесты Soup.
- Kaggle T4: `python -m pytest solution -q`; `run_task.py --only solution --profile` (доля
  кернелов, пик); полный `run_task.py` + `check_rules.py` — все 10 PASS; `score.py`.
- Каждый worker сдаёт ровно один `worker_done` с diff, командами и реальными логами;
  неисполненное помечается UNVERIFIED.
