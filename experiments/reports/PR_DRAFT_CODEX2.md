# Codex-2: ревью robust и черновик PR

## Область ревью

Проверен коммит `8382a1c7ef964d38cd51030767778a34c8ae67b2` относительно его
родителя `c3781d468b7412a7aa4bf91caf9ca9f53dae696b`. Это независимое ревью
runner, а не повторная GPU-валидация решения. Исходники не исправлялись;
единственный файл, созданный этим исполнителем, — данный отчёт. Commit, push,
публикация PR и запуск Kaggle не выполнялись.

Во время ревью общий HEAD продвинулся. Находки ниже относятся именно к
`8382a1c`; отдельно указано замеченное исправление в последующем коммите.
Локальной ветки `release/final` при проверке не было, поэтому вторая часть —
черновик для координатора, а не подтверждение состава ещё не созданного PR.

## Находки

### P1 — robust останавливается при чтении YAML как JSON

**Место:** `tools/kaggle/run.py:249` в `8382a1c`.

`json.loads((REPO / 'task' / 'task.yaml').read_text())` читает неизменённую
YAML-конфигурацию, начинающуюся с комментария. После подготовки данных и subset
получается `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`.
Ни один вызов harness или `check_rules` не достигается; `robust.json` отсутствует.
Notebook сохраняет ошибку и отрицательный status, поэтому ложного PASS здесь нет.

Точное минимальное воспроизведение, привязанное к проверяемому SHA:

```bash
python3 - <<'PY'
import json, subprocess
sha = '8382a1c7ef964d38cd51030767778a34c8ae67b2'
text = subprocess.check_output(['git', 'show', sha + ':task/task.yaml'], text=True)
try:
    json.loads(text)['training']['seed']
except json.JSONDecodeError as exc:
    print(type(exc).__name__ + ': ' + str(exc))
PY
```

Также выполнена сама сгенерированная robust-ветка с заглушкой внешних команд и
реальным YAML из этого SHA: достигнуты только `prepare_data` и `robust_subset`,
затем воспроизведено указанное исключение.

**Исправление:** читать YAML через `yaml.safe_load`, как делает официальный
harness, сохранив неизменным `task/task.yaml`. Координатор уже внёс именно это
в `ed59fa5251528cf5b0ebda3675d65e09394e1c9c` (`run.py:249`, `run.py:250`).
Diff исправления проверен статически; выполнение исправленной ветки с настоящим
PyYAML локально **не проверено**: в доступном `python3` PyYAML отсутствует.

### P2 — harness перезаписывает лог генерации subset

**Места:** `tools/kaggle/run.py:247` и `tools/kaggle/run.py:256` в `8382a1c`.

Подготовка JSONL вызывает `command('robust_subset', ...)`. Позже case `subset`
получает то же имя через `command('robust_' + name, ...)`. Общая функция
`command` в `tools/kaggle/iterate.ipynb`, code-cell 2, строки 22–38, открывает
`OUT / (name + '.log')` в режиме `w`. Поэтому успешное начало второго harness
стирает `robust_subset.log` генератора с количеством строк, seed и границами
выборки. Оба status-entry остаются, но указывают на один перезаписанный лог;
вывод генератора ещё может сохраниться в выводе notebook-cell.

Воспроизведено исполнением неизменённых `save_status` и `command`, извлечённых
через AST из notebook на указанном SHA: последовательные локальные команды
с одинаковым именем печатали разные маркеры. После второй команды в файле
остался только второй маркер. GPU и Kaggle не использовались.

**Исправление:** дать генератору отдельное имя, например `robust_subset_data`.
На момент просмотра последующего HEAD `785614d839883ca5a53e4a856fd2bd5dea8e89c1`
эта коллизия ещё присутствовала. Обе находки отправлены координатору через Orca.

## Проверка требований и evidence

Все проверки снимка ниже выполнены на
`8382a1c7ef964d38cd51030767778a34c8ae67b2` без внешних запросов.

| Требование | Результат и основание |
| --- | --- |
| `harness/`, `task/`, корневой `tests/` не изменены | Подтверждено `git diff-tree --no-commit-id --name-status -r 8382a1c`. Только `tools/kaggle/run.py` и `experiments/reports/ROBUST.md`. Дополнительно `git diff --name-only origin/main...8382a1c -- harness task tests` пустой; это сравнение с уже имеющимся локальным ref, без fetch. |
| Альтернативные строки исключают публичный eval | `run.py:236` импортирует официальные `DATASET` и `REVISION`; `run.py:240` выбирает уникальные индексы только из `[256, 40000)`. Официальные интервалы: train `[0, 256)`, eval `[40000, 40256)` (`task/prepare_data.py:16`, `:17`). Пересечение по индексам математически невозможно. Это не утверждение об отсутствии текстовых дублей в исходном датасете или пересечения с неизвестными hidden rows. |
| Воспроизводимость subset | Выполнен настоящий вложенный serializer с fake `datasets.load_dataset` и перенаправлением выходного файла во временный каталог. Получены 256 уникальных индексов; min 351, max 39765; пересечение с публичными train/eval индексами пустое. Сохраняются только `instruction`, `input`, `output`, JSONL корректно разбивается по строкам. Настоящий датасет не скачивался. |
| Порядок запусков | Вызов содержит `--repeats 2` (`run.py:257`). Исполнен настоящий scheduling loop из `harness/run_task.py:187` с записывающей заглушкой `train`: `base_1, solution_1, solution_2, base_2`. |
| `check_rules` | Вызов есть после каждого успешного harness: `run.py:260`, `:261`, правильный `--result`, без `--skip-tests`. Check 10 действительно запускает Soup tests (`harness/check_rules.py:453`); `checks.json` пишется рядом с выбранным result (`:262`). Но в исходном `8382a1c` эти вызовы недостижимы из-за P1. GPU checks фактически не запускались. |
| Синтаксис | `compile` для runner и всех code-cells после настоящего `robust_notebook` проходит; синтаксическая проверка не обнаруживает P1. Функция загружалась из `git show` через AST, без импорта тренировочного стека. |
| Секреты | В добавленных строках не обнаружены шаблоны GitHub/provider tokens, AWS access keys или private-key blocks. Архив снимка, прочитанный в памяти, не содержит проверенных credential-файлов (`kaggle.json`, `auth.json`, `.env`, `*.env`, `api_key.txt`), `.kaggle` или symlinks. Новая ветка не читает credentials и выводит публичные строки/метрики; существующий archive/hash bootstrap и приватность notebook сохранены. Значения секретов не читались и не выводились. |

Зафиксированный вывод локальных диагностик:

```text
REPRODUCED actual branch: JSONDecodeError after prepare_data/robust_subset; harness_calls=0; check_calls=0; robust.json=absent
PASS subset serializer with fake public data: rows=256; min=351; max=39765; eval_index_overlap=0; only_official_fields=True
PASS unchanged harness scheduling: base_1, solution_1, solution_2, base_2
REPRODUCED tools/kaggle/run.py:247,256: duplicate robust_subset log name overwrites data-generation provenance
LOCAL_DEPENDENCY: PyYAML unavailable; fixed-branch execution requires the pinned training environment
```

Отдельная команда
`KAGGLE_USERNAME=review-user PYTHONDONTWRITEBYTECODE=1 python3 tools/kaggle/run.py run --mode robust --dry-run --tag review-codex2`
успешно создала приватный T4 notebook, не вызывая Kaggle CLI. Она исполнилась уже
на `ed59fa5251528cf5b0ebda3675d65e09394e1c9c` с предупреждением о чужом dirty-file,
поэтому **не используется** как evidence исходного SHA или прохождения robust.
Локальные временные пути и чужие изменения в отчёт не включены.

`robust.json` сейчас записывается лишь после полного успеха обоих cases, а
assert при сбое завершает ветку. Поэтому отсутствие агрегата следует читать
вместе с `status.json`, per-case `result.json`, `checks.json` и логами, а не как
отсутствие выполненной попытки. Это ограничение диагностики, отдельный баг ему
в данном ревью не присваивается.

Фактические GPU-запуски robust, численная устойчивость на альтернативных данных,
новый `full2`, solution suite и Soup suite этим исполнителем **не запускались**.

---

# Черновик PR: `release/final` → `main`

Часть ниже предназначена для тела Draft PR. Все численные результаты взяты
только из `README.md`, `SUBMISSION.md` и `experiments/best_config.json`.
Исторические результаты не переназначаются будущему release SHA.

## Summary

Интегрируем итоговое Triton-решение и инструменты воспроизведения фиксированного
Qwen2.5-1.5B-Instruct NF4/LoRA задания. Подтверждённая конфигурация
`96a890c57a6c9dd28ca92649e27a7a992aa3de51` с пустым набором environment overrides
даёт **1.338× ускорение** и **3.806× base/solution по пиковой GPU-памяти** в
одной Kaggle T4 сессии; публичный eval loss отличается на **−0.001 %**.

Решение объединяет unpadding, асинхронный offload MLP, chunked frozen-head CE,
собственные Triton-ядра и предварительную компиляцию с подготовкой CPU launchers
в дочернем процессе. Финальная документация описывает принятые и отклонённые
эксперименты, границы численной эквивалентности и команды воспроизведения.

## Why

Задача оценивает весь `soup train`, включая запуск и компиляцию, и пиковую
память при сохранении качества и Soup outputs. Поэтому подготовка launchers
снижает стартовые затраты, а offload и исключение padding уменьшают память и
работу в тренировочном цикле. Решения о defaults опираются на парные измерения
внутри одной сессии: baseline между сессиями варьировался от 77.7 до 91.4 s.
Более экономный, но медленный эксперимент остаётся отдельным экспериментом.

## Files

- `solution/precompile.py`, `solution/mlp.py`, `solution/kernels/nf4.py`,
  `solution/lora.py`, `solution/lora_pair.py` и собственные `solution/tests/`:
  принятые улучшения, opt-in экспериментальные пути и их проверки.
- `tools/kaggle/`: runner, A/B registry, kernel benchmark, token statistics,
  startup profiling и дополнительный robust mode.
- `README.md`, `SUBMISSION.md`, `experiments/best_config.json`,
  `experiments/registry.jsonl`, `experiments/results.csv`, `experiments/reports/`,
  review-отчёты в `tasks/` и coordinator handoff: результаты и воспроизводимость.
- Защищённые `harness/`, `task/` и корневой `tests/` сохранены; официальный
  evaluator, checks и scoring используются без изменений.

Состав файлов нужно сверить с фактическим diff созданной `release/final` перед
переводом PR из Draft.

## Test evidence

Документированные результаты для `BEST_VALIDATED`,
`96a890c57a6c9dd28ca92649e27a7a992aa3de51`:

| Команда | Результат в источниках |
| --- | --- |
| `python -m pytest solution -q` | 518 passed, 1 skipped, 0 failed; skipped относится к отклонённому opt-in эксперименту. |
| `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q` | 79/79 Soup regression tests. |
| `python harness/run_task.py --repeats 2` и `python harness/check_rules.py` | 10/10 checks; base, solution, solution, base. |

Источник: `README.md`, `SUBMISSION.md` и `BEST_VALIDATED.evidence` в
`experiments/best_config.json`. Unit-артефакты указаны там как
`runs_kaggle/*unit_96a890c_promote010*`, полные GPU-артефакты — ниже.
Они локальные и не включены в Git; автор ревью не запускал эти тесты заново и
не подтверждает их по первичным логам. Перед merge нужны доступные логи и
актуальные evidence для зафиксированного release SHA, включая повторную
проверку изменений после `BEST_VALIDATED` и независимое ревью.

## GPU evidence

Источник измерений — одна Kaggle Tesla T4 сессия на solution commit
`96a890c57a6c9dd28ca92649e27a7a992aa3de51`, без SOUP overrides:
`runs_kaggle/20261031T133425573730Z_full2_96a890c_promote010/out/`
(локальные артефакты; источник пути — `README.md` и `SUBMISSION.md`).

| Метрика | Base | Solution | Сравнение |
| --- | ---: | ---: | --- |
| Wall time, среднее двух повторов | 87.61 s | 65.49 s | 1.338× |
| Wall time отдельных повторов | 91.37 / 83.85 s | 65.11 / 65.87 s | Одна сессия |
| Пиковая GPU-память, nvidia-smi | 14,345 MiB | 3,769 MiB | 3.806× base/solution |
| Публичный eval loss | 2.04966 | 2.04964 | −0.001 % |
| Доля собственных Triton-ядер, худший повтор | — | 13.3 % | Check 4 PASS |
| Loss curve | — | первый шаг 0.02 %, среднее 0.01 % | PASS |
| Проверки harness | — | 10/10 | ALL CHECKS PASSED |

Это опубликованные результаты проверенной конфигурации, а не измерения
будущего release-коммита. Публичный сохранённый Kaggle notebook пока отмечен
`<KAGGLE_LINK>`; финальный commit — `<FINAL_COMMIT>`. Перед завершением выпуска
нужно заполнить их согласованно в README/SUBMISSION и приложить доступные логи.
Новые GPU-попытки и upload в рамках этого ревью не выполнялись.

## Risks

- Fast head ограничен Qwen2 labelled training с frozen bias-free dense head
  и проверенным T4/fp16 окружением; unsupported случаи используют fallback.
- Ранний fp16 norm output сохраняет dropout masks/RNG, но меняет округление
  значений и накопление общих градиентов. Публичные проверки не доказывают
  эквивалентность на hidden rows или других seed.
- Код после `BEST_VALIDATED` содержит экспериментальные пути и инструменты;
  старые результаты не подтверждают весь будущий release diff. Исправления
  robust из независимого ревью требуют проверки перед использованием режима.
- PR остаётся Draft до evidence актуального release SHA, доступных артефактов
  и независимого ревью. Публичные submission links и team placeholders ещё
  нужно заполнить.

## Rollback

Откатывать интеграцию отдельным revert PR с независимым ревью, сохранив
исходные логи и результаты экспериментов. Последнюю проверенную конфигурацию
восстанавливать по `BEST_VALIDATED` из `experiments/best_config.json` с пустым
набором environment overrides; прежняя запись там — `f4de4e4`. После любого
отката, влияющего на вычисления или defaults, повторить штатные тесты,
`full2` и `check_rules` на SHA отката.
