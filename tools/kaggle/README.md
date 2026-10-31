# Прогоны Soup на Kaggle T4

Runner использует Python stdlib и Kaggle CLI 2.2.5. Команды и metadata сверены с
его `docs/kernels.md`, `kernels_metadata.md`, `datasets.md`, `datasets_metadata.md`
и `configuration.md`. Реальный upload и GPU-прогон пока **UNVERIFIED**.

Положите выданный Kaggle файл авторизации в `~/.kaggle/kaggle.json`, вне
репозитория, с правами `chmod 600 ~/.kaggle/kaggle.json`. Runner не открывает этот
файл: авторизацией занимается CLI. Username берётся из `KAGGLE_USERNAME` или
захваченного `kaggle config view`; полный config не выводится и не сохраняется.
Ключи не нужно передавать аргументами команд. Аккаунт должен иметь разрешение
на Internet и GPU T4 на Kaggle.

Из корня checkout:

```bash
export KAGGLE_BIN=<scratch>
# Необязательно, если username уже есть в конфигурации CLI:
export KAGGLE_USERNAME=your_kaggle_username

python3 tools/kaggle/run.py run --mode unit --dry-run
python3 tools/kaggle/run.py push-code
python3 tools/kaggle/run.py run --mode unit
python3 tools/kaggle/run.py run --mode solution
python3 tools/kaggle/run.py run --mode full
python3 tools/kaggle/run.py run --mode full2
# Отдельный эксперимент, не настройки финального прогона жюри:
python3 tools/kaggle/run.py run --mode solution --env SOUP_FUSED_NORM_FP16=1 --tag norm16
```

По умолчанию `KAGGLE_BIN=kaggle`. Dry-run никогда не вызывает CLI, не требует
токена и оставляет собранный notebook/metadata во временном каталоге; без
username использует `dry-run-user`. Для запуска нужен настоящий username.

`--env SOUP_KEY=VALUE` можно повторять; при повторе ключа берётся последнее
значение. Разрешены только `SOUP_`-ключи, кроме `SOUP_APPLY_SOLUTION`: этот ключ
остаётся под управлением шага pytest. Значения передаются как строки без shell
интерпретации и устанавливаются в notebook до запуска дочерних процессов;
`harness.common.ENV` наследует их. Флаги и tag видны в notebook, `experiment.json`,
`launch.json`, `out/status.json` и сводке; dry-run показывает их без запуска CLI.
Набор флагов получает стабильный короткий хеш, независимый от порядка аргументов,
в slug и имени каталога. Без флагов/тега прежние имена сохраняются.
`--tag` принимает до 32 ASCII-букв, цифр, `_` или `-`; переводится в нижний регистр,
`_` заменяется на `-`. Полный tag сохраняется в каталоге и metadata эксперимента,
а slug содержит первые 10 символов, чтобы ограничить общую длину имени ядра.

`push-code` отправляет **только закоммиченный HEAD** через `git archive` и печатает
полный SHA. При грязном checkout выводится предупреждение; незакоммиченные
изменения не попадут в dataset. Поэтому координатор должен сначала принять и
закоммитить изменения, затем повторить `push-code`. Загружается приватный
dataset `<username>/soup-kernel-code`: первый вызов создаёт его без `--public`,
последующие создают версии; существующий dataset принимается только при
подтверждённом `isPrivate=true`. Неподтверждённая приватность останавливает upload.
ZIP проверяется на пути авторизации `.kaggle`, `kaggle.json`, `auth.json` и
символические ссылки; домашний `~/.kaggle` вне git archive. Manifest содержит SHA
коммита, SHA256 ZIP и всех файлов. Проверьте готовность версии dataset через
`kaggle datasets status <username>/soup-kernel-code` перед запуском notebook.

Каждый `run` создаёт отдельный **приватный** notebook с уникальным slug, Internet
и `NvidiaTeslaT4` (T4 x2). Встроенная проверка требует `T4` в выводе `nvidia-smi`,
как шаблон жюри. Pip-аргументы совпадают с `templates/kaggle_notebook.ipynb`:
`soup-cli[train]==0.77.0`, `transformers==5.16.1`, `peft==0.20.0`, `trl==0.29.1`,
`bitsandbytes==0.50.2`, `accelerate==1.14.0`, `pytest`. Torch и Triton отдельно не
переустанавливаются. Dataset копируется в `/kaggle/working/repo`; поддерживаются
ZIP и автоматическая распаковка Kaggle. При несовпадении SHA/хешей обучение не
начнётся; это также защищает от запуска старой версии dataset после нового HEAD.

В режимах обучения сначала выполняются `task/prepare_data.py` и
`python -m pytest solution -q -rs`. Затем:

| Режим | Команды |
| --- | --- |
| `unit` | `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q` |
| `solution` | `python harness/run_task.py --only solution --out runs_try --profile` |
| `full` | `python harness/run_task.py`, `python harness/check_rules.py`, `python harness/score.py runs/result.json` |
| `full2` | те же команды, но `run_task.py --repeats 2` |
| `probe` | import-time диагностика apply/Soup и отдельные cold/warm вызовы ядер; без подготовки данных и обучения |

Для диагностики старта после принятия/коммита изменений и `push-code`:

```bash
python3 tools/kaggle/run.py run --mode probe --tag startup
```

`probe_import_apply.log` и `probe_import_baseline.log` содержат вывод
`python -X importtime`: оба процесса импортируют torch, инициализируют CUDA и
импортируют transformers, затем выполняют одинаковые импорты Soup/TRL/Qwen до
построения модели. Первый процесс перед этим вызывает `solution.apply()` и
печатает `apply_s`, оба печатают `soup_pre_model_s`. Поэтому виден и перенос
стоимости импорта между этапами; уменьшение apply_s само по себе не доказывает
уменьшение полного wall time.

[`probe.py`](probe.py) запускается отдельным процессом и пишет
`probe_compile.log` и `probe_compile.json`: перехватывает вызовы Triton compiler
API с CPU-временем компиляции каждого имени/варианта; отдельно измеряет cold и
warm wall time вызовов, включая GEMM/launch/backward. Формы: N=2048, H=1536,
V=151936, MLP I=8960. Пробуются CE, RMSNorm, add+RMSNorm, tiled SwiGLU, исходный
SwiGLU-пример и LoRA-эпилог, включая fp16/fp32 и optional norm-weight gradients.
Сумма compile time включает эти дополнительные варианты и не равна напрямую
стоимости компиляции конкретного training run. JSON хранит разбиение по case.
Используется новый временный `TRITON_CACHE_DIR`, удаляемый при выходе; этот кэш
и compiler instrumentation не передаются в harness. Прогон последовательный.
Ошибки сохраняются в JSON/status и возвращают ненулевой код.

Все команды исполняются последовательно, дополнительная GPU-работа не
запускается. Коды возврата сохраняются в `/kaggle/working/out/status.json`;
pytest/harness failures не мешают сохранению логов и сбору диагностики. Ошибки
установки, подготовки данных или SHA прерывают зависимые команды с записью
причины. Kernel может завершиться как COMPLETE даже при failed checks:
локальный runner проверяет `status.json` и возвращает ненулевой exit code.

Runner опрашивает status каждые 30 секунд до завершения/ошибки, максимум
90 минут, затем вызывает `kernels output`, в том числе после ошибки или таймаута.
Он не перезапускает kernel при сетевой ошибке. Результаты сохраняются в
`runs_kaggle/<UTC-time>_<mode>_<sha7>/` (исключён из git): submitted notebook,
metadata, launch/status и скачанный `out/` с логами, `result.json`, `checks.json`,
`metrics.json`, `profile.json`, eval JSON и хвостами `train.log`. Полные
train-логи также сохраняются. После сбора notebook удаляет рабочую копию repo,
данные и веса адаптеров, чтобы они не раздували экспорт; evidence остаётся в out.

На экране — wall time, GPU peak, eval loss и ratios из result, PASS/FAIL каждого
check, `own_kernel_ms` без `example_*` и доля суммы этих ядер от
`1000*timed_loop_s`, top-15 ядер каждого `profile.json` по GPU-времени
(миллисекунды, доля GPU-времени, число вызовов), хвосты pytest и notebook status.
Авторитетная проверка засчитываемой доли собственных
ядер — check 4, поскольку он дополнительно исключает сходство с примером.
Режим `solution` не запускает checks/score: сравнение с base требует `full`.

При прерванном локальном ожидании сведения для ручного продолжения находятся
в `runs_kaggle/.../launch.json`; повторять `run` для сбора старого результата не
нужно — он создаст новый запуск. Используйте команды CLI с kernel ID из launch:

```bash
kaggle kernels status username/kernel-slug
kaggle kernels output username/kernel-slug -p runs_kaggle/recovered
```

Этот notebook предназначен для итераций; публичная финальная сдача по шаблону
жюри и доступ к командному GitHub остаются отдельными шагами.
