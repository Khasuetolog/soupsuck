# S-017 — независимое ревью solution перед финальными замерами

Дата: 2026-10-30. Исполнитель: Codex-3. Проверенный снимок: **de836e0** (HEAD при старте).
Во время чтения HEAD обновился до d228f99 (S-015): изменения `nf4_patch.py`,
`kernels/nf4.py`, `tests/test_nf4.py` и добавленный экспорт в `kernels/__init__.py`
**не входят** в это ревью, согласно явному исключению задания. Остальной solution
между этими коммитами не изменился (`git diff --stat de836e0 d228f99`).

Изменён только этот файл; коммитов, Kaggle-запусков и исправлений реализации нет.
Локальный `python3` не имеет torch (`find_spec('torch') == None`), поэтому выводы
о новых CUDA-путях ниже статические, а не результаты GPU-тестирования.
Прочитаны README kit, `harness/check_rules.py`, `_train.py`, `common.py`, PLAN,
STATUS, RUNTIME_AUDIT, все Python-файлы solution в проверенном снимке, доступные
result/metrics/profile и pytest-логи предыдущих T4-прогонов.

`site/` ниже обозначает реально прочитанные pinned-исходники в
`/private/tmp/claude-501/-Users-arystan-orca-workspaces-decentrathon-6-0-featherduster/c29b337b-6a21-4aad-b075-87130cee4abf/scratchpad/src/site/`.
Пакетов bitsandbytes и torch в этом каталоге нет; их реализацию этим чтением
независимо не подтверждали. Сведения о bnb из предыдущего RUNTIME_AUDIT отмечены
как зависимость, а не новое измерение.

## Приоритизированные находки

### P0-1 — check 4 пока не подтверждён для сдаваемого снимка; последние измеренные версии ниже 5%

- **Якоря:** `solution/patch.py:229` (переключение на собственный MLP),
  `solution/kernels/swiglu_fused.py:21`; `harness/check_rules.py:357` (`c4`).
- **Доказательство:** последний скачанный
  `runs_kaggle/20261030T104354956814Z_solution_0e12e01_e4e1167_fp16norm/out/runs_try/solution_1/metrics.json`:
  `timed_loop_s=44.787`, сумма времени собственных ядер без `example_*` =
  **1771.724 ms**, доля **3.9558889856%**. `result.json` того же каталога:
  wall 76.64 s, peak 5843 MiB, eval 2.04966. Предыдущие доли: 3.3493%
  (25dc62d), 3.6264% (0e12e01 default). Единственный скачанный full checks-файл
  `runs_kaggle/20261030T102700096213Z_full_9bf579d/out/runs/checks.json`
  прямо содержит FAIL 4 (1.2%), остальные девять checks PASS.
- **Граница вывода:** эти прогоны использовали пример SwiGLU. de836e0 меняет его
  на собственный MLP; старые числа **не доказывают**, что текущий код провалит
  check 4. Но ни одного скачанного результата с новым `swiglu_tile_engine`
  среди просмотренных артефактов нет. Это блокер доказательства допуска,
  а не доказанный дефект производительности нового MLP.
- **Минимальное действие:** завершить полный T4-прогон неизменённого сдаваемого
  снимка без экспериментальных env overrides; проверить собственные launches,
  минимум доли по двум solution runs и все 10 checks. Нельзя прибавлять время
  примера или cuBLAS к времени собственных ядер. Желателен запас выше 6%:
  сам harness добавляет note при доле от 5 до 6% из-за скрытых строк.
- **Влияние:** без PASS 4 нет допуска, баллы скорости/памяти недоступны;
  влияние нового MLP на loss/wall/peak пока UNVERIFIED. Старое снижение памяти
  не заменяет проверки нового снимка.

### P1-1 — all-ignored batch может вернуть NaN dX после уже выполненной маскировки

- **Якоря:** `solution/autograd.py:16` (нулевой denominator),
  `solution/kernels/cross_entropy.py:162`, `:108`, `:175`.
- **Суть и статическое доказательство:** для всех `targets=-100` loss sum=0,
  denominator=0; training loss остаётся NaN, как при baseline mean CE.
  Градиент деления по каждому token loss при этом `+inf`.
  `gain=abs(grad)/1024` становится inf. `vocabulary_derivative_kernel`
  маскирует ignored строки в **0**, но заключительное `dx.mul_(gain)`
  превращает их обратно в NaN (`0*inf`). Та же проблема воспроизводима напрямую
  без деления на ноль, если дать ignored строке бесконечный upstream.
  Контракт ignored CE — нулевой градиент; это также явно обещано в docstring
  `cross_entropy.py` и проверяется для конечного upstream в существующих тестах.
  Pinned `site/transformers/loss/loss_utils.py:32`–`:45` подтверждает baseline
  sum/mean и деление; `site/soup_cli/trainer/rewind_hf.py:209`–`:215`
  документирует нулевые ignored per-token losses. Новый CUDA-repro не запускался.
- **Точное воспроизведение на T4:**

  ```python
  import torch
  from solution.autograd import fused_head
  from solution.kernels.cross_entropy import fused_linear_cross_entropy

  x = torch.randn(6, 32, device="cuda", requires_grad=True)
  w = torch.randn(1000, 32, device="cuda") / 32**0.5
  y = torch.full((6,), -100, device="cuda", dtype=torch.long)
  per_token = fused_linear_cross_entropy(x, w, y)[0]
  per_token.backward(torch.full_like(per_token, float("inf")))
  print("ignored dX finite/zero:", torch.isfinite(x.grad).all(), (x.grad == 0).all())

  x2 = x.detach().clone().reshape(2, 3, 32).requires_grad_()
  labels = y.reshape(2, 3)
  loss, step = fused_head(x2, w, labels)  # denominator=0
  loss.backward()
  print("loss/dX:", loss, torch.isfinite(x2.grad).all())

  # Сравнить с baseline CE при тех же ignored labels и той же нормировке.
  xr = x.detach().clone().requires_grad_()
  logits = torch.nn.functional.linear(xr.half(), w.half()).float()
  ref = torch.nn.functional.cross_entropy(logits, y, reduction="mean")
  ref.backward()
  print("baseline loss/dX:", ref, torch.isfinite(xr.grad).all(), (xr.grad == 0).all())
  ```

- **Минимальное исправление:** сделать `gain` конечным (например, 1) именно
  для ignored targets перед заключительным масштабированием; сохранить их
  `dZ=0`. Не заменять training loss на 0 и не менять denominator/task.
  Отдельно проверить полностью ignored batch и одну ignored строку среди
  непустых. Сейчас последняя при конечном denominator проблемой не является.
- **Влияние:** на обычный batch не влияет; при пустой supervision NaN dX
  может загрязнить градиенты адаптера/вызвать пропуски GradScaler и ухудшить
  checks 5–7. Цена исправления — O(N) маска/выбор, без `[N,V]` буфера.
  У baseline сам loss такого batch уже NaN; нельзя обещать, что этим
  исправлением полностью решается допуск при all-ignored batch.

### P1-2 — post-attention fp16 output не проверяет, может ли следующий MLP принять fp16

- **Якоря:** `solution/decoder.py:68`–`:82`; `solution/mlp.py:64`–`:65`, `:179`.
- **Доказательство:** decoder по умолчанию всегда отдаёт fp16 норму независимо
  от autocast, типа/precision MLP и training/eval. Собственный MLP правильно
  отказывается от fp32 `nn.Linear` без AMP, но fallback получает уже fp16 x.
  Pinned Qwen2MLP вызывает `gate_proj(x)` напрямую; исходный decoder подавал
  туда fp32 norm output. В
  `runs_kaggle/20261030T104354956814Z_solution_0e12e01_e4e1167_fp16norm/out/pytest_solution.log:195`–`:292`
  есть реальный fail `test_bridge_without_amp_preserves_dense_fallback`:
  `Half != float` в gate_proj после `decoder.py`. Старый стек содержал пример
  SwiGLU; de836e0 убрал пример, но отказ `_eligible` и вызов оригинального
  fp32 MLP воспроизводят ту же несовместимость. Это статическое подтверждение
  сохранения причины в текущем коде, а не новый GPU-прогон.
- **Воспроизведение:** на T4 с env по умолчанию:
  `python -m pytest solution/tests/test_integration.py::test_bridge_without_amp_preserves_dense_fallback -q`.
  Также проверить `apply(); Qwen2ForCausalLM(small_config).cuda().eval()` и
  обычный forward/generate без autocast. Голова в этих случаях уходит в
  fallback, decoder продолжает понижать precision.
- **Минимальное исправление:** разрешать fp16 output лишь при подтверждённом
  совместимом downstream (целевой frozen bnb fp16 MLP или подходящий AMP путь);
  иначе использовать `out_dtype=None`. Guard должен учитывать не только
  доступность CUDA, но и исходную арифметику MLP. Обычный FP32 Qwen без AMP
  должен работать после apply без ручного env override.
- **Влияние:** доказанный crash в поддерживаемом fallback; это не доказанный
  crash фиксированной NF4 T4-тренировки и не доказанный fail Soup 79 tests.
  Узкий guard сохраняет экономию памяти целевого пути; точный fp32 fallback
  требует больше активационной памяти там, где исходный путь был fp32.
  На loss/время целевого пути guard не должен влиять, проверить на T4.

### P2-1 — публичное описание и воспроизводимость ещё не отражают реализацию

- **Якоря:** `README.md:79`–`:86`, `:214`–`:221`; дополнительно
  `docs/README_DRAFT.md:3` (вне solution, прочитан как возможный текст для сдачи).
- **Доказательство:** root README остаётся правилами starter kit и описывает
  работающий пример; draft утверждает, что пример включён и GPU-измерений нет,
  перечисляет только CE/RMSNorm, хотя в de836e0 есть decoder/LoRA/own MLP.
  В root нет готового SUBMISSION.md и публичной ссылки Kaggle; есть только
  template. В tasks/STATUS есть более свежие измерения, чем в draft.
- **Минимальное исправление:** после финального допуска заполнить root README
  и SUBMISSION реальными ссылками/commit, описать default fp16 norm + store MLP,
  recompute как эксперимент, gradient gain как численное отличие, ограничения
  fallback и итоговые paired T4-результаты. Не переносить старые числа на новый
  commit и не выдавать только solution run за все 10 PASS.
- **Влияние:** до 15 баллов README/воспроизводимости; loss/wall/peak не меняет.

## Проверка пяти обязательных областей

### 1. Правила, импорты и параметры — нарушений в проверенном снимке не найдено

AST всех Python-файлов HEAD проверен против `harness/common.py:BANNED_MODULES`:
запрещённых импортов и непарсящихся файлов не найдено. Названия Liger/Apex/Cut CE
в docstrings — ссылки на идеи, не импорты. В runtime-коде нет `torch.compile`,
CUDA graphs, установки пакетов, фоновых процессов, смены GPU/allocator, чтения
данных/файлов за пределами repo, чтения seed, имени pytest или режима profiler.
Импортный PathFinder — загрузка установленных разрешённых пакетов, не чтение
скрытых строк. Test fixtures используют temp files/subprocess для теста import
hook, production path этого не делает.

Разрешённые developer switches: `SOUP_FUSED_DISABLE` default false,
`SOUP_FUSED_NORM_FP16` default true, `SOUP_MLP_RECOMPUTE` default false (store).
Они не определяются по измерению/тестированию. Сдача должна измерять эти defaults
и документировать overrides только как отдельные эксперименты.

Trainer args, optimizer, schedule, sampler/порядок данных и сохранение адаптера
не переписываются. `inputs['use_cache']=False` и удаление `_prediction_loss_only`
повторяют pinned TRL `sft_trainer.py:1202`, `:1210`. Значения существующих
trainable weights/LoRA settings не заменяются. MLP создаёт скалярные tensor
anchors с requires_grad, но **не nn.Parameter**, не регистрирует их на model и
не добавляет в optimizer/state_dict: это локальная autograd-техника, а не новые
обучаемые параметры адаптера. Hook guards для fake modules есть; cold apply
не импортирует torch/TRL. Check 10 для нового снимка требует реального запуска.

### 2. Численная эквивалентность — находки P1-1/P1-2; остальных доказанных дефектов нет

| Путь | Проверенный контракт и границы доказательства |
| --- | --- |
| Loss | `autograd.py` pad-right/shift по каждой строке, последний target ignored; сумма / переданный num_items либо число valid targets. Соответствует pinned `loss_utils.py:49`–`:70`. Не усредняет средние строк. Исключение all-ignored gradient — P1-1. |
| Entropy | Считается для всех B*S позиций, затем attention-mask mean; не assistant-only mask. Совпадает с TRL `:1238`–`:1257`. Argmax выбирает первый индекс при равных fp16-rounded logits, в том числе между chunks. |
| Accuracy/tokens | Accuracy использует labels[:,1:] и 0 при total=0; num_tokens — накопленная сумма attention_mask. Соответствует TRL `:1259`–`:1320`; masks намеренно разные. |
| Rewind | Свежий detached ChunkedStep, тот же `_soup_chunked_step` (pinned `stream_loss_chunk.py:298`); model state очищается в начале forward и после чтения, trainer state в начале compute_loss. Последняя loss-позиция каждой строки 0; row_tokens=0 даёт row_loss=0 в неизменённом helper. Повторного использования статистики прошлого batch не найдено. |
| AMP head | Проверка precision перенесена внутрь model forward; structural preflight её пропускает. Это соответствует Accelerate `accelerator.py:1818`–`:1829`. fp32 chunk GEMM округляется через fp16 перед CE/entropy/argmax, затем fp32 statistics. Accelerate возвращает baseline logits в fp32 (`operations.py:930`). |
| CE backward | Есть upstream GradScaler multiplier. Gain уменьшает fp16 underflow и потому **не гарантирует побитовое совпадение** с baseline fp16 dZ. Существующие тесты намеренно сравнивают с идеальным градиентом, не требуют равенства AMP autograd; это не новая ошибка, но конечный допуск checks 6–7 нужен на другом seed. |
| RMSNorm | variance/rsqrt fp32, normalized cast к input dtype до weight multiply. Backward учитывает cast. Add+RMSNorm сохраняет residual sum dtype, суммирует оба gradient edges. Optional fp16 output — отдельное precision-решение с дефектом guard P1-2. |
| LoRA S-011 | В целевой конфигурации input_layernorm остаётся fp32, fp16 post-attention output идёт в MLP, а не в q/v. При fp16 ingress patch восстанавливает fp32 **до dropout**, вызывает штатный dropout ровно один раз после base_layer и сохраняет alpha/r, dtype scale/add. Маски/Philox сравниваются имеющимися тестами. Округлённый x16.float() не восстанавливает исходные x32 values; на целевом default это не новый LoRA ingress. |
| MLP store/recompute S-012 | Default store сохраняет gate/up fp16 и local projection GradientEdges; recompute сохраняет x fp16 и повторяет gate/up под захваченным AMP. Frozen exact projection types, hooks/trainable projections исключены. Корректность fp16-сжатия для bnb зависит от реально fp16 projection arithmetic; `_eligible` проверяет compute_dtype/type_is_set. Для dense fp16 SiLU промежуточное округление отличается от fp32 fused engine, поэтому заявлять bitwise equivalence нельзя. |
| retain_graph | Два локальных `autograd.grad(..., retain_graph=True)` удерживаются ctx, не глобальным списком; сохраняют возможность повторного first-order backward. Глобального растущего кеша activation/ctx не найдено. Однако тест памяти только сравнивает store/recompute на 4 dense layers; длительный leak/lifetime на bnb 28 layers этим не доказан. Утечку без измерения не утверждаю. |
| Eval/generation | Head и bridge уходят в original при eval/no labels, не требуют side-channel; decoder/norm/LoRA/MLP остаются patched. Поэтому нельзя говорить, что весь eval/generation неизменён; P1-2 действует и там. Внешний `harness/eval_adapter.py` вообще не импортирует solution и проверяет сохранённый LoRA обычным PEFT. |

В latest downloaded pytest_solution.log есть ещё два падения dgamma с ошибкой
около 1.2–1.6e-5 и несовместимый предустановленный torchao. Текущий test_add_rmsnorm
уже использует magnitude-scaled bound; текущий integration PEFT fixture обходит
неподходящий optional quantizer через custom module mapping. Старые FAIL нельзя
переносить на HEAD; PASS новых tests также пока не доказан. Frozen norms целевой
тренировки вообще не выполняют dgamma kernels.

### 3. Скрытые строки, seed, формы — P1-1; других доказанных дефектов не найдено

Ни одна реализация не фиксирует B=4/S=512/N=2048 в indexing; формы берутся из
тензоров. Row boundaries loss сохраняются. Хвосты N, H/I, V маскируются;
имеются тесты N=77/1001, H=33/80/1000, V=7/4103 и реальные размеры CE.
Первый unready bnb MLP может использовать original до установки compute type;
после прогрева eligible path включается — это readiness guard, не ветвление
по номеру измеряемого шага.

Длинные исходные строки форматируются Soup с `data_cfg.max_length=512`
(`site/soup_cli/data/sft_format.py:63`, `:200`–`:203`). Сам solution не режет
их повторно, не читает скрытые данные и не миксует строки. Длинный prompt может
оставить мало/ноль assistant labels: отдельная пустая строка даёт нулевой row
loss/count, полностью пустой supervised batch требует P1-1. Даже при сохранении
baseline NaN loss этот край нужно проверить явно. Новый seed меняет dropout и
порядок строк; фактического сравнения de836e0 на другом seed нет.

### 4. Все @triton.jit, likeness, реальные launches и busy work

Локально без torch импортированы функции **самого** `harness/check_rules.py`,
AST получен через `git show de836e0:<file>`, затем вызван `example_likeness`.
Никакой source-код примера не переименовывался и harness не редактировался.

| Kernel | Файл:строка | Likeness | Запуск/использование |
| --- | --- | --- | --- |
| add_rmsnorm_forward_kernel | kernels/add_rmsnorm.py:38 | 0.3211 | residual+norm потребляет decoder; есть launches в последних T4 metrics |
| add_rmsnorm_backward_kernel | kernels/add_rmsnorm.py:55 | 0.3392 | dr/dd уходят в decoder backward; есть launches |
| vocabulary_statistics_kernel | kernels/cross_entropy.py:39 | 0.1686 | state/best потребляют outputs/accuracy; есть launches после S-009 |
| vocabulary_outputs_kernel | kernels/cross_entropy.py:80 | 0.4595 | loss/entropy/lse потребляют train/Rewind/backward; есть launches |
| vocabulary_derivative_kernel | kernels/cross_entropy.py:96 | 0.2854 | dz участвует в dX GEMM; есть launches |
| lora_scale_add_kernel | kernels/lora_epilogue.py:25 | 0.6920 | результат q/v forward; есть launches |
| lora_scale_gradient_kernel | kernels/lora_epilogue.py:38 | 0.5446 | gradient adapter branch; есть launches |
| rmsnorm_forward_kernel | kernels/rmsnorm.py:42 | 0.2966 | input/final norm; есть launches |
| rmsnorm_input_gradient_kernel | kernels/rmsnorm.py:55 | 0.4166 | dX norm; есть launches |
| rmsnorm_weight_partials_kernel | kernels/rmsnorm.py:73 | 0.3596 | только weight.requires_grad; не запускается при frozen norms |
| rmsnorm_weight_finalize_kernel | kernels/rmsnorm.py:92 | 0.3938 | только weight.requires_grad; не запускается при frozen norms |
| swiglu_tile_engine | kernels/swiglu_fused.py:21 | 0.3044 | act/dgate/dup потребляют frozen MLP; новый путь, скачанного T4 evidence нет |
| example_swiglu_fwd_kernel | swiglu.py:16 | 1.0000 | исключён из own, de836e0 не регистрирует |
| example_swiglu_bwd_kernel | swiglu.py:26 | 1.0000 | исключён из own, de836e0 не регистрирует |

Все пути в таблице относительны solution/. Все 12 собственных ядер ниже 0.8;
автоматических lookalike notes нет. Это не заменяет ручного решения жюри об
оригинальности. Готовых чужих реализаций по прочитанному коду не обнаружено.
Все запускаемые production kernels делают полезную работу; optional dgamma
не надо запускать искусственно ради checklist. Нет busy work.

Важно: `c3` **не требует запуска каждого** найденного ядра; оно требует
непустого пересечения запущенных own kernels во всех solution runs. Два optional
dgamma появятся в detail `not launched in every solution run`, это само по себе
не FAIL. В full 9bf579d ровно такая ситуация и c3 PASS. В c4 учитывается только
пересечение запущенных ядер и минимум доли по всем runs; проблема — P0-1.

### 5. Triton гонки, offsets, recompilation — дефектов в пределах задачи не найдено

Каждый norm/CE-row пишет свою строку; vocabulary chunks запускаются последовательно
в текущем CUDA stream и обновляют state без межпрограммных конфликтов.
Weight gradients имеют независимые partial tiles и второй reduction pass, без
atomic races. SwiGLU двумерные tiles, LoRA плоские blocks пишут разные элементы.
Все tail loads/stores ограничены масками; p*(-inf) в статистике явно избегается.

Полного `[N,V]` indexing нет. При N<=2048 chunk width<=4096 offsets Z/dZ меньше
8,388,608 элементов; MLP N*8960=18,350,080; norm N*1536=3,145,728. Это далеко
ниже signed int32 limit. Даже N*V=311,164,928 элементов не переполняет int32,
но такой full buffer solution не создаёт. Для произвольно огромных публичных
API-входов N/strides int64 guards отсутствуют; это вне фиксированного task,
не объявляется P1 на скрытых строках с тем же max_length.

H/EPS/BLOCK и forward/backward flags — constexpr; N/row strides — runtime.
Это предотвращает отдельную специализацию на каждую длину sequence, но **не
гарантирует один binary**: Triton может специализировать runtime ints по alignment
и значениям 1; FIRST, dtype и vocab tail создают дополнительные варианты.
CE `_chunk_size` меняется при пересечении memory budgets, для фиксированного
N<=2048,H=1536 остаётся 4096. В kernels нет autotune/compile warmup/готовых
компиляторных кешей. Первые CUDA launches компилируются в измеряемой команде;
SwiGLU MLP tests проверяют <=4 variants на пяти длинах, norm <=2. Для нового
HEAD cold wall/variant-count на T4 ещё UNVERIFIED.

## Предложенные проверки на T4 (здесь НЕ запускались)

1. Исправить/подтвердить P1-1 и P1-2 точными repro выше. Выполнить `python -m pytest
   solution -q` и `SOUP_APPLY_SOLUTION=1 python -m pytest tests -q` на pinned runtime;
   отдельно сохранить counts/skips, а не только exit training.
2. На самом сдаваемом commit, defaults без SOUP overrides:
   `python harness/run_task.py --repeats 2 --out runs_final`, затем
   `python harness/check_rules.py --result runs_final/result.json`.
   Сохранить обе solution metrics и все 10 checks, config/shapes адаптера,
   per-step loss/entropy/accuracy/num_tokens и row_tokens/row_loss Rewind.
3. Повторить paired сравнение на другом разрешённом seed и других доступных
   публичных строках, без чтения скрытых строк. Изменение seed/data для
   экспериментального сравнения делать одинаково обеим сторонам через штатный
   harness CLI, сверив его help; это не утверждение о финальном task config.
4. Для MLP сравнить store/recompute с baseline при N=4*S, S=1/17/77/257/512,
   I=8960 и хвостовом I; несколько seed, upstream с GradScaler. В долгом
   64-step bnb прогоне сохранить allocated memory после backward и удаления
   выходов, чтобы отличить удержанный текущий ctx от роста между шагами.
   Сравнить качество/память/время отдельно для каждого экспериментального режима.
5. Решение о допуске принимать по свежему full result/checks; прежние profile.json
   используют другой MLP, а profiler run останавливается на 24 шагах и не заменяет
   измеряемый 64-step run. Дополнительный NF4 commit требует отдельного ревью
   и полного измерения, этот отчёт его не одобряет.


## S-021 — ревью NF4 S-015 и fused CE S-018

Дата: 2026-10-30. Проверены **фиксированные коммиты** `d228f99` (NF4 kernel,
installer и tests) и `8f8ebbc` (CE и tests). Незакоммиченные decoder/rmsnorm/lora/
patch изменения S-016 не читались для этого ревью и не одобряются им.
Изменён только этот отчёт, без коммита и без Kaggle-запуска.

**Итог:** новых доказанных P0/P1 в этих двух изменениях не найдено.
S-017 P1-1 закрыт структурой CE backward на уровне кода; GPU-проверка новых
тестов и paired качества/производительности остаётся UNVERIFIED. Есть три P2
ниже. Правильность NF4 подтверждена сверкой формата и call sites, а не новым
утверждением о побитовом GPU-результате.

### Независимые источники и воспроизводимость чтения

Скачан **вне репозитория**, без установки, официальный Linux wheel:
[bitsandbytes 0.50.2, PyPI](https://pypi.org/project/bitsandbytes/0.50.2/#files),
`bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl`.
SHA256 файла сверен с JSON PyPI:
`55348a9a4a21bfd99cf8c7b32fe67b4030ae5c2a05738e03c1747f65fa6ec283`.
Извлечены только Python-исходники в `/private/tmp/s021-bnb-0.50.2/`:
`bitsandbytes/functional.py`, `backends/cuda/ops.py`, `autograd/_functions.py`,
`nn/modules.py`. Далее **bnb/** обозначает этот каталог bitsandbytes.

Дополнительно прочитан [csrc/kernels.cu тега 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/kernels.cu),
сохранённый в `/private/tmp/s021-bnb-kernels.cu`, SHA256
`a88d7a522b21b3d6e50cacad890c0c3ac44ad17b5f205c4b34ae91995ba1e5d6`.
Номера строк ниже взяты из скачанных файлов, а solution — из `git show <commit>:<path>`.
Python urllib первоначально не прошёл локальную проверку сертификата; штатный
curl успешно скачал файлы с обычной TLS-проверкой, без отключения верификации.
Torch/CUDA локально не исполнялись.

### P2-021-1 — functional NF4 с out= выделяет лишний полный выход и копирует его

- **Файл:строка:** `solution/nf4_patch.py:63`–`:67` в d228f99;
  `solution/kernels/nf4.py:124`.
- **Доказательство:** даже если caller передал подходящий `out`, wrapper сначала
  вызывает `dequantize_nf4`, который делает `torch.empty(shape)` и декодирует туда,
  затем выполняет `out.copy_(...)`. Оригинальный
  `bnb/functional.py:1058`–`:1061` вызывает `.out` dispatcher, который пишет прямо
  в предоставленный буфер (`bnb/backends/cuda/ops.py:437`–`:451`).
  При shape=(8960,1536) лишний output — **27,525,120 bytes = 26.25 MiB** плюс
  полный copy. Это расчёт по dtype/shape, не измеренный пик.
- **Как проверить на T4:** сравнить peak allocation и output storage при
  native/patched `functional.dequantize_4bit(packed, state, out=preallocated)`;
  output должен содержать те же fp16 bits, но patched путь сейчас имеет второй
  shape-sized allocation. Существующий integration NF4 test проверяет alias и
  значения out, но не allocation.
- **Минимальное исправление:** дать nested decoder поддерживаемый destination
  buffer и писать прямо в out; сохранить исходную transpose-view семантику.
- **Влияние:** численные значения не меняет; лишние память и bandwidth только
  у functional out= пути. Целевой MatMul4Bit.backward не передаёт out, а backend
  wrapper уже пишет в out напрямую: регресс именно этого целевого train path
  данной находкой не доказан.

### P2-021-2 — non-nested functional fallback всё равно может запускать собственный flat NF4

- **Файл:строка:** `solution/nf4_patch.py:55`–`:58`, `:75`–`:84`;
  `solution/tests/test_nf4.py:143`–`:145` (FP4 routing), `:155`–`:159` (direct state rejection).
- **Доказательство:** `_validate_state` отклоняет nested=False, wrapper вызывает
  saved `original_f`, но тот через `.default`/`.out` dispatcher вызывает **уже
  заменённый** backend global `_dequantize_4bit_impl` (bnb functional:1058–1070,
  backend:423–451). При NF4/block64/fp16 с обычными fp32 absmax `_validate_flat`
  проходит, и собственный kernel запускается с `NESTED=False`.
  Для FP4 оба wrapper действительно возвращаются в original; этот случай
  покрыт test. Для non-nested NF4 аналогичного routing test нет.
- **Оценка:** это не обнаруженное изменение значений — flat decoder как раз
  реализует ту же формулу и не знает, были scales nested раньше или нет.
  Но утверждение «не-nested вообще остаётся на прежней реализации» неверно,
  и fallback на functional-уровне сам по себе не доказывает отсутствие launches.
- **Как проверить на T4:** расширить LaunchSpy из
  `test_patch_covers_functional_dispatcher_forward_and_backward`: quantize NF4
  с `compress_statistics=False`, сохранить native bits, установить patch,
  вызвать functional и сравнить bits/список NESTED launches. Ожидаемый по коду
  launch — `[False]`, а не пустой список. Здесь не запускалось.
- **Минимальное решение:** явно документировать/test побитовую совместимость
  non-nested flat NF4 как поддерживаемого backend случая. Если требуется строго
  не трогать этот путь, нужен scoped bypass backend wrapper при таком functional
  fallback; не отключать глобально backend hook для всего обучения.
- **Влияние:** потенциальная разница latency/kernel share вне nested task;
  численного дефекта для корректного non-nested NF4 не найдено. Нельзя назвать
  это P1 для текущей double-quant задачи без дополнительных доказательств.

### P2-021-3 — новый bitwise CE test доказывает совпадение одинаковых chunk GEMM, не полной головы

- **Файл:строка:** `solution/tests/test_cross_entropy.py:162`–`:216` в 8f8ebbc;
  fixture `:39`–`:48` и reference `:69`–`:73`.
- **Доказательство:** новый test сравнивает torch.mm с F.linear на **том же**
  chunk shape и затем использует те же Triton online-statistics kernels для
  expected/actual. Это полезная проверка GEMM dtype/dispatch, но не независимый
  oracle всех статистик. Его shape N=77,V=4103 также не является полной
  тренировочной головой V=151936. Большинство остальных correctness tests
  fixture запускает с reduced-precision reduction=False, а parametrized test
  отдельно включает True/False только для этого chunk сравнения.
- **Минимальное действие:** дополнить проверкой loss/entropy/argmax против
  независимого full-vocabulary F.linear+CE/softmax при штатных backend flags
  и representative rows, затем paired checks 5–7 на T4. Не менять flags
  настоящего baseline или допуски ради прохождения теста.
- **Влияние:** production-код не меняется; это граница доказательства, не
  обнаруженный numerical fail. Выбор алгоритма cuBLAS зависит от форм, поэтому
  переход к native fp16 output сам по себе не обещает bitwise равенства
  чанковой и полной головы. Конечные потери и quality всё ещё проверяет harness.

### NF4: проверенные свойства и остаточные границы

| Требование | Результат и доказательство |
| --- | --- |
| Упаковка | High nibble даёт более ранний элемент, low nibble — следующий. Совпадает с quantize/dequantize ветками официального csrc и `nf4_decode_kernel:42`–`:44`. Локально проверены все 256 пар кодов байта. |
| Таблица NF4 | AST `_NF4_VALUES` сравнен с `nf4_dequantization_lut` скачанного тега: **16/16 float32 bit patterns совпадают**, через `struct.pack('f')`, без GPU. |
| Nested scales | `code2[qabs[block]] * absmax2[block//256]`, затем offset, всё fp32; `enable_fp_fusion=False` на обоих launch sites. Это согласуется с отдельным fp32 blockwise output, затем offset-add в `bnb/functional.py:1052`–`:1056`. Flat backend получает уже восстановленные scales. |
| Transpose | Decoder создаёт state.shape, затем возвращает `.t()` только при packed.shape[0]==1; original functional имеет тот же BC shim `:1072`–`:1077`. В forward/backward MatMul4Bit packed B нормализуется в canonical [-1,1], поэтому обычный training output не транспонируется повторно. out= сохраняет view/alias, но имеет P2-021-1. |
| Хвосты | `index<N` маскирует Packed/Abs/Scale2/Code2/Table и store; последний нечётный элемент использует high nibble существующего последнего байта. `_validate_state` требует packed bytes >= ceil(N/2), absmax >= ceil(N/64), state2 absmax >= ceil(blocks/256). Не обнаружено OOB для принятых состояний. |
| Offsets | Для максимального из заданных весов 8960×1536: N=13,762,560, packed bytes=6,881,280, scales=215,040, second-level scales=840. Все element/byte offsets далеко ниже int32 limit, без произведения batch×vocabulary. Для произвольно огромного стороннего API-веса >2³¹ guards не предусмотрены; это не размер фиксированного task. |
| Functional hook | `MatMul4Bit.backward` читает `F.dequantize_4bit` при каждом вызове: **bnb/autograd/_functions.py:384**. F — модуль functional, не захваченная отдельная функция. Hook действительно покрывает этот backward. |
| Backend hook | `_dequant_linear_fallback` вызывает global `_dequantize_4bit_impl`: `bnb/backends/cuda/ops.py:905`–`:916`; оба registered dequantize operators также используют global. Замена helper покрывает их без перерегистрации torch operators. |
| Какие forward не покрыты | Custom fused GEMM/GEMV не обязаны вызывать decoder. Для CUDA M>1536 backend безусловно выбирает fallback (`ops.py:924`, `:953`–`:954`); более короткие batch shapes зависят от heuristic/выравнивания. Поэтому «NF4 patch ускоряет любой forward» было бы неверно. |
| Unsupported states | Functional проверяет quant_type/nested/blocksize/dtypes/state2/buffers; backend проверяет quant_type/blocksize/output dtype и buffer layout. Version gate точный 0.50.2, CPU/disabled возвращаются в original. FP4 остаётся original в обоих местах; non-nested routing — P2-021-2. Ошибки kernel/compiler не скрываются широким except. |
| Реальная регистрация | Installer `patch_bnb_dequant` идемпотентный и ставит обе точки. В проверенном committed `solution/patch.py` **нет его вызова**. До явной интеграции kernel не участвует в default training; наличие installer не является evidence launches. Незакоммиченный S-016 в эту оценку не входит. |
| Оригинальность/правила | NF4 likeness по актуальному harness=**0.4150**, ниже 0.8. Запрещённых импортов, измерительных ветвлений, package install и готовых compiled caches в этих файлах нет; LUT — данные формата, не чужая реализация. Per-device LUT cache фиксированного размера и не хранит веса/hidden rows. |

NF4 GPU tests корректно снимают **обе** обёртки при построении native oracle;
используют int16 view, чтобы различать signed zero; проверяют реальные размеры,
нечётные хвосты, transpose, FP4 fallback, functional out=, dispatcher, forced
M=1537 fallback и backward с LaunchSpy. Это содержательные acceptance tests,
но их наличие не заменяет лог выполнения на T4. Новых замеров NF4 этим ревью нет.

### CE: dtype, compaction, P1-1 и память

- **Логиты:** `cross_entropy.py:149`, `:195` теперь вызывают fp16-input
  `torch.mm` **без** fp32 out_dtype; результат fp16, затем kernels загружают
  его в fp32. Это та же категория операции, что bias-free F.linear над
  этими operands. Production-код не меняет reduced-precision reduction flags;
  FP32 выход используется только для dX GEMM `:202`. Полного bitwise совпадения
  с полной головой другой формы не утверждаю, см. P2-021-3.
- **Online statistics:** fp16 load и `other=-inf` корректны: half допускает -inf;
  после promote masked p=0, а weighted term умножает p на `where(col<Width,z,0)`.
  Поэтому NaN от `0*(-inf)` на замаскированном хвосте не возникает. FIRST/последующие chunks,
  target_z, tie argmax и output mask остались согласованными.
- **Compaction:** `nonzero(targets != ignore_index)` даёт уникальные возрастающие
  row indices; index_select одинаков для hidden/targets/lse/upstream. При CUDA
  nonzero требует host-visible dynamic result и является sync; `.numel()` уже
  не дополнительный value `.item()` sync. Он вызывается один раз до vocab loop,
  в том числе для all-valid batch. All-valid skips gathers/scatter, но не
  nonzero; partial-valid scatter пишет ровно выбранные строки, остальные нули.
- **Пустые/полные наборы:** n=0 возвращает пустой zero gradient; valid_n=0
  возвращает zero gradient без gain и без backward GEMMs; valid_n=n использует
  исходные tensors. В partial случае original labels/entropy/argmax в forward
  не компактизируются: assistant/padding mask не удаляет нужные TRL statistics.
- **P1-1 закрыт в коде:** `:175`–`:178` возвращает нули до любого nonfinite
  вычисления, а `:183`–`:186` исключает ignored upstream из gain. Бывший
  `0*inf` после ignored dz невозможен для этих строк. Валидные строки с
  собственным nonfinite upstream по-прежнему могут давать nonfinite dX, как
  ожидается от такого входа; они не маскируются без основания. Scalar loss
  denominator не менялся: all-ignored mean loss остаётся baseline NaN.
- **Dtype dX:** accumulator fp32; перед return/scatter `:206` приводит к
  ctx.hidden_dtype, а full zero output `:208` выделяется сразу в том же dtype.
  fp16 ingress и исходный fp32 hidden поэтому получают соответствующий dX.
  Higher-order backward по-прежнему явно не поддерживается.
- **Сохранения/гонки:** сохранены hidden fp16, frozen weight reference, targets,
  lse; chunk z/dz/w/contribution не сохраняются между вызовами. Distinct
  nonzero indices исключают collision scatter. Чтение/запись chunks и kernels
  идут в текущем stream, state между ними остаётся упорядоченным.

**Память, статический расчёт для N=2048,H=1536,chunk=4096** (не замер):

| Одновременно живые временные тензоры | Объём |
| --- | ---: |
| w cast fp16 | 12 MiB |
| z fp16, каждая chunk аллокация | 16 MiB |
| dz fp16 | 16 MiB |
| dx accumulator fp32 | 12 MiB |
| Наибольшая сумма до `del z`: w+z+dz+dx | **56 MiB** |
| После `del z`: w+dz+contribution(fp32)+dx | **52 MiB** |

Это согласуется с docstring, если не включать inputs/outputs/cuBLAS workspace.
При partial compaction добавляется fp16 work_hidden (менее 6 MiB для valid_n<N)
и O(N) row vectors; old full saved hidden остаётся входом backward. При
scatter full_dx выделяется **после** удаления chunk temporaries. Эти extra
buffers явно названы в docstring, поэтому 56 MiB не надо выдавать за полный
peak всего backward. Если исходный hidden fp32, его forward cast отдельно
занимает 6 MiB. `_chunk_size` ограничивает z ≤64 MiB и w cast ≤32 MiB,
для task даёт chunk4096. Buffer-size нарушения/нового полного [N,V] allocation
не найдено; nvidia-smi/PyTorch peak и cuBLAS workspace всё равно требуют T4.

Compaction меняет M dX/logit GEMM и может менять cuBLAS algorithm, а nonzero
добавляет sync. Ожидаемое сокращение GEMM работы — не доказанный speedup;
сравнение wall/quality нового коммита со старым требуется отдельно.

Существующие новые tests покрывают full/some/all ignored, zero denominator
трёх видов, ignored inf/NaN/огромный upstream, один ignored inf среди valid,
порядок/размер GEMM operands и обе настройки reduced precision. Kernel likeness
в 8f8ebbc: statistics **0.1704**, outputs **0.4595**, derivative **0.2915**;
автоматического сходства с примером выше порога нет.

### Следующие точные проверки (предложены, здесь не запускались)

```bash
python -m pytest solution/tests/test_nf4.py -q -s
python -m pytest solution/tests/test_cross_entropy.py -q -s
python -m pytest solution -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2 --out runs_s021
python harness/check_rules.py --result runs_s021/result.json
```

NF4 pytest устанавливает patch вручную и проверяет охват, но default training
без installer регистрации NF4 не измеряет. Для опыта с включённым NF4 сначала
нужна принятая отдельная интеграция; не запускать hidden patches через notebook,
которые отсутствуют в сдаваемом solution. Дополнить non-nested routing и out=
allocation проверки, независимый default-flags full-vocabulary CE oracle,
затем сохранить launches/own times и paired metrics нового снимка.

Обновление прежнего P0-1 S-017: для **de836e0** уже получен отдельный full ×1
`runs_kaggle/20261030T105526661826Z_full_de836e0/out/runs/checks.json` с 10/10 PASS
и own share 8.6%; это закрывает прежнее отсутствие evidence именно для той пары.
Ни NF4 d228f99, ни CE 8f8ebbc не наследуют это измерение автоматически.
