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
`<scratch>`.
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
Извлечены только Python-исходники в `<tmp>/s021-bnb-0.50.2/`:
`bitsandbytes/functional.py`, `backends/cuda/ops.py`, `autograd/_functions.py`,
`nn/modules.py`. Далее **bnb/** обозначает этот каталог bitsandbytes.

Дополнительно прочитан [csrc/kernels.cu тега 0.50.2](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/csrc/kernels.cu),
сохранённый в `<tmp>/s021-bnb-kernels.cu`, SHA256
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


## S-023 — S-016: fp16 нормы, readiness guards и NF4 registration

Дата: 2026-10-30. Проверен только коммит **3c560d1ff1a468c7dfb8cea81435f5b6c0894ca1**:
`solution/patch.py`, `decoder.py`, `lora.py`, `kernels/rmsnorm.py` и изменённые
integration/RMSNorm tests. Неотслеживаемые или последующие файлы не ревьюились.
Изменён только этот отчёт; коммитов и Kaggle-запусков нет.

**Итог:** P1-2 (dense fp32 MLP без AMP) логически закрыт. Scope reset, исходный
dtype для dgamma и последовательность NF4 post-import callbacks согласованы.
Нового доказанного P0 нет. Есть **P1 численного риска input-norm compression**:
одинаковые dropout masks не дают одинаковых значений LoRA и не доказывают
эквивалентности backward накопления. Это не утверждение о проваленном check —
для нового default необходим paired T4 результат. Условный runtime отказ LoRA
разобран отдельно ниже: обычного пути от alignment к такому отказу не найдено.

### P1-023-1 — fp16 input norm не является точным переносом cast через LoRA/dropout

- **Файл:строка:** `solution/patch.py:64`–`:83`, `solution/lora.py:97`,
  `:119`–`:123`; `solution/decoder.py:64`–`:68`.
- **Доказательство forward:** новый default разрешает fp16 input norm для q/v
  PEFT bnb LoRA. В baseline LoRA dropout потребляет исходный fp32 norm output,
  **до** autocast A projection. Патч использует `x16.float()` перед тем же
  dropout. Условия readiness доказывают допустимость этой ветки и RNG-путь,
  но не равенство её значений: раннее half округление не коммутирует с
  умножением dropout на `1/(1-p)`.
- **Независимый локальный численный пример:** сохранённый элемент dropout,
  p=0.05, x32=**0.10006000101566315**, round_half(x32)=**0.10003662109375**,
  scale32=**1.0526316165924072**. Перед fp16 GEMM A:
  baseline `half(fp32(x32*scale32))` = **0.1053466796875**,
  half-ingress `half(fp32(float(half(x32))*scale32))` = **0.10528564453125**.
  Маска для этого элемента одинакова — он сохранён в обоих случаях.
  Пример рассчитан Python stdlib `struct` с явными IEEE fp32/fp16 округлениями,
  без CUDA и без предположения о случайном seed:

  ```python
  import struct
  f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
  f16 = lambda x: struct.unpack('e', struct.pack('e', x))[0]
  x, scale = f32(0.10006), f32(1/0.95)
  assert f16(f32(x*scale)) != f16(f32(f32(f16(x))*scale))
  ```

- **Доказательство backward границы:** нормализованный tensor теперь fp16 и
  разделяется между q/k/v. Baseline отдельные fp16 cast edges возвращают
  вклады к общему fp32 norm tensor; новый общий fp16 tensor получает вклады в
  half. Накопление в half в общем случае отличается от fp32 accumulation до
  RMSNorm Jacobian. Например, две представимые half величины 1 и 2^-11:
  fp32 сумма **1.00048828125**, half сумма **1.0**. Это дополнительная граница
  округления, даже если бы forward потребители совпадали. Wrapper q/v также
  возвращает base/result в другом dtype, согласно bnb inp_dtype контракту.
- **Как доказать на T4:** существующий
  `test_input_final_half_norm_with_real_bnb_lora_preserves_dropout` сравнивает
  RNG/masks, loss и toleranced LoRA gradients, но сравнение выполняется при
  одновременно изменённых input/final norms. Добавить isolated input-norm
  сравнение с nonzero LoRA B и штатным p=0.05, сохранить logits/loss, dx
  input norm и adapter gradients на нескольких seed; затем checks 5–7
  paired с baseline. Численный пример выше не доказывает превышение допусков.
- **Минимальное решение:** если требуется точная предпосылка «потребители
  уже кастовали и результат тот же», оставить input norm fp32 при активной
  LoRA dropout/многоветочном потреблении. Если принимается приближённая
  оптимизация, явно описать это ограничение и подтвердить допуски качества
  новым T4 full comparison; не выдавать RNG equality за numerical equality.
- **Влияние:** дополнительное округление может изменить adapter updates,
  loss curve и held-out quality на другом seed/скрытых строках; величина
  эффекта UNVERIFIED. fp32 input norm уменьшает выигрыш памяти и возвращает
  downstream casts; точного измеренного overhead этой альтернативы нет.

### 1. Потребители fp16 и согласованность readiness с настоящим LoRA forward

| Потребитель | Что проверено |
| --- | --- |
| q/k/v direct dense Linear | `_plain_half_consumer:47` требует CUDA fp16 autocast, same-device supported weight, exact nn.Linear, отсутствие instance override/hooks. Каждая исходная linear действительно кастует operands до GEMM. Несколько потребителей всё равно имеют backward границу P1-023-1. |
| k direct bnb Linear4bit | Exact module/name, frozen parameters, compute_dtype=fp16, compute_type_is_set и quant_state; fixed bnb forward кастует input до matmul. Он возвращает inp_dtype: forward значения до последующего потребления могут совпадать, dtype и gradient accumulation уже не обязаны совпасть. |
| q/v LoRA | `fp16_ingress_ready` требует actual PEFT Linear4bit type, установленный wrapper marker, отсутствие wrapper/base hooks и overrides, base compute fp16 и готовый quant state. Затем вызывает **тот же** `_eligible`, подставляя будущий input_dtype=fp16. Original Qwen2Attention вызывает q/k/v без дополнительных projection kwargs; mixed-adapter prehooks были бы отклонены. |
| Post-attention MLP | `_post_norm_half_ready:106` требует реальный Qwen2MLP без hooks/instance override, одинаковые projection types и проверку каждой projection. fp32 dense MLP без AMP отклоняется; frozen bnb fp16 может пройти и без AMP. Это убирает прежний dtype crash, но не является общим доказательством всех промежуточных rounding для любого нестандартного activation. Fixed task использует исходный SiLU и собственный fused MLP. |
| Final head | `_final_norm_half_ready:86` требует CUDA fp16 autocast, exact nn.Linear, supported same-device weight, отсутствие model/norm/head hooks и запроса output_hidden_states. Dense AMP head действительно потребляет fp16 operands; fused head также кастует hidden. При head fallback эта оптимизация может работать, но не включается без AMP. |

**Проверка alignment и позднего отказа:** `_eligible` применяет один и тот же
`data_ptr()%16` check в preflight и actual forward. Preflight смотрит на
исходный residual x, а actual смотрит на **новый** output `rms_norm`, выделенный
`torch.empty` (`kernels/rmsnorm.py:113`). Норма не возвращает offset view,
output contiguous, обычная CUDA allocator аллокация имеет достаточное
выравнивание. Таким образом, при aligned исходном x actual output не становится
misaligned; при misaligned исходном x preflight консервативно оставляет fp32.
Обычного воспроизводимого случая readiness=True → actual=False именно из-за
alignment/layout/dtype не найдено. Shape, device, autocast и статические module
flags в штатной последовательности также не меняются между проверками.

Если **принудительно** изменить состояние/forward или eligibility после
preflight, actual `_eligible=False` вызывает original PEFT с уже fp16 tensor
(`lora.py:110`–`:111`). Защитного восстановления fp32 для этой ветки нет, так
что original dropout тогда получает half. Это условный результат чтения кода,
а не найденная гонка штатного task; concurrent module mutation, custom norm
instance forward или намеренно изменяющие output обёртки не покрыты доказанной
готовностью. Нельзя заявлять защиту от любого последующего отказа. Существующий
тест отклоняет `_eligible` **одновременно** для preflight и actual, поэтому
проверяет ранний отказ, а не такой изменившийся во времени случай.

Во время обычного forward между preflight и q/v не добавлен RNG-вызов.
При принятом fast path `x.float()` даёт aligned contiguous fp32 input Dropout
того же размера; маска и продвижение RNG проверяются отдельным GPU-test.
До выполнения test на T4 это остаётся проверенной структурой, не новым логом.

### 2. ContextVar ownership, исключения, backward и eval/generation

`_half_norm_scope:30` сохраняет token и выполняет reset в `finally`.
Scope содержит **identity одного norm instance**, а не class/global bool.
Внешний final-norm scope охватывает decoder forward; внутренний input-norm
scope временно заменяет owner и затем восстанавливает final owner. Так final
scope не превращает все input/post norms в half. Для неподходящего пути scope
устанавливает None и после возврата восстанавливает предыдущий контекст.

Локально без torch/pytest исполнен AST **точной функции из 3c560d1**: вложенный
scope, ValueError во внутреннем scope, восстановление внешнего owner и reset
к None после выхода — **PASS**. Production-файлы не менялись.

Ни backward RMSNorm, ни add+RMSNorm, ни CE не читают `_HALF_NORM`; precision
решается в forward, backward использует saved tensors и ctx.exact_dtype.
Глобальной утечки scope в следующий batch или backward не найдено.

Eval/no-label/generation head уходит в original, но final scope может быть
включён в CUDA fp16 autocast, если plain head и остальные guards подходят.
Input norm scope также не ограничен training; без AMP он выключен, с AMP
может работать. LoRA Dropout в eval не бросает случайные элементы. Generation
cache/attention kwargs передаются прежним способом, output_hidden_states
отключает input/final compression. Это не обещание неизменности всех outputs:
слойные precision оптимизации остаются активны и имеют P1-023-1 boundary.

### 3. NF4 post-import callbacks — порядок, already-loaded, idempotence, фейки

Использованы те же verified wheel sources 0.50.2 из S-021; дополнительно
из wheel вне repo извлечён `bitsandbytes/__init__.py`.
Root импортирует autograd/nn и CUDA ops (при CUDA available), а
`__version__='0.50.2'` устанавливает только в конце (`:78`).

`_NF4_MODULES` содержит root, functional и cuda ops. Callback требует все три
реальных ModuleType и отказывает, пока **другой** module ещё initializing
(`patch.py:402`–`:415`). Текущий callback исключён из проверки: underlying
exec_module уже завершился, хотя Python ещё не сбросил его spec flag.

Для обычного `import bitsandbytes` после apply functional/backend callbacks
не ставят patch, пока root инициализируется. Root callback после своего
exec_module видит готовые children и уже заданный __version__, поэтому вызывает
installer без преждевременного circular import. При поздней загрузке cuda ops
его callback допускается, когда root/functional закончены. Внутренние
importlib.import_module installer получают уже имеющиеся объекты; новый
неготовый пакет не импортируется ради регистрации.

`install:459` обрабатывает и модули, загруженные **до** apply; порядок обхода set
не существенен, если они полностью загружены. Повторный installer сохраняет
обёртки через markers на обеих функциях. Сам finder устанавливается один раз.
Фейки SimpleNamespace/модули без package root отсекаются раньше; missing
functional/backend callable тоже не вызывают installer. Точный version gate
0.50.2 остаётся в `nf4_patch.py`.

Локально исполнен AST точного `_register_module` с metadata modules и spy
installer (не с torch/bnb runtime): missing backend → skip, root initializing
при callback backend → skip, root final callback → install, повторные callbacks
already-loaded modules → без повторного изменения spy wrappers, fake root → skip:
**PASS**. Spy доказывает scheduling, а не numerical correctness NF4.
Реальный pinned CUDA import и оба patched function markers проверяет добавленный
`test_nf4_hooks_registered_after_bnb_import`; его GPU-результат здесь UNVERIFIED.
Предыдущий вывод S-021 «NF4 installer не вызывается default apply» для этого
нового коммита **устарел**: 3c560d1 явно добавляет регистрацию.

### 4. RMSNorm out_dtype backward — исходный promote dtype сохранён

Forward сохраняет `ctx.exact_dtype = promote_types(x.dtype, weight.dtype)`
**до** optional final fp16 output store. Нормализованное q приводится к x.dtype
перед weight multiply, как раньше; out_dtype меняет только конечный store.

В input backward upstream half загружается как fp32, что соответствует обратному
cast half→исходный promoted dtype; `u=g*weight` затем приводится к x.dtype до
Jacobian. dx allocation остаётся x.dtype. Нет зависимости от активного scope
или forward autocast в момент backward.

В dgamma `OUT_HALF` теперь выбирается по **ctx.exact_dtype**, а не grad.dtype
(`rmsnorm.py:148`–`:154`). При x fp16/weight fp32 итог до optional cast был fp32:
промежуточный g*q поэтому не должен округляться в half. При x/weight обоих
fp16 exact output half и этот product cast нужен. Приведение final dw к
weight.dtype остаётся прежним. Ошибки этой dtype-логики не найдено.

Добавленные tests покрывают четыре пары dtype и frozen/trainable gamma,
comparison с exact-output.half, Qwen reference и variant counts по длинам.
Сравнение exact.half использует тот же kernel и не является независимым oracle
редукции; дополнительно есть независимая формула и отдельный fixed-row bitwise
Qwen reference. GPU execution локально не выполнялся.

### 5. P1-2: логический путь теста без AMP

В `test_bridge_without_amp_preserves_dense_fallback` модель содержит обычные
fp32 nn.Linear, CUDA autocast выключен. `_input_norm_half_ready` и
`_final_norm_half_ready` сразу возвращают False. Для post norm
`_plain_half_consumer(nn.Linear, x)` возвращает False без CUDA fp16 autocast,
поэтому `decoder.py:82` передаёт out_dtype=None. Норма возвращает fp32;
MLP original получает fp32 и не повторяет прежний Half != float crash.
Head по-прежнему возвращает logits в dense fallback и метрики через bridge.

Локальный metadata probe точных функций `_post_norm_half_ready` и
`_plain_half_consumer` из коммита: dense no-AMP reject, тот же dense AMP accept,
добавленный projection prehook reject — **PASS**. Это проверка guards без
GPU, не ложное заявление о pytest pass. P1-2 **закрыт статически**, а повторный
реальный regression test — обязательное оставшееся подтверждение.

### Предложенные проверки на T4 (не запускались)

```bash
python -m pytest solution/tests/test_integration.py -q
python -m pytest solution/tests/test_rmsnorm.py -q
python -m pytest solution/tests/test_nf4.py -q
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
python harness/run_task.py --repeats 2 --out runs_s023
python harness/check_rules.py --result runs_s023/result.json
```

Отдельно изолировать input/final compression для оценки P1-023-1; сравнить
RNG, значения после dropout перед A, gradients перед RMSNorm и adapter
updates при nonzero B/нескольких seed. Проверить NF4 functional/backend markers
после cold import и already-loaded apply. Значения скорости/памяти/quality
коммита de836e0 не переносятся на этот новый default автоматически.

## S-028 — unpad S-024 и взаимодействие с fused RoPE S-022

Дата: 2026-10-30. Проверен **неизменный снимок `2004b33`**:
`solution/unpad.py`, конец Qwen callback в `solution/patch.py`,
`solution/tests/test_unpad.py`; RoPE — реализация из `585de28`, присутствующая
в этом снимке. Текущие параллельные изменения S-027 не ревьюились.
Изменён только этот раздел отчёта; коммитов и Kaggle-запусков нет.

**Итог:** нового доказанного P0/P1 для штатного Qwen2/NF4/LoRA training-пути
не найдено. Множество допустимых attention-ключей и позиции real tokens
сохранены; dropout восстанавливает padded геометрию до native вызова.
Это статический вывод, **не GPU-сертификация**: изменяются формы GEMM/SDPA,
их округления и потенциально выбранные реализации. Численный риск раннего
fp16 input norm из P1-023-1 остаётся. Найдены P2 по стоимости синхронизаций
и полноте независимого integration acceptance.

### Источники и воспроизводимость

Здесь `SITE` означает локальный pinned source root:
`<scratch>`.
Ссылки на строки `solution/*` ниже относятся **к `git show 2004b33:<path>`**,
а не к меняющемуся checkout; ссылки RoPE также совпадают с `585de28`.

- Transformers 5.16.1: `SITE/transformers/models/qwen2/modeling_qwen2.py:87-102`
  (cos/sin), `:105-138` (RoPE), `:195-234` (attention), `:341-407` (model loop).
- `SITE/transformers/masking_utils.py:235-285`, `:487-519`, `:851-859`,
  `:935-995`: SDPA causal/padding mask, skip и packed-position detection.
- `SITE/transformers/integrations/sdpa_attention.py:15-37`, `:94-107`,
  `:124`, `:158-170`: repeat KV, masked GQA policy, SDPA и contiguous output.
- PEFT 0.20.0: `SITE/peft/tuners/lora/bnb.py:513-552`,
  `SITE/peft/tuners/lora/layer.py:1033-1071`: base→dropout→A→B порядок.
- PyTorch 2.11.0: [native CUDA dropout](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/cuda/Dropout.cu#L188-L243),
  `:360-388` (launch/Philox). Прочитан upstream source, не выполнялся GPU-код.

Локально распакован `git archive 2004b33 solution` во временный каталог,
в нём выполнено:

```text
<tmp>/s004-integration-venv/bin/python -m pytest solution/tests/test_unpad.py -q -rs
16 passed, 15 skipped in 0.17s
```

Причина всех 15 skips: отсутствует torch. Это проверяет layout/scope/fake
control flow, но **не** CPU numerical equivalence и **не** CUDA/bnb.
Независимая stdlib-проверка causal key sets перебрала 5628 layouts
(B=1..4, S=1..7, все непустые right-padded lengths): расхождений нет.
Это математическая проверка масок, а не запуск настоящего SDPA.

### P2-028-1 — заявленная одна синхронизация на batch не описывает весь preflight

- **Файл:строка:** `solution/unpad.py:220-226`; `_scatter:96-103`,
  `_bind_dropout:254-259`, attention `:285-299`.
- `attention_mask.detach().tolist()` переносит mask на host. Затем CUDA
  boolean indexing `labels[attention_mask == 0]` имеет динамический размер
  (обычно nonzero/index path с определением размера на host), а `bool(...all())`
  отдельно требует scalar device→host. Поэтому обещать только один sync
  для всего batch нельзя. Внутри layer loop повторных `.item()` не найдено.
- На каждой из q/v dropout ветвей каждого слоя `_scatter` создаёт padded
  zeros, затем **out-of-place** `index_copy` выделяет ещё один padded результат;
  native dropout создаёт output/mask, после чего `_pack` делает gather.
  Два padded fp32 буфера и полная dropout-mask остаются ценой RNG-equivalence.
- Attention делает до B SDPA вызовов на слой, затем `cat`; для B=4 и 28 слоёв
  это до 112 SDPA вызовов в forward вместо 28. Helper дополнительно делает
  transpose/contiguous для каждой строки. Маски кэшируются по length только
  внутри одного PackedBatch и переиспользуются между слоями, не между шагами.
- **Минимальное улучшение:** проверять padding labels одним scalar reduction
  без динамического boolean selection; проверить безопасное заполнение
  destination без второго полного padded allocation. При любом изменении
  оставить padded native dropout и autograd-контракт. Отдельно измерить
  preflight, gather/scatter/cat и реальный SDPA backend на T4; знак выигрыша
  нельзя вывести из числа убранных pad tokens.

### P2-028-2 — standalone unpad tests не гарантируют совместный путь всех патчей

- **Файл:строка:** `solution/tests/test_unpad.py:129-146`, `:236-313`.
- Fixture устанавливает только `patch_qwen2_unpad(module)`. Она не вызывает
  `solution.apply()`, не проверяет RoPE marker и не требует `outputs.logits
  is None`. bnb fixture отдельно включает lora.py, но не norm/decoder/head.
  `_statistics:219-233` принимает и dense logits, и fused step, поэтому
  standalone зелёный тест может не упражнять fused CE/bridge вообще.
- При запуске всей suite предыдущие integration tests могут уже вызвать
  apply; тогда тот же тест попадает в совместный путь из-за глобального
  состояния. Это полезный дополнительный охват, но не независимый acceptance.
- **Минимальный тест:** в отдельном процессе вызвать apply дважды, создать
  реальную Qwen2+PEFT bnb q/v LoRA с nonzero B, включить fp16 AMP, убедиться
  в non-None plan, активном RoPE marker и `outputs.logits is None`, сравнить
  masks/RNG/loss/A,B gradients и ChunkedStep с SOUP_UNPAD=0. Отдельно прогнать
  trainer bridge/Rewind, чтобы проверялись gather/token counts и row outputs.
- Fallback parametrization `:316-344` покрывает eval/cache/left/no-mask/hidden/
  supervised padding. Нет отдельных численных тестов inputs_embeds,
  gradient checkpointing, explicit position_ids и exception после входа
  в label scope. Static guards ниже проверены; это пробел тестов, не доказанный
  дефект реализации. Полные 79 Soup tests на этом снимке здесь не запускались.

### 1. Attention: допустимые ключи совпадают, dispatch не обещает bitwise equality

Baseline SDPA создаёт bool mask `causal AND key_padding`. Для real query
позиции i строки b с длиной L_b её true keys равны `0..i`, i<L_b; правые
pad keys не входят. Unpad `:286-296` вызывает SDPA отдельно на `[0,L_b)` с
bool tril и `is_causal=False`: множество ключей то же, других строк нет.
Пустые/left/interior-padded строки отклоняет `right_padding_lengths:43-66`.
Полностью непаддированный batch остаётся original (`:224`).

`self.scaling` передаётся без изменения, attention dropout допускается
только 0 (`:124`), поэтому сегментация не потребляет attention RNG.
Оба masked helper-вызова получают non-None bool mask: SDPA helper повторяет
KV по `num_key_value_groups`, не включает GQA fast path. Autocast и tensor
dtypes не переопределяются; q/k RoPE promotion и v следуют тем же правилам.
Backend enable/disable settings не изменяются. Однако форма batch/sequence
и strides меняются: **mem-efficient backend не зафиксирован этим кодом**,
его выбор и численное совпадение надо проверить в профиле T4. Отсутствие
pad keys математически не меняет real softmax, но иной reduction/GEMM shape
не доказывает равенство floating-point выходов или gradients побитно.

### 2. Position ids, cache и RoPE в packed форме

При position_ids=None baseline Qwen2 строит `[0..S-1]` из sequence length.
Unpad `:324-328` строит те же позиции **до** packing и выбирает real индексы:
получаются `[0..L_0-1, 0..L_1-1, ...]`, а не единый arange(T).
Явные position_ids допустимой формы также выбираются, не перенумеровываются.
Baseline при non-None 2D attention_mask не включает отдельное обнаружение
packed sequences по resets (`masking_utils.py:851-859`), поэтому отсутствие
этого распознавания в unpad само по себе не меняет контракт.

`attention_mask={"full_attention": None}` в `unpad.py:335` соответствует
Qwen2Model's prepared-mask contract и исключает глобальную causal mask
между packed строками. Настоящую mask применяет наш segmented attention.
Sliding layers и non-default/dynamic RoPE отвергаются preflight. Cache
обязательно отсутствует/выключен. Отдельный `cache_position` попадает в
неизвестные kwargs и вызывает original, поэтому с cache-offset он не теряется.

Qwen2RotaryEmbedding вычисляет cos/sin из переданных позиций, с shape
`[1,T,D]`. q/k после projection имеют `[1,Hq|Hk,T,D]`; BHSD strides и even D
удовлетворяют `solution/rope.py:25-45`. Kernel адресует token как `row % Tokens`,
cos batch через CBatches и не предполагает, что позиции монотонны глобально.
RoPE — независимая операция по token, reset между строками ей не мешает.
`unpad.py:284` вызывает модульный атрибут, действительно перехватывая патч.

`rope_pair_forward_kernel` (`kernels/rope.py:25-61`) отдельно округляет оба
произведения в promoted dtype перед суммой, FMA выключен при launch `:109-113`.
Backward `:65-98` использует adjoint: dx1=g1*c1+g2*s2,
dx2=g2*c2−g1*s1, с cast каждой ветви к input dtype до accumulation.
Для двух половинок вращения это правильные знаки; unpad не меняет формулу.
Сохраняются cos/sin и metadata, не q/k; masked loads/stores разделяют q/k
и хвосты. S/T — runtime argument; recompilation tests существуют, но packed
B=1/T sweep здесь не запускался. Unsupported layouts/cos gradients идут в
собственный eager fallback, не в optional hub wrapper (`rope.py:70-76`).
Ни один из этих патчей не запрашивает готовый hub kernel.

### 3. LoRA dropout: instance ownership и fp16 ingress

`_dropouts:127-176` разрешает vanilla одиночный q/v адаптер, прямые A/B fp32,
точный Dropout/Identity, без variants/DoRA/merged/disabled/mixed hooks.
Другие обучающие stochastic Dropout в дереве приводят к original до RNG.
Native Dropout class не меняется: только выбранные объекты получают MethodType
с marker (`:241-262`); повторный bind не оборачивает второй раз.
Вне `_ACTIVE` bind вызывает сохранённый original. `_packing_scope:82-88`
и causal label scope `:353-360` восстанавливаются через finally, включая
исключения; backward native dropout не требует этих scope (checkpointing
отвергнут). Привязка остаётся на instance и безопасно обслуживает eval/fallback.

На fp16 входе зарегистрированный bnb wrapper сначала выполняет base_layer,
затем `x.float()`, затем тот же Dropout (`lora.py:116-123`). Bind видит
`[1,T,H] fp32`, scatter даёт `[B,S,H] fp32 contiguous` с allocator alignment;
после original dropout real rows собираются обратно. Тип/форма guard `:254`
для этого пути не падает. `_eligible` не запрещает certified dropout bind,
а `fp16_ingress_ready` не принимает instance override самого projection.
Dense PEFT LoRA кастует input к fp32 A перед dropout, что также подходит.

По CUDA Dropout.cu геометрия/Philox increment зависят от numel/device,
vectorization — от dtype/layout/alignment. Маска берётся из seed/offset/thread,
не из значений. При одинаковом исходном RNG, p и порядке q/v восстановленный
fp32 padded вызов сохраняет mask и расход RNG. Это условно на обычный aligned
baseline norm output; произвольный невыравненный custom forward не сертифицирован.
Паддинговые нули не влияют на RNG, но не позволяют считать `output != 0`
доказательством равенства маски на самих pad позициях. Тест корректно сравнивает
real позиции и конечное generator state.

Уже существующее раннее half-округление из P1-023-1 не становится точным:
`dropout(float(half(x)))` не равно `dropout(x)`, и общие half gradients имеют
иные accumulation boundaries. Packing дополнительно меняет GEMM shapes;
одинаковая маска не доказывает численную эквивалентность A/B updates.

### 4. Fallbacks и наблюдаемые выходы

`_plan:178-238` требует training, labels текущего causal call и тот же объект
attention_mask, SDPA, use_cache=False, без past/checkpoint/sliding/dynamic RoPE.
Eval/generation без labels, cache, left padding, missing mask, requested
hidden_states/attentions, unknown kwargs и nonignored pad labels вызывают
original до stochastic work. `_ACTIVE` предотвращает повторную упаковку.
**inputs_embeds не является fallback**: при согласованных B,S/device оно
поддержано и pack выполнен в `:332`; numerical test этого варианта отсутствует.
Fake-модули отсекаются точными module/class checks `:375-380` и guard callback;
однако статическая проверка фейков не заменяет запуск всех 79 Soup tests.

После model norm real hidden states scatter возвращает в исходные row-major
позиции, padded hidden states становятся нулём (`:337`). Поэтому **полный
last_hidden_state и padded logits не равны baseline**, если наблюдать их
непосредственно. Запрос hidden_states приводит к fallback; контракт обычного
training loss/метрик сравнивает допустимые позиции, не pad logits.
Shift остаётся отдельно внутри каждой исходной строки (`autograd.py:10`):
последний real token перед padding предсказывает ignored pad target, между
строками target не переезжает. Последняя позиция всей строки также ignored.
Loss sum/denominator не меняются, ignored token_loss равен нулю.

Entropy на padded logits меняется, но `_log_metrics` умножает её на исходную
attention_mask; accuracy учитывает только labels[:,1:] != -100. Поэтому
паддинговые предсказания не входят ни в один из этих агрегатов. num_tokens
использует прежнюю mask. ChunkedStep остаётся `[B,S]`, с прежними ignored
losses и valid counts, пригодными для Rewind per-row means. Это формулы и
static graph proof, не выполненный trainer/Rewind integration test.
В `2004b33` head всё ещё обрабатывает полный scattered `[B,S,H]`, поэтому
CE forward/entropy вычисляются и для padding. Packed head — отдельное S-027,
его корректность и экономия этому ревью не приписываются.

### Следующие проверки T4 (здесь UNVERIFIED)

1. Standalone совместный integration test из P2-028-2; несколько seed,
   nonzero LoRA B, обычный r=16/p=0.05, mixed/length-one/equal-short rows.
2. Explicit position_ids и inputs_embeds; disabled scopes после исключения,
   checkpoint/use_cache/generation fallbacks. RoPE B=1/T sweep с backward.
3. `pytest solution -q -rs` и `SOUP_APPLY_SOLUTION=1 pytest tests -q` в новых
   процессах; затем неизменный paired harness/checks, с финальным snapshot.
4. Профиль реального SDPA backend, числа SDPA вызовов, host sync, dropout
   scatter/gather, cat и остаточной padded CE. Не фиксировать backend ради
   хорошего профиля и не переносить сюда результаты ce1ff09 автоматически.

## S-030 — async MLP offload и partial recompute S-025

Дата: 2026-10-30. Проверен **только коммит
`796bf0e2940de90df94b497dd52c6f8b3d0a6a73`**: `solution/mlp.py` и
`solution/tests/test_mlp.py`, с чтением вызываемого SwiGLU backward.
Строки ниже относятся к `git show 796bf0e:<path>`, не к checkout.
Изменён только отчёт; коммитов, исправлений кода и Kaggle-запусков нет.

**Итог:** доказанного нового P0/P1 для штатного последовательного Qwen2
forward/backward не найдено. D2H/H2D и повторное использование upload/staging
имеют явные event-зависимости; host leases и weak links разделяют lifetime
CPU activation data и графов. Обнаружены P2 по границе concurrent replay,
неполному stress acceptance и цене фиксированной reservation. Всё CUDA
поведение, скорость/пики памяти и совместный unpad-прогон здесь UNVERIFIED.
Default store математически не изменён; оба новых режима opt-in.

### Источники и выполненные проверки

- Снимок: `git show 796bf0e:solution/mlp.py`, `:solution/tests/test_mlp.py`,
  `:solution/kernels/swiglu_fused.py:67-78`; diff относительно родителя.
- Локальная копия torch 2.11 `torch/csrc/autograd/init.cpp:516-518`
  (`<tmp>/featherduster-torch-autograd-init.cpp`) подтверждает binding
  `_get_current_graph_task_keep_graph` в `torch._C._autograd`.
- [PyTorch 2.11 CUDA stream semantics](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-streams)
  и [Tensor.record_stream](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.record_stream.html):
  порядок разных streams требует зависимостей; allocator tracking защищает
  lifetime, но не заменяет event wait перед чтением/перезаписью. CUDA backward
  op использует stream соответствующего forward op; consumer gradients на
  другом stream всё равно должен синхронизировать вызывающий код.

Неизменный `git archive 796bf0e solution` распакован во временный каталог.
Python syntax compilation обоих review-target файлов прошла без исполнения.
Команда в этом каталоге:

```text
<tmp>/s004-integration-venv/bin/python -m pytest solution/tests/test_mlp.py -q -rs
SKIPPED solution/tests/test_mlp.py:8: No module named 'torch'
1 skipped in 0.08s; exit code 5 (нет собранных исполнимых тестов)
```

Это **не PASS численных тестов**. Torch/CUDA/bnb локально отсутствуют;
не было ни измерения памяти, ни stress проверки реальных CUDA events.

### P2-030-1 — одновременный replay одного ticket не имеет отдельного ownership

- **Файл:строка:** `solution/mlp.py:162-169`, `:214-242`, `:414-443`.
- Последовательный retain/replay поддержан, но ticket хранит ровно один
  `device_slot` и `compute_stream`. Повторный `begin_backward` при уже
  используемом slot просто возвращает тот же slot и переписывает stream;
  отдельного per-invocation reader count или запрета reentry нет.
- Допустимое на уровне Python state-machine чередование двух backward
  одного retained ctx: A и B вызвали begin; A получил tensors и вызвал
  release_device; B ещё не вызвал tensors. A очищает ticket.device_slot,
  затем B обращается к `slot.ready` у None. Если B уже получил views,
  первый release снимает in_use до завершения второго читателя и позволяет
  pool назначить slot другому ticket. Pool RLock защищает отдельные методы,
  но не весь begin→read→release интервал одного reader.
- Это статическая граница, **не воспроизведённая CUDA гонка**. Обычный Soup
  запускает последовательный backward; serial repeated backward с retain
  её не вызывает. Упомянутые в коде независимые concurrent backward graphs
  имеют разные tickets и не тождественны concurrent replay одного ctx.
  PyTorch обычно направляет CUDA backward этого узла на его forward stream,
  поэтому различие stream IDs само по себе не служит доказательством гонки.
- **Минимальный фикс:** явно запретить overlapping begin одного ticket
  до release (понятная ошибка до использования buffer) либо вести ownership
  для каждого reader и ждать всех completion events перед eviction.
  Задокументировать serial-only replay, если concurrency не входит в контракт.
- **Влияние:** robustness нестандартного многопоточного autograd; не доказанный
  дефект training checks и не основание менять default store.

### P2-030-2 — stress-тесты не проверяют ключевые нештатные переходы пула

- **Файл:строка:** `solution/tests/test_mlp.py:226-271`, `:310-356`, `:376-392`.
- Есть хорошие проверки bits/gradient относительно store, serial retained
  backward, повторного использования host pointers при разных lengths,
  multilayer prefetch/peak и call counts partial recompute. Но нет явных
  тестов out-of-order branches, eviction speculative prefetch, двух
  одновременно живых retained graphs, growth→retirement при rows>capacity,
  graph destruction до завершения D2H и исключения после begin_backward.
- Cross-stream test создаёт forward **и backward внутри одного producer
  context**; он упражняет side-stream pool и внешний wait, но не stress
  независимых graph readers с одновременно занятыми staging slots.
- При исключении между begin и release нет finally, возвращающего lease
  (`mlp.py:420-443`): cleanup зависит от context/ticket destruction. Пока
  caller держит failed graph, slot остаётся leased/in_use; следующий graph
  может расширить pool. Это не установленная утечка нормальных шагов, но
  полезный тест восстановления после ошибки/освобождения графа отсутствует.
- **Минимальное дополнение:** adversarial маленькие GPU тесты указанных
  переходов с проверкой equality/store, event completion, live leases и
  количества allocations после gc. Для нового unpad отдельно sweep `[1,T,H]`
  с разными T, partial+offload и реальным bnb/fp16 AMP; текущий length test
  использует `[4,S,H]` и только один слой.
- **Влияние:** новая async ветка пока не имеет воспроизводимого GPU acceptance
  в этом ревью; source proof не заменяет проверку allocator scheduling.

### P2-030-3 — резервирование не уменьшается при малом T / partial recompute

- **Файл:строка:** `solution/mlp.py:100-117`, `:256-275`, `:279-299`.
- Lazy first use берёт capacity=max(rows,2048) и host count из полного
  config.num_hidden_layers. Даже если unpad даёт намного меньше T или
  offload применяется только к нескольким слоям после partial recompute,
  reservation остаётся полной: два host fp16 tensors на каждый из count
  slots, upload pair и два staging pairs на capacity строк.
- Для I=8960, capacity=2048, count=28 это **расчёт**, не измерение:
  1960 MiB pinned CPU и 210 MiB CUDA staging. Объём фактических transfer
  использует текущие rows, но размер постоянно живущих buffers не сокращается.
  После opt-in pool глобально кэшируется; выключение флага внутри того же
  процесса не освобождает уже выделенную reservation. Default никогда не
  включает offload и сам по себе таких allocations не делает.
- При очень малом T фиксированное staging может перекрыть экономию store;
  full+partial recompute могут оставлять большую часть host slots пустыми.
  Рост при большем N/retained graphs остаётся в high-water capacity.
- **Минимальное улучшение:** optional preparation с явным корректным bound
  и числом реально offloaded layers либо задокументированный memory budget;
  auto-grow со временем жизни старого pool уже предусмотрен. Не уменьшать
  capacity так, чтобы ломать допустимые большие T или retained graph.
- **Влияние:** pinned RAM budget, init cost и низкотокенные memory peaks;
  реальная выгодность требует T4 измерений, не меняет numerical result.

### 1. Event order и lifetime: штатная последовательная ветка

| Ресурс / переход | Защита в 796bf0e | Вывод |
| --- | --- | --- |
| gate/up → upload | `store:139-146`: copy на текущем compute stream, затем produced | Исходный gate/up не читается напрямую из D2H; его storage можно освободить после собственных current-stream uses. |
| Предыдущий D2H → новая запись upload | `store:143`, `:155`: wait upload_free, event после обеих D2H copies | Следующий слой не перетирает upload раньше завершения DMA. |
| upload → pinned host | `:147-159`: D2H ждёт produced, upload record_stream для обоих streams | Готовность данных и allocator lifetime учтены раздельно. |
| Предыдущее H2D чтение host → новая D2H запись host | `:149-153`: wait slot.read при reuse | CPU lease можно снять до фактического completion, поскольку следующая запись ждёт чтение. D2H writes упорядочены одним stream. |
| host → staging | `prefetch:189-199`: wait host.written, wait slot.used, record ready/read | Нет чтения незавершённого D2H или перезаписи staging до завершения прошлого compute reader. |
| staging → SwiGLU backward | `tensors:225-230`: compute.wait ready, record_stream compute | Даже speculatively prefetched ticket ждёт собственную загрузку. |
| Последний read staging → reuse | `:439-443`, `release_device:232-242`: used после kernel launch | SwiGLU возвращает новые dgate/dup allocations; последующий projection backward уже не читает gate/up staging. |

Upload_free первоначально записывается на allocation stream (`:110-112`),
учитывая возможные старые uses от caching allocator. Staging allocations
создаются на H2D stream (`:113-114`), там же впервые заполняются. Persistent
pool держит device/host storage; tickets держат pool, но pool/previous/owner
ссылаются на ticket слабо и не удерживают autograd graph глобально.
`record_stream` не принимается за защиту от логической перезаписи: её дают
именно ready/used/read/upload_free waits в таблице.

Если два staging заняты speculative prefetch, required ticket может вытеснить
только `not in_use` slot (`:170-176`); старый ticket теряет device_slot,
но сохраняет host data и может загрузиться снова. Предыдущая speculative
H2D copy и новая copy сериализованы общим H2D stream. In-use slot не вытесняется;
для дополнительных независимых simultaneous readers выделяется ещё slot.

### 2. Порядок backward, retain и освобождение

`previous` связывает **offloaded** calls в прямом forward порядке слабой
ссылкой (`:134-138`, `:209`). Стандартный обратный проход по цепочке слоёв
сначала требует последний ticket, затем prefetch его predecessor (`:221-223`),
что совпадает с depth-first reverse layer order. Recomputed layers отсутствуют
в цепочке и не требуют загрузки. last между шагами указывает на closed/dead
previous, который отбрасывается, если обычный backward был завершён.

Нестандартный последовательный порядок не использует predecessor как источник
математических данных: каждый ctx вызывает required prefetch **своего** host
slot. Предсказание может зря загрузить другой ticket, но корректность данных
от него не зависит. Closed previous пропускается, speculatively prefetched
данные можно вытеснить безопасно. Полезность prefetch для ветвлений не доказана.

`_get_current_graph_task_keep_graph` запрашивается **до** nested autograd.grad
(`:415-424`), поэтому локальное retain_graph=True не подменяет флаг внешнего
GraphTask. Serial retained backward освобождает staging, но оставляет host
lease; финальный backward без retain вызывает close. Replays без сохранённого
графа должны завершаться явной ошибкой (`:215-216`), не читать reused host.
Старый torch без private query консервативно ждёт context destruction.
Если retained graph удалён без повторного backward, `__del__:251-253` закрывает
lease; pending read/write защищаются events при следующем reuse.

### 3. Pool capacity, variable T и default/partial precision

Pool key — `(x.device, intermediate_width)`, current rows — numel(x)/H.
Ticket сохраняет **свою** shape и rows (`:203-209`); store/prefetch копируют
`[:rows]`, tensors возвращает view исходной shape. Поэтому `[B,S,H]` и
unpad `[1,T,H]` совместимы без сохранения прошлого T или паддингового хвоста.
Последовательные шаги с T<=capacity и своевременно closed tickets переиспользуют
host slots. Непустые более старые shapes не участвуют в новом view.

При росте capacity создаётся новый pool, старый попадает в retired list;
cleanup требует отсутствия leased host slots и завершения host write/read и
staging used events (`:262-273`). Tickets дополнительно держат свой старый
pool. Очистка retired list происходит при следующем `_pool_for`, не фоново;
нет обещания мгновенного возврата pinned RAM после последнего enlarged шага.
Retained/concurrent graphs могут расширять host count, independent readers —
staging count; это не фиксированный upper bound для произвольного autograd.

При unset новых env flags recompute=False/offload=False: сохранение gate/up
в fp16, projection edges и backward store совпадают с родителем коммита.
`SOUP_MLP_OFFLOAD` включается только точным `1`, partial count по умолчанию 0;
полный `SOUP_MLP_RECOMPUTE=1` имеет приоритет, recomputed layer не offload.
Grad-disabled/input-without-grad путь не создаёт ticket и не меняется.

Partial выбирает первые k по forward depth (`:302-322`), k проверяется
от 0 до config.num_hidden_layers. Explicit prepare назначает индексы при
обходе modules; lazy режим полагается на общий config и обычный порядок
первого Qwen forward. Для нестандартного вызова отдельных слоёв это будет
порядок первого использования, а не доказанный structural layer index;
для такого сценария нужен prepare. Weak order map не хранит modules сильно.

Recompute сохраняет input fp16 и исходный input dtype/AMP state (`:383-389`),
при backward восстанавливает input dtype и ровно captured autocast (`:427-432`).
Gate/up проекции frozen и гарантируют half compute eligibility; исходные
bnb fp32 outputs представляют half values, поэтому offload half copy не
добавляет округления сверх существующего store. Возврат dgate/dup сохраняет
исходный projection-output dtype; projection backward выполняется вне AMP,
dX приводится к исходному dtype. Никаких новых RNG calls нет.
Partial смесь использует тот же full recompute механизм по выбранным слоям,
не новую формулу производной. Численная эквивалентность и реальные bnb
quantization/allocator взаимодействия всё равно требуют GPU тестов.

### Предложенный acceptance T4 (не запускался)

1. `python -m pytest solution/tests/test_mlp.py -q -rs` на pinned torch/bnb;
   обратить внимание на bits, retained backward, pointers и memory assertions.
2. Добавить P2-030-2 stress cases с variable `[1,T,H]`, final graph deletion,
   capacity growth, independent graphs/out-of-order branches и ясным отказом
   overlapping same-ticket replay (либо корректной reader accounting).
3. Измерить no-env default отдельно от partial k/offload; проверить pinned CPU
   high-water, CUDA peak, PCIe transfers и event stalls без переноса cache.
4. Совместный unpad+offload+partial paired harness/checks с прежними trainer
   settings и отсутствием параллельной GPU-работы. Ни ce1ff09 результаты,
   ни локальная syntax compilation этот opt-in режим не валидируют.
