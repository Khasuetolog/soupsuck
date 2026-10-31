# Codex-2: correctness/autograd/memory audit (S-038)

Дата начала: 2026-10-30. Проверяемый HEAD:
`b1dcb59602ea014a6f0273233057d620d4f42dd5`.
Основание: `tasks/ERROR_AUDIT.md`, а не прежние failing логи.
Ownership: MLP/unpad/embedding/decoder/LoRA/RoPE/NF4 installer и
`solution/tests/test_unpad.py`, `solution/tests/test_mlp.py`.

## Результат и границы

Доказанного текущего дефекта в проверенном коде пока не воспроизведено.
Реализация и допуски тестов **не менялись**; оптимизации/изменения политики
не выполнялись. Установленный default async offload и unpad сохранены.
Незакоммиченные `.DS_Store` и `team-share/` не являются исходниками и не
использовались. `harness/`, `tests/`, `task/` не изменялись.

**VERIFIED — новый T4 unit-прогон текущего HEAD**, получен 2026-10-31
по локальному времени. Прочитаны сами артефакты:
`runs_kaggle/20261030T185618838149Z_unit_b1dcb59_correctness/out/`.

| Evidence | Новый результат |
| --- | --- |
| `status.json:expected_commit` | `b1dcb59602ea014a6f0273233057d620d4f42dd5` |
| `status.json:commands` | gpu/pip/prepare_data/pytest_solution/pytest_soup exit code 0 |
| `pytest_solution.log` | **323 passed, 0 failed, 0 skipped**, 10 warnings, 127.85 s |
| `pytest_soup.log` | **79 passed, 0 failed**, 14 warnings, 9.90 s |

Полная solution suite включает исторический `lengths3-9-bnb`, оба fresh-process
joint bridge tests и все MLP offload stress cases. На этом SHA падение
**не воспроизвелось**; это новое подтверждение, не перенос 0c58998 истории.
Отдельного запуска одной ноды не было: подтверждение взято из полного
прогона без skipped, с проверкой состава параметров в исходниках.
Warning bnb про H=80 относится к предусмотренному native fallback и не
скрывает failed tests. Точных max_abs/oracle tensor чисел успешный `-q` лог
не выводит: прохождение assertions подтверждено, числа ошибок не выдуманы.

Проверенные GPU unit assertions подтверждают loss/entropy/accuracy, LoRA
A/B gradients и dropout RNG в тестовых конфигурациях, а также имеющиеся
offload lifecycle/exception/replay tests. Это не абсолютное доказательство
отсутствия гонок для всех schedules. Real module reload ниже и новый
full harness на b1dcb59 отдельно не запускались; их статус UNVERIFIED.

## Локально выполнено

Python 3.14/macOS, torch/CUDA недоступны. Использован существующий pytest venv:

```bash
<tmp>/s004-integration-venv/bin/python -m pytest \
  solution/tests/test_unpad.py solution/tests/test_mlp.py -q -rs
# 17 passed, 28 skipped in 0.38s; exit 0

<tmp>/s004-integration-venv/bin/python -m pytest \
  solution/tests/test_integration.py solution/tests/test_embedding.py \
  solution/tests/test_lora.py solution/tests/test_rope.py \
  solution/tests/test_nf4.py -q -rs
# 17 passed, 30 skipped in 0.22s; exit 0

python3 -m py_compile solution/mlp.py solution/unpad.py \
  solution/embedding.py solution/decoder.py solution/lora.py \
  solution/rope.py solution/nf4_patch.py \
  solution/tests/test_unpad.py solution/tests/test_mlp.py
# PASS
```

Все skips уже существовали и вызваны отсутствием torch. Ни одного skip,
нового допуска или изменения численного acceptance не добавлено.
Локальные PASS проверяют layout, fake-module guards, ContextVar ownership,
холодную регистрацию, но не CUDA арифметику и не DMA lifetime.

Дополнительно в отдельном Python процессе прошли три idle reload-цикла
`solution.unpad`, `embedding`, `decoder`, `lora`, `rope`, `nf4_patch` с
двойным apply после каждого. `_PatchFinder` остаётся один; exception внутри
`_packing_scope` очищает scope; torch/triton не импортируются. Это **не**
reload настоящего Qwen/PEFT/CUDA module и **не** reload во время active graph.

## Исторический bnb-параметр: что сравнивает текущий тест

Точная нода:

```bash
python -m pytest \
  'solution/tests/test_unpad.py::test_padded_equivalence_loss_statistics_lora_gradients_rng[lengths3-9-bnb]' \
  -vv -rs -s --tb=long
```

Конфигурация (`test_unpad.py:530-654`): B=4, S=9, lengths=(5,5,5,5),
T=20, H=64, два Qwen2 слоя, q/v LoRA r=4, alpha=8, p=0.05, nonzero B,
реальные bitsandbytes Linear4bit при CUDA, AMP fp16. Forward сравнивает
SOUP_UNPAD=0 и 1 при одном initial generator state; backward loss*128.

Проверки остаются:

- exact CUDA RNG state и exact dropout masks на real positions;
- packed plan присутствует ровно при наличии padding;
- scalar loss и token_loss, masked entropy и accuracy/count;
- Rewind row means/counts, включая строку длины 1;
- конечность всех LoRA A/B gradients;
- прежний padded/packed global gradient norm error <=3%*norm(reference)+2e-6;
- обе полные gradients относительно независимого fp64 model oracle
  с тем же norm budget; elementwise rtol=0.03, atol=0.001 для каждой
  фактической Linear dW относительно fp64 contraction её собственных X/dY;
- finally-clean scopes `_ACTIVE`, `_LABELS`, `_HEAD_CAPTURE`.

FP64 oracle (`:412-498`) воспроизводит весь малый Qwen, frozen weights/NF4
после dequantization, attention/RoPE/RMSNorm/MLP, dropout masks, LoRA и loss.
Это не только локальный GEMM ideal. Существующий dense-native sanity test
`:501-527` проверяет oracle equations. Историческая причина S-034 —
fp16 cancellation в upstream dY, а не доказанный текущий баг. Здесь она
**не принимается как новое воспроизведение** и не используется для ослабления
budget. При новом failure нужен полный diagnostic report по A/B (max_abs,
worst coordinates, reference/actual, X/dY, own contraction и model oracle),
а не новая настройка rtol/atol.

## Unpad, causal positions и метрики: static verification

Ссылки ниже относятся к HEAD b1dcb59.

| Область | Проверенный контракт / исходники | Статус |
| --- | --- | --- |
| Geometry | `_batch_metadata:160-177` валидирует binary mask и ignored pad labels; `right_padding_lengths:47-68` отвергает пустые/left/interior padding | CPU control tests PASS; CUDA transfer не проверен локально |
| Packed indices | `_indices_from_lengths:180-195` сохраняет row-major real-token order, `_pack:145-146`, `_scatter:149-157` возвращают B,S порядок | CPU metadata tests PASS; numerical scatter требует torch |
| Label shift | Model возвращает исходный padded layout; PackedHead хранит batch indices и identity labels; head использует исходный per-row shift и denominator | Совместный GPU тест PASS; head/autograd.py отдельно проверяет Codex-1 |
| Attention | `:355-390`: SDPA отдельно для каждой исходной строки, bool tril, прежние scaling и GQA helper, dropout=0 | Нет межстрочного attention; floating-point reduction shape меняется, не bitwise обещание |
| Positions | `:414-425`: arange(S) до gathering; real positions каждой строки 0..L-1; prepared full_attention mapping отключает только глобальную packed mask | Позиции не заменены на arange(T); cache/sliding/dynamic RoPE отсекаются |
| Capture lifetime | `CapturedHead.take:107-115` одноразовый, identity/dtype/device/shape guards; `:119-141` finally reset; `:427-429` publish только текущего model call | Pure capture/nesting/exception test PASS |
| Dropout | `_bind_dropout:331-352` восстанавливает contiguous padded fp32, вызывает тот же native module один раз, собирает real rows; instance MethodType вместо глобального патча | Ownership CPU PASS; exact CUDA masks/RNG PASS в новом T4 unit |
| Fallbacks | `_plan:271-328` отсекает eval/cache/no-labels/checkpoint/sliding/unknown kwargs и неподдерживаемые outputs | Existing fallback tests PASS в новом T4 unit |

Для real query i<L_b causal+padding mask baseline разрешает keys 0..i;
per-row tril unpad даёт то же множество. Pad queries могут иметь другие
hidden/logits, но labels ignored, entropy агрегируется с исходной mask,
accuracy с shifted labels!=-100; num_tokens не берётся из packed T.
Голова сохраняет token stats в B,S формате для Rewind. Полное равенство
padded hidden/logits не заявляется и не требуется выдавать за проверенное.

`test_joint_patches_and_bridge_in_fresh_process` (`:914+`) создаёт новый
процесс, делает двойной apply, проверяет RoPE calls, fp16 norm, packed CE
rows, TRL metrics, consumed STEP_ATTR, Rewind callback и LoRA gradients/RNG.
Это устраняет прежний P2-028-2 с зависимостью coverage от порядка suite.
Оба backend-варианта вошли в новый T4 прогон без skips/failures.

## LoRA / embedding / decoder / RoPE / NF4 installers

- `lora.py:36-97`: только certified vanilla один активный bnb adapter,
  supported CUDA dtype/layout/alignment и fp16 compute; hooks/instance
  overrides projection и неподдерживаемые variants отсекаются readiness.
  `:116-125`: base → x.float() для half ingress → тот же dropout → A/B →
  rounded epilogue. RNG-equivalence не восстанавливает fp32 значения до
  раннего округления; риск P1-023-1 не объявлен устранённым.
- `decoder.py:64-68`: input-norm owner ContextVar в finally scope;
  `:74-89`: fused residual+norm только для supported exact norm/layout,
  обе residual/norm gradient branches сохраняются. Новых cast-политик нет.
- `embedding.py:118-140`: полная lossless validation до замены storage;
  та же Parameter/tying identity, embedding gather обратно fp32,
  dense non-AMP head временно использует float32 weight. Frozen guard и
  PEFT auto-save checks исключают обновление/saving другого набора параметров.
  Никакая полная старая table не удерживается closure намеренно.
- `rope.py:25-45,70-79`: типы/strides/even D/cos-grad guards,
  PyTorch eager fallback вместо optional hub. Строка packing является
  tokenwise layout, не нарушает formula; own kernels самих производных
  находятся в отдельном ownership Codex-1.
- `nf4_patch.py:58-120`: functional/backend/linear markers отдельные;
  nested ContextVar восстанавливается через finally. Unsupported state
  явно идёт в original, compiler/CUDA errors не глотаются. Installer reload
  smoke не эквивалентен реальному reload bnb backend.

## Async offload по умолчанию: события, память и исключения

| Edge / lifetime | Обоснование в mlp.py |
| --- | --- |
| source gate/up → upload | `:147-154`: current-stream copies до produced; D2H читает persistent upload, не временный gate/up |
| upload reuse | `:151,163`: новый writer ждёт upload_free, который записан после обеих D2H copies |
| D2H → pinned host → H2D | `:155-163`, `:197-205`: host write ждёт прежний read; H2D ждёт written; read event после обеих загрузок |
| staging overwrite | `:178-200`: только free/speculative not-in-use slots; replacement ждёт used; независимые readers могут получить дополнительный slot |
| staging compute use | `:252-257`: compute.wait ready и record_stream; `.view` сохраняет exact ticket shape/rows |
| release after last read | `:482-490`: SwiGLU backward создаёт новые dgate/dup; used record после чтения gate/up и до переиспользования |
| retain/replay | `:223-250`: overlapping same-ticket begin отклоняется под lock; serial retained data остаются в leased host, final replay закрывает |
| exception cleanup | begin rollback `:239-242`, outer backward finally `:488-490`: reader/device освобождается даже при ошибке, при keep_graph=True host остаётся для replay |
| pool growth / GC | `:283-307`: retired pools держатся до lease/write/read/use completion; ticket держит свой pool, links/owners weak; final context destruction закрывает lease |

На serial steps с T<=capacity нет необходимости расширять reservation;
разный T хранится в каждом ticket и копируются только [:rows]. High-water
buffer reservation — cache, не доказанная утечка. Retained concurrent graphs
закономерно держат дополнительные host slots. Отсутствие постоянного роста
на реальном CUDA allocator проверяется stress-тестами, не выводится только
из числа Python references.

Существующие `test_mlp.py` проверяют equality/store, repeated backward,
variable packed T+partial+offload, graph discard до backward, speculative
prefetch eviction/out-of-order graphs, error/retry, growth с живым старым
graph и exact partial reservation. Новые реализации для этих проверок
не добавлялись: новое T4 выполнение уже имеющихся тестов завершилось PASS.

## Команды T4 для координатора

Все команды из корня того же checkout; сохранить stdout/stderr, exit code
и точный SHA. Не запускать несколько GPU-тестов/харнесса одновременно.
Не менять mode flags, backend policy, tolerance или caches ради прохождения.

```bash
git rev-parse HEAD
python -m pytest \
  'solution/tests/test_unpad.py::test_padded_equivalence_loss_statistics_lora_gradients_rng[lengths3-9-bnb]' \
  -vv -rs -s --tb=long
python -m pytest solution/tests/test_unpad.py solution/tests/test_mlp.py \
  -q -rs -s --tb=long
python -m pytest solution -q -rs
SOUP_APPLY_SOLUTION=1 python -m pytest tests -q
```

Полный pytest solution включает историческую ноду и все MLP stress tests;
лог полного прогона с zero skipped/failures допустим как новое подтверждение,
если SHA и состав suite совпадают. Полный harness/checks — решение
координатора после unit-логов, не GPU-запуск этого worker.

Для реального idle reload отдельный процесс (дополнительная проверка,
ещё не выполнена):

```python
import importlib
import torch
import solution
solution.apply(); solution.apply()
from transformers.models.qwen2 import modeling_qwen2 as qwen
for _ in range(3):
    qwen = importlib.reload(qwen)
    solution.apply(); solution.apply()
    assert hasattr(qwen.Qwen2ForCausalLM.forward, '_solution_unpad_original')
    assert hasattr(qwen.Qwen2Attention.forward, '_solution_unpad_original')
    assert hasattr(qwen.Qwen2RMSNorm.forward, '_solution_fused_original')
    cfg = qwen.Qwen2Config(vocab_size=127, hidden_size=64,
                          intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2,
                          use_cache=False)
    model = qwen.Qwen2ForCausalLM(cfg).cuda().train()
    ids = torch.randint(1, 127, (2, 7), device='cuda')
    mask = torch.ones_like(ids)
    labels = ids.clone()
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(input_ids=ids, labels=labels, attention_mask=mask,
                       use_cache=False)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    torch.cuda.synchronize()
    del output, model
```

Reload во время незавершённого Python forward/backward не является
проверенной/заявленной операцией; эта команда проверяет idle module reload.
Реальный PEFT/bnb backend reload и helper reload с живыми графами требуют
отдельной изолированной проверки, если такой контракт нужен потребителю.

## Изменения и оставшееся

Изменён только этот отчёт. Доказанный дефект на текущем снимке не найден,
поэтому исправления реализации/тестовых допусков не делались. Performance
backlog не пополнялся: performance предложения вне correctness-фазы.

Обязательное новое воспроизведение выполнено через координатора полным T4
unit-прогоном на b1dcb59; исторический bnb case и offload stress проходят,
Soup regression suite также проходит. Оставшиеся проверки: приведённый idle
reload настоящего Qwen/PEFT/bnb, дополнительные seeds/schedules и новый full
harness, если координатор сочтёт его нужным. Их прохождение не заявляется.
Код не менялся, отдельного повтора двухрепетичного benchmark в этой задаче нет.
