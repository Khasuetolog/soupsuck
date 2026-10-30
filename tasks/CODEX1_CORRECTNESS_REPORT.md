# S-037: аудит корректности кернелов и интеграции

Проверен HEAD `b1dcb59602ea014a6f0273233057d620d4f42dd5`, 2026-10-30/31.
Основание: `tasks/ERROR_AUDIT.md`. Исполнитель: Codex-1.

## Результат и границы доказательства

Новых воспроизведённых дефектов не обнаружено. Production-код и тесты не
изменены: оснований для исправления по контракту этой фазы нет. Допуски,
skip-условия, конфигурации launch, offload/unpad и регистрация не менялись.
CUDA/Triton execution, численные сравнения и проверка памяти не выполнялись
локально. Перед завершением координатор прислал T4 unit-прогон именно этого
HEAD; логи прочитаны исполнителем: **323 passed, 0 failed, 0 skipped** для
`pytest solution` (127.85 s), **79 passed** для Soup с патчами (9.90 s).
`status.json` содержит `verified_commit=b1dcb59602ea014a6f0273233057d620d4f42dd5`,
оба pytest exit_code=0 и `ok=true`.

Артефакты T4:

- [pytest solution](../runs_kaggle/20261030T185618838149Z_unit_b1dcb59_correctness/out/pytest_solution.log)
- [pytest Soup](../runs_kaggle/20261030T185618838149Z_unit_b1dcb59_correctness/out/pytest_soup.log)
- [status.json](../runs_kaggle/20261030T185618838149Z_unit_b1dcb59_correctness/out/status.json)

Это подтверждает существующий GPU-suite на проверенном HEAD. Дополнительный
full-V probe, sanitizer и новый полный training/harness run в этой задаче
не выполнялись. Они не входят в заявленные PASS выше.

Локально получено:

- `py_compile`: PASS для 19 файлов (три integration-модуля, восемь файлов
  `solution/kernels/*.py`, восемь назначенных тестовых файлов).
- Назначенные pytest-файлы: **17 passed, 33 skipped**, 0 failed. Существующие
  skip/importorskip сработали из-за отсутствующего torch; шесть GPU-модулей
  пропущены при collection, остальные GPU-cases — при setup. Это не счётчик
  всех CUDA-тестов, которые будут собраны на T4.
- Отдельный cold `solution.apply()` дважды: один import finder, без импорта
  torch/Triton/transformers/TRL/PEFT/Soup и без `solution.swiglu`.
- Cold `patch_soup_setup_embedding()` без Soup: штатный False, без ошибки.
- `harness.check_rules.source_kernels`: 15 own, 0 lookalikes, 0 запрещённых
  импортов, 0 ошибок разбора. Два `example_swiglu_*` остаются в kit-файле
  `solution/swiglu.py`; own-путь их не регистрирует.
- AST-константы NF4 против сохранённого исходника bnb 0.50.2
  `/private/tmp/s021-bnb-kernels.cu`: 16/16 fp32 bit patterns совпали.
  Разложение всех 256 байтов на high/low nibble также проверено.

В системном Python нет pytest. Для CPU-проверок создано отдельное окружение
`/private/tmp/s037-audit-env` с pytest; torch не устанавливался. Репозиторий
и зависимости обучения этим не менялись. Команда локального pytest:

```sh
/private/tmp/s037-audit-env/bin/python -m pytest solution/tests/test_cross_entropy.py solution/tests/test_rmsnorm.py solution/tests/test_add_rmsnorm.py solution/tests/test_lora.py solution/tests/test_nf4.py solution/tests/test_rope.py solution/tests/test_embedding.py solution/tests/test_integration.py -q -rs
```

## Проверка всех 15 @triton.jit

Ниже — статическая проверка исходника, а не результат исполнения на GPU.
Номера строк относятся к указанному HEAD. Для каждой строки «фикс» — **нет**:
воспроизведённого дефекта в целевом диапазоне не получено.

| Файл:строка / кернел | Проверенная логика, адреса и dtype | Тест для T4 |
| --- | --- | --- |
| `solution/kernels/cross_entropy.py:52` `vocabulary_statistics_kernel` | `[row, col]` использует фактический Width последнего чанка; load маскирован `col < Width`. Маскированные `-inf` заменяются нулём до `p*z`. m/s/t ремасштабируются одним новым максимумом; target-logit попадает ровно в один чанк; при равных max выбирается минимальный индекс. Статистика fp32. | `test_forward_and_arbitrary_upstream`, `test_repeat_calls_and_first_argmax_on_ties`, `test_logit_rounding_precedes_argmax_and_statistics` |
| `solution/kernels/cross_entropy.py:93` `vocabulary_outputs_kernel` | State имеет четыре fp32 элемента на строку; хвост N маскирует все чтения/записи. loss игнорируемой строки ровно 0; entropy/lse вычисляются для каждой строки. | `test_forward_and_arbitrary_upstream`, `test_empty_rows` |
| `solution/kernels/cross_entropy.py:109` `vocabulary_derivative_kernel` | dZ использует тот же Width/Start и saved lse; хвост V маскирован. Вычитание onehot происходит до масштабирования arbitrary signed g. g и gain читаются из compact-векторов той же длины. | `test_backward_compacts_exactly_the_valid_rows`, `test_grad_scaler_sized_signed_upstream`, `test_upstream_exceeding_fp16_range_has_finite_fp32_gradient` |
| `solution/kernels/rmsnorm.py:43` `rmsnorm_forward_kernel` | runtime row stride и weight stride; cols < H; статистика fp32; cast нормализации к x.dtype до умножения на weight. | `test_matches_qwen2_forward_and_gradients`, `test_runtime_strided_2d_rows_without_input_copy`, `test_half_cast_occurs_before_weight_multiply` |
| `solution/kernels/rmsnorm.py:56` `rmsnorm_input_gradient_kernel` | runtime strides x/g/w, включая broadcast upstream. u округляется к x.dtype; Jacobian использует неокруглённый a=x*rstd; padded cols дают нулевой вклад при конечном rstd. | `test_broadcast_upstream_and_weight_gradient_only`, `test_matches_qwen2_forward_and_gradients` |
| `solution/kernels/rmsnorm.py:74` `rmsnorm_weight_partials_kernel` | Двумерная маска rows<N и cols<H; partial хранится `[group,H]`. dgamma использует округлённый q; OUT_HALF относится к исходному promoted output, а не к optional output cast. | `test_fused_output_half_rounding_and_gradients`, `test_matches_qwen2_forward_and_gradients` |
| `solution/kernels/rmsnorm.py:93` `rmsnorm_weight_finalize_kernel` | GROUPS runtime, цикл по фиксированным BLOCK_R; каждая загрузка masked. fp32 суммы, один store в dtype weight; без atomics. | те же train_weight=True cases |
| `solution/kernels/add_rmsnorm.py:38` `add_rmsnorm_forward_kernel` | Только contiguous residual/delta, shape совпадает; add в fp32 с cast к promoted sum dtype до статистики. Weight-stride runtime; cols<H; сохраняется именно округлённый sum. | `test_both_outputs_and_gradients`, `test_dtype_promotion_and_cast_order` |
| `solution/kernels/add_rmsnorm.py:55` `add_rmsnorm_backward_kernel` | HAS_GS/HAS_GY защищают отсутствующие grad-указатели; WRITE_R/WRITE_D защищают output aliases-заглушки. Нормальный grad округляется к sum dtype до сложения с gs, затем приводится к dtype каждого входа. | `test_one_output_has_no_upstream`, `test_both_outputs_and_gradients` |
| `solution/kernels/swiglu_fused.py:21` `swiglu_tile_engine` | Маска по обеим осям N/I, независимые runtime strides всех буферов; gate/up/sigmoid/derivatives в fp32, выходы в dtype буфера. BACKWARD отделяет чтение da и запись dg/du от forward. | `solution/tests/test_mlp.py` целиком (MLP/autograd boundary проверяет Codex-2) |
| `solution/kernels/lora_epilogue.py:25` `lora_scale_add_kernel` | Flat contiguous buffers; Count-tail mask. Сначала adapter*scale округляется к adapter dtype, затем optional cast к base dtype, затем add; FMA отключён. | `test_epilogue_rounding_promotion_and_gradients`, `test_epilogue_scaled_half_must_round_before_add` |
| `solution/kernels/lora_epilogue.py:38` `lora_scale_gradient_kernel` | AddBackward cast к adapter dtype предшествует scale; отсутствующие градиенты не запускаются; upstream приводится к contiguous. | `test_epilogue_rounding_promotion_and_gradients` |
| `solution/kernels/nf4.py:37` `nf4_decode_kernel` | Более раннее значение в high nibble. Для i<N packed[i//2], abs[i//64], state2.absmax[i//16384] находятся в проверенных буферах. Nested scale multiply, затем offset, затем NF4 multiply; fp32, FMA выключен. | `test_nested_nf4_all_bits_match`, `test_zero_weight_sign_bits`, `test_packed_transpose_matches_bnb_shim` |
| `solution/kernels/rope.py:25` `rope_pair_forward_kernel` | Объединённые q/k heads; q и k accesses защищены взаимно исключающими масками. Cos/sin broadcast batch использует CBatches; tail tokens и half-width masked. Промежуточные products округляются перед add; FMA выключен. | `test_forward_bits_and_qk_gradients`, `test_broadcast_and_strided_cos_upstream` |
| `solution/kernels/rope.py:62` `rope_pair_backward_kernel` | Adjoint учитывает c1/c2/s1/s2 независимо; gradients cast к dtype q/k перед branch accumulation. NEED_Q/NEED_K защищают отсутствующие указатели; upstream имеет четыре независимых runtime strides. | `test_optional_gradient_and_empty_sequence`, `test_broadcast_and_strided_cos_upstream` |

### Офсеты, пустые входы, специализация

- Для реальных contiguous форм N<=2048, H=1536, I=8960: наибольший
  MLP-offset 18,350,079 элементов; RMSNorm — 3,145,727; chunk logits —
  8,388,607. Наибольший NF4 output-offset для 8960x1536 — 13,762,559.
  Все существенно меньше 2^31. Адресная арифметика относится к элементам,
  не к байтам. RoPE-views q/k используют strides исходных projection views;
  при 2048 токенах и 12 heads по 128 это также в пределах int32.
- В общем API runtime strides и program_id-derived offsets не везде явно
  расширены до int64. Экзотические as_strided-входы или размеры с offset
  >=2^31 **не сертифицированы этим аудитом**; T4-прогон их не создаёт.
  Повреждение на таких входах здесь не воспроизведено. Это ограничение
  доказательства, а не доказательство корректности произвольного размера.
- N=0: CE возвращает пустые outputs/zero dx; RMSNorm и add-RMSNorm не
  запускают row kernels, dweight при необходимости обнуляется. RoPE не
  запускает kernel при пустом batch/sequence. NF4 отклоняет пустую state.shape.
- RMSNorm принимает contiguous leading dims либо 2-D row-strided input;
  прочие leading layouts отклоняются. Add-RMSNorm отклоняет noncontiguous
  residual/delta. CE делает contiguous-копии hidden/targets и weight-чанка.
  RoPE допускает неплотные leading strides с contiguous последней осью.
- Ни один из 15 JIT не имеет N/S/leading shape в constexpr. Остаются H,
  геометрия tile, EPS, IGNORE и структурные флаги. Смена dtype/флагов и
  alignment/==1 runtime specialization может создать дополнительные variants.
  Это согласуется с сохранённым `/private/tmp/featherduster-triton-3.6.0-jit.py`;
  реальные cache-count assertions требуется повторить на T4.

## CE и интеграция

| Файл:строка | Причина проверки и вывод | Тест / остаточный риск |
| --- | --- | --- |
| `solution/autograd.py:18` | Shift в исходном `[B,S]` через pad(-100), затем selection packed positions. Последний токен строки не предсказывает начало следующей. Denominator сохраняется из исходных shifted targets либо num_items_in_batch. | `test_loss_gradients_metrics_and_rewind`, joint unpad-тесты Codex-2; packed invariant сам по себе за пределами этого файла. |
| `solution/autograd.py:33` | При all-ignore и denominator=0 сохраняется NaN loss, как в HF. CE backward предварительно исключает ignored rows, поэтому nonfinite upstream не должен загрязнять zero dx. | `test_fused_head_zero_denominator_keeps_nan_loss_and_zero_ignored_dx`, `test_one_ignored_row_with_infinite_upstream_cannot_poison_valid_gradients`. |
| `solution/kernels/cross_entropy.py:136` | Нет allocation `[N,V]`: z/dz имеют только текущий vocabulary chunk. Сохраняются hidden, weight-reference, targets, lse. Compact backward собирает только valid rows и scatter-ит dx в нули. | `test_peak_memory_real_shape`, `test_backward_compacts_exactly_the_valid_rows`. Memory/time локально не измерялись. |
| `solution/patch.py:248` `_log_metrics` | Entropy mask — attention_mask; accuracy mask — shifted label!=-100; num_tokens — attention_mask либо length position_ids. Формулы сверены с сохранённым TRL 0.29.1 `sft_trainer.py:1238-1320`. | `test_loss_gradients_metrics_and_rewind`, `test_empty_accuracy_and_distinct_masks`. |
| `solution/patch.py:293` `_patch_trainer` | Refusal сбрасывает прошлый step и использует original; fused bridge получает свежий step, dense fallback восстанавливает статистики из logits. Fused activation проверяется по logits=None и fresh step, а не по одному наличию маркера. | `test_accelerate_peft_forward_enables_amp_after_trainer_preflight`, `test_bridge_without_amp_preserves_dense_fallback`. |
| `solution/patch.py:370` `_register_module`, `solution/patch.py:462` `install`, `solution/__init__.py:16` `apply` | Cold/lazy install без heavy imports; loaded-модули патчатся немедленно, later import — до возврата import. Маркеры и один finder предотвращают stack wrappers. Fake modules отфильтровываются. | CPU cold/import-hook/idempotence tests PASS; real Qwen/TRL/bnb callbacks входят в T4 323 PASS. |
| `solution/patch.py:338` `_patch_lazy_mlp` | Trampoline сначала восстанавливает original, чтобы общий marker не блокировал установку своего MLP; повторная регистрация оставляет forward. Kit example не импортируется. | `test_lazy_mlp_registers_own_implementation_without_example`: PASS локально. |
| `solution/lora.py:100` (read-only) | A -> B, scaling и dropout остаются в локальном torch graph; epilogue не detach-ит ветку. Dropout вызывается один раз после base, исходный fp32 RNG-path восстанавливается для half ingress. | `test_lora_same_dropout_rng_and_close_forward_backward`; half ingress меняет значения на округлённые, это известный риск P1-023-1, а не bitwise guarantee. Файл не изменялся. |
| `solution/nf4_patch.py:42` (read-only) | Functional покрывает backward MatMul4Bit; scoped backend helper покрывает forward dequant-linear fallback. Unsupported state использует original; kernel/compiler exception не маскируется. | `test_patch_covers_functional_dispatcher_forward_and_backward`, `test_non_nested_functional_forward_and_backward_stay_native`. |
| `solution/tests/test_embedding.py:64` | Числа таблицы roundtrip-проверяются целиком до замены; тесты сверяют tying, значения, loss/LoRA gradients и adapter save. Обычный head без autocast поднимает weight в fp32 на лету. | GPU-тесты embedding входят в T4 323 PASS; source embedding просмотрен read-only, не менялся. |

Проверка больших/некратных N и V уже есть: N=77/1001/2048,
V=1000/4103/151936, custom ignore, all/some/none valid, arbitrary signed
upstream, GradScaler-sized upstream, повторный вызов. RMSNorm имеет H=1000
и 1536, fp16/fp32 input/weight, trainable/frozen gamma, large values,
strided/broadcast upstream и optional half output. Новые допуски не вводились.

## Открытые риски, которые этот аудит не закрывает

1. **P2-021-3 остаётся открытым:** `test_cross_entropy.py:162` сравнивает
   mm/F.linear на одинаковых chunks и использует те же online kernels для
   expected statistics. Это dispatch/rounding regression, а не независимое
   доказательство всех метрик. Независимый `_reference:65` использует полный
   V, но по 16 строк и обычно с reduced-precision reduction=False. Его форма
   GEMM отличается от реального полного head; bitwise identity полной и
   чанковой головы не утверждается. Дополнительная команда ниже проверяет
   полную голову с default reduction flag и исходными metric tolerances.
2. CE ideal-gradient oracle использует fp32 dZ и fp32 mm без TF32 при
   baseline-rounded logits; AMP dx только диагностический. Поэтому PASS
   означает соблюдение аналитического gradient norm-критерия, а не
   побитовое совпадение с half-rounded AMP dZ.
3. SwiGLU engine использует fp32 intermediate activation, тогда как eager
   F.silu(fp16) отдельно округляет activation перед multiply. Здесь требуется
   существующая численная проверка MLP; bitwise equality с eager не обещается.
4. По неограниченным входам (nonfinite logits, очень большие strides/H,
   иные модели/версии) результат этого аудита нельзя экстраполировать на
   общий Triton API. Поддерживаемый training-path ограничен guards и задачей.
5. MLP stream/event/lifetime и packed layout invariants проверяет Codex-2.
   Его GPU stress evidence и joint unpad/bnb regression нельзя заменить
   просмотром соответствующих вызовов из patch/autograd.
6. Новый unit-прогон T4 для b1dcb59 подтверждён логами (323 solution + 79
   Soup). Исторический full 10/10 из ERROR_AUDIT не повторён в этой задаче;
   новый unit-прогон не заменяет полный training/harness acceptance.

## Команды для координатора на T4

В корне репозитория с закреплёнными training-зависимостями (вторая команда
уже выполнена координатором на проверенном HEAD, 323 PASS):

```sh
python -m pytest solution/tests/test_cross_entropy.py solution/tests/test_rmsnorm.py solution/tests/test_add_rmsnorm.py solution/tests/test_lora.py solution/tests/test_nf4.py solution/tests/test_rope.py solution/tests/test_embedding.py solution/tests/test_integration.py solution/tests/test_mlp.py -q -rs
python -m pytest solution -q -rs
```

Для полного T4 окружения ожидаются нулевые skips; их причины нужно
рассматривать как отсутствие acceptance evidence, не как PASS GPU-case.

Независимая full-V проверка default backend flag (продуктовые флаги не
переопределяет; текущие METRIC_RTOL/METRIC_ATOL/ARGMAX_AGREEMENT не меняет):

```sh
python - <<'PY'
import torch
import torch.nn.functional as F
from solution.kernels.cross_entropy import fused_linear_cross_entropy
from solution.tests.test_cross_entropy import (
    _inputs, METRIC_RTOL, METRIC_ATOL, ARGMAX_AGREEMENT,
)
assert torch.cuda.is_available()
print('device:', torch.cuda.get_device_name())
print('allow_fp16_reduced_precision_reduction:',
      torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
for dtype in (torch.float16, torch.float32):
    x, w, y = _inputs(77, 1536, 151936, dtype, hidden_dtype=torch.float32)
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        z = F.linear(x.half(), w.half()).float()
        logp = F.log_softmax(z, dim=-1)
        expected = (F.cross_entropy(z, y, reduction='none'),
                    -(logp.exp() * logp).sum(-1),
                    z.argmax(-1), z.logsumexp(-1))
        actual = fused_linear_cross_entropy(x, w, y)
        for index in (0, 1, 3):
            torch.testing.assert_close(actual[index], expected[index],
                                       rtol=METRIC_RTOL, atol=METRIC_ATOL)
        assert (actual[2] == expected[2]).float().mean().item() >= ARGMAX_AGREEMENT
    del z, logp, expected, actual, x, w, y
    print(dtype, 'full-V independent metrics PASS')
PY
```

Дополнительно, если в образе доступен NVIDIA compute-sanitizer, проверить
малые/некратные входы и маски с остановкой на ошибке адресации:

```sh
compute-sanitizer --tool memcheck --error-exitcode 1 python -m pytest solution/tests/test_cross_entropy.py solution/tests/test_rmsnorm.py solution/tests/test_add_rmsnorm.py solution/tests/test_nf4.py solution/tests/test_rope.py -q -k 'empty or strided or arbitrary_upstream or nested_nf4_all_bits_match or one_output_has_no_upstream'
```

Sanitizer, full-V probe и T4-команды **не запускались** самим исполнителем;
полный pytest solution и Soup запустил координатор, результаты прочитаны выше.
Изменён только этот отчёт; performance-идей для backlog в этой фазе не добавлено.
