# S-001: аудит рантайма Soup 0.77.0

Дата: 2026-10-30. Исполнитель: Codex-2. Только статическое чтение, без запуска torch/GPU и без продуктовых изменений.

## Область доказательств

Все пути `site/...:строка` ниже относительны к **реально прочитанному** каталогу:
`<scratch>`.
В `site` распакованы soup_cli 0.77.0, transformers 5.16.1, peft 0.20.0, trl 0.29.1, accelerate 1.14.0. Это исходники, не работающая локальная GPU-инсталляция.
Пути без `site/` — файлы checkout. Для отсутствующего bitsandbytes использован официальный исходник тега 0.50.2 по ссылкам ниже; он не установлен и не добавлен в репозиторий. Номера строк GitHub — обычные, начиная с 1.

Workspace подтверждён командой `pwd`: `<repo>`.
Аккаунт и effective launch **UNVERIFIED**: `orca terminal show` и `orca orchestration check` возвращают `runtime_access_denied`, `The Codex sandbox blocked this command from connecting to Orca (EPERM)`. Политика сессии запрещает повышение прав; отправка escalation/heartbeat тоже блокируется. По названию терминала аккаунт не выводился.

## 1. Путь soup train и фактический trainer

1. `harness/_train.py:112` задаёт argv `soup train --config ... --yes`; `harness/_train.py:130` вызывает `solution.apply()` до Soup; `harness/_train.py:175` импортирует `soup_cli.cli.run`. Точные якоря вызовов можно проверить поиском имён (проверены также ниже в этом отчёте).
2. `site/soup_cli/cli.py:1023` — `run()`, `:1033` — `app()`; регистрация команды train — `:101`. Команда — `site/soup_cli/commands/train.py:565`, построение wrapper/setup — `:1946`, `:1947`, запуск wrapper.train — `:2051`.
3. `site/soup_cli/trainer/dispatch.py:20` — `build_trainer`; ветка `task == "sft"` на `:43` возвращает `SFTTrainerWrapper` на `:48`.
4. `site/soup_cli/trainer/sft.py:731` — `setup`; TRL-класс импортируется внутри setup на `:734`. Для text + transformers + stream_layers=false выбирается `_setup_transformers` на `:767`; сама реализация — `:1611`.
5. В ней загружаются tokenizer/model (`:1660`, `:1719`), применяется kbit-prep (`:1750`), выбираются LoRA targets (`:1877`), создаётся LoRA config (`:1965`), вызывается `get_peft_model` (`:1980`). `auto` делегирует известные архитектуры PEFT: `site/soup_cli/utils/peft_wiring.py:381`; Qwen2 mapping — `site/peft/utils/constants.py:101`, именно q_proj/v_proj.
6. TrainingArguments — `site/soup_cli/trainer/sft.py:950`; SFTConfig с max_length=512 — `:1083`; trainer kwargs — `:1088`. `rewind_log` по умолчанию true (`site/soup_cli/config/schema.py:3428`), поэтому обычная text SFT ветка создаёт recording trainer (`site/soup_cli/trainer/sft.py:1219`, `:1226`, `:699`). Фабрика задаёт имя `RewindSFTTrainer`: `site/soup_cli/trainer/rewind_hf.py:272`, `:371`. MRO: RewindTrainer → trl.SFTTrainer → transformers.Trainer.
7. Wrapper.train — `site/soup_cli/trainer/sft.py:2388`; перед train выравнивает dtype trainable LoRA (`:2521`), вызывает `self.trainer.train` (`:2528`), затем закрывает rewind и сохраняет модель (`:2530`, `:2537`).

**stream_loss_chunk не активен.** Его default=None (`site/soup_cli/config/schema.py:3915`); task/task.yaml его не задаёт. Hook устанавливается streaming setup только при заданном chunk (`site/soup_cli/trainer/stream_setup.py:1544`); default hook=None (`:798`). Только при наличии hook setup выбирает `make_chunked_loss_trainer_class` (`site/soup_cli/trainer/sft.py:1115`). Нельзя включать этот config switch ради ускорения: правила фиксируют настройки задачи.

Однако этот файл полезен как проверенный образец интерфейса: `site/soup_cli/utils/stream_loss_chunk.py:30` — ChunkedStep(token_entropy, token_loss, correct, labelled); `:186` — ChunkedHeadLoss, который временно удерживает head и делает no-logits output; `:301` — trainer, обходящий TRL compute_loss и восстанавливающий метрики. Он **не доказывает** численную идентичность raw-head metrics базовому AMP-пути (см. пункты 3/5).

## 2. compute_loss, сдвиг, нормировка, метрики

### Полная цепочка

Soup: `site/soup_cli/trainer/rewind_hf.py:307` сохраняет labels **до** super (`:337`), запрашивает `return_outputs=True` (`:338`), передаёт num_items_in_batch. Recorder работает только при рабочем state, model.training, наличии labels и отсутствии shift_labels (`:321`).

TRL: `site/trl/trainer/sft_trainer.py:1200` выбирает mode по self.model.training, удаляет `_prediction_loss_only`, сохраняет labels, ставит `inputs['use_cache']=False` (`:1209`), вызывает super с return_outputs и num_items (`:1234`). Liger-only kwargs skip_logits/return_token_accuracy/use_token_scaling (`:1213`) здесь не активны. Обычный путь не задаёт logits_to_keep.

Transformers: `site/transformers/trainer.py:1974`; при отсутствии smoother/custom loss оставляет labels в inputs (`:2005`), добавляет num_items при model_accepts_loss_kwargs (`:2009`), затем `model(**inputs)` (`:2014`). Для обычной колляции входы input_ids, labels, attention_mask плюс use_cache=False и num_items_in_batch. Qwen forward имеет **kwargs, поэтому определение model_accepts_loss_kwargs на `:496` даёт true (если явный accepts_loss_kwargs не переопределён). Возвращается outputs['loss'] (`:2047`). Межпроцессный scale на `:2049` при одном GPU равен 1.

Qwen: `site/transformers/models/qwen2/modeling_qwen2.py:423` → self.model (`:452`) → lm_head (`:465`) → self.loss_function (`:469`). Default logits_to_keep=0 означает `slice(0,None)`, т.е. **все** позиции (`:432`, `:464`), а не ноль логитов. Loss dispatcher `site/transformers/modeling_utils.py:4700` выбирает ForCausalLM; сама функция — `site/transformers/loss/loss_utils.py:49`.

### Training loss

Пусть `y[b,t]` — collator labels, `z[b,t,v]` — logits головы; `q[b,t]=y[b,t+1]` для t<S-1, `q[b,S-1]=-100`; `m=(q!=-100)`.
ForCausalLMLoss сначала делает z.float (`site/transformers/loss/loss_utils.py:59`), pad labels справа -100 и shift (`:63`), flatten (`:67`) и fixed_cross_entropy (`:70`).
`L = sum_{b,t:m} CE(float(z[b,t]),q[b,t]) / D`.
Если num_items_in_batch отсутствует, D — число неигнорированных токенов (reduction=mean); иначе reduction=sum и деление на переданный num_items (`site/transformers/loss/loss_utils.py:32`). Это не среднее из средних по строкам.

Trainer считает num_items по всем microbatch текущего accumulation window: `site/transformers/trainer.py:2150`, `:2174`. Для causal loss считает `labels[...,1:]`, а не всю labels (`:2177`); детекция causal loss — `:517`, `:525`. Для этого task gradient_accumulation_steps=1, один GPU: D=sum(labels[:,1:]!=-100), т.е. среднее по supervised токенам всего batch. При all-ignored D=0 исходник не содержит clamp: не обещать finite 0 вместо исходного NaN без отдельного решения.

### Entropy

`site/trl/trainer/sft_trainer.py:1239` считает под **torch.no_grad**:
`H[b,t] = -sum_v exp(log_softmax(Z[b,t])) * log_softmax(Z[b,t])`.
`Z` здесь — outputs.logits, после внешней AMP-обёртки Accelerate, то есть fp32 в целевом запуске (пункт 5).
Функция `site/trl/trainer/utils.py:572` flatten leading dims, разбивает N=B*S на чанки **128 позиций** (`:600`), log_softmax без явного dtype (`:601`), exp/multiply/sum (`:602`), concatenate/reshape (`:605`). Сама функция не содержит no_grad; его задаёт вызывающий trainer.

`entropy = sum(H * attention_mask) / sum(attention_mask)`; затем `gather_for_metrics(entropy).mean().item()` (`site/trl/trainer/sft_trainer.py:1249`). Использует **все непаддинговые позиции**, включая prompt, assistant и последнюю позицию, **без causal shift и без маски labels**. Padding-free fallback: mean по всем H при position_ids (`:1252`); не активен здесь. Виртуальные токены исключаются только для prompt tuning (`:1244`), LoRA их не добавляет.

### Accuracy и num_tokens

`site/trl/trainer/sft_trainer.py:1284` — тоже no_grad. Argmax logits[..., :-1, :] (`:1292`, `:1303`) сравнивается с labels[...,1:], mask labels!=-100 (`:1306`). Суммы correct и total собираются по процессам, `accuracy=sum(correct)/sum(total)` или **0.0 при total=0** (`:1313`). Последняя позиция logits не участвует. `shift_labels`/virtual-token ветки предусмотрены, но не нужны этому конфигу.

`num_tokens` — другой счётчик: в train накапливает **все** attention_mask.sum(), включая prompt, не только supervised tokens (`site/trl/trainer/sft_trainer.py:1259`). В eval счётчик не увеличивает, но метрика остаётся. Сохранить все три имени метрик и периодичность логирования.

## 3. Rewind per-row loss и смысл dtype

`site/soup_cli/trainer/rewind_hf.py:338` получает те же outputs, что TRL, без второго forward. `:345` выбирает outputs.logits; `row_losses` вызывается на `:350`, запись microbatch на `:362` с step=global_step+1 и прежней FIFO идентичностью строк.

`row_losses` (`:183`) под no_grad (`:205`) для каждой строки делает:
`u[b,t] = cross_entropy(Z[b,t,:], y[b,t+1], reduction='none', ignore_index=-100)` при 0<=t<S-1 (`:212`),
затем **только результат CE** приводит в fp32 (`:217`), умножает на mask и суммирует (`:218`),
`row_tokens[b] = sum_t(y[b,t+1]!=-100)`,
`row_loss[b] = sum_t(u[b,t]*mask) / max(row_tokens[b],1)` (`:221`).
Пустая строка даёт 0.0 и 0. Это статистика plain CE независимо от custom training loss.

Фраза «в dtype головы и только потом cast» буквально верна для `_chunk_terms`: `site/soup_cli/utils/stream_loss_chunk.py:59` получает raw head logits; `:62` делает log_softmax, `:67` gather и лишь `:68` .float; training CE отдельно upcast на `:72`. Но **обычный row_losses работает в dtype переданных outputs.logits**, не обязательно raw head. Accelerate преобразует fp16 logits в fp32 перед возвратом TRL/Soup (`site/accelerate/accelerator.py:1818`; `site/accelerate/utils/operations.py:889`). Значит в целевом baseline rewind CE и entropy вычисляются по fp32-копии **уже округлённых fp16 логитов головы**. Нельзя просто повторить комментарий `_chunk_terms` и считать, что rewind ожидает fp16 log_softmax.

No-logits совместимый интерфейс уже есть: `site/soup_cli/utils/stream_loss_chunk.py:298` задаёт STEP_ATTR="_soup_chunked_step"; Rewind читает импортированный атрибут на `site/soup_cli/trainer/rewind_hf.py:348`. При logits=None использует chunked.token_loss (`:351`) и `row_losses_from_token_losses` (`:168`): ожидается [B,S], target t+1, **последний столбец 0**, sum fp32 и те же row_tokens из labels[:,1:]. Нужно устанавливать свежий step на каждом forward, не оставлять данные предыдущего batch.

## 4. Qwen2 forward и рекомендуемая точка перехвата

`site/transformers/models/qwen2/modeling_qwen2.py:406` — Qwen2ForCausalLM, bias-free lm_head на `:416`. Decoder возвращает last_hidden_state после final norm (`:398`); forward головы читает его (`:462`) и **всегда** вызывает lm_head (`:465`), даже если labels=None. Loss_function вызывается только при labels (`:468`). Возвращается CausalLMOutputWithPast(loss, logits, past_key_values, hidden_states, attentions) (`:471`).

logits_to_keep>0 сокращает позиции, но всё равно материализует выбранные logits и не сохраняет entropy prompt/last и supervised alignment без дополнительных действий. output_hidden_states=True добавляет состояния, но не отключает голову. Обычный forward hook после lm_head срабатывает уже после аллокации; pre-hook сам по себе не отменяет Linear.

**Одна рекомендуемая точка: узкий patch `Qwen2ForCausalLM.forward` на границе `outputs.last_hidden_state` → `lm_head`, строки 462–465.** В поддерживаемом labelled training forward выполнить прежний self.model с теми же kwargs, затем fused head+CE+statistics на его last_hidden_state; вернуть обычный CausalLMOutputWithPast с loss и без logits, а небольшую статистику передать trainer. Это сохраняет decoder, PEFT LoRA, attention mask и кэш-поля и предотвращает обе большие аллокации (fp16 head logits и fp32 AMP copy).

Соответствующий trainer bridge должен пропустить именно TRL часть, читающую logits, вызвать `transformers.Trainer.compute_loss` и воспроизвести метрики, как `site/soup_cli/utils/stream_loss_chunk.py:312`–`:339`; после этого Rewind может использовать STEP_ATTR. Перехват только loss_function слишком поздний: logits уже созданы. Нельзя включать use_liger_kernel ради обхода метрик, это меняет проверяемые settings. Patch forward должен сохранить исходную сигнатуру/**kwargs и fallback для generation/eval/no-label/unsupported config. Не подменять весь PEFT model.forward и не обходить его хуки.

## 5. Dtypes, RMSNorm, AMP

`site/peft/utils/other.py:186` замораживает **все** существующие параметры; `:198` переводит fp16/bf16 параметры в fp32, кроме Params4bit (`:201`). Docstring про output embedding require_grads не заменяет этот исполняемый код. При gradient_checkpointing=False input-requires-grad hook из `:221` не устанавливается. Затем PEFT добавляет LoRA A/B; adapter dtype promotion fp16/bf16→fp32 документирован и вызывается в `site/peft/tuners/tuners_utils.py:590`. Soup дополнительно исправляет bf16 trainable lora_* перед fp16 train (`site/soup_cli/utils/mixed_precision.py:85`, `:109`; вызов `site/soup_cli/trainer/sft.py:2521`).

lm_head и embed_tokens — обычные параметры, после kbit-prep fp32 и requires_grad=False. Связь задана `_tied_weights_keys` (`site/transformers/models/qwen2/modeling_qwen2.py:407`), но **реальное tying зависит от checkpoint config.tie_word_embeddings**: дефолт Qwen2Config false (`site/transformers/models/qwen2/configuration_qwen2.py:72`). README kit подтверждает tied для выбранной базы, но локального config.json базы нет: фактический alias **ASSUMED, проверить `head.weight is embed_tokens.weight` на T4**. Не выводить alias только из mapping класса. Заморозка обоих сохраняется при targets q/v.

Qwen2RMSNorm (`site/transformers/models/qwen2/modeling_qwen2.py:238`): weight=Parameter(ones(H)) (`:244`), eps из config (`:331`), default конструктора 1e-6 (`:239`). Алгоритм: запомнить input dtype; x.float(); variance=mean(x²) в fp32; x*=rsqrt(variance+eps); **нормализованную x привести к input dtype, затем умножить на weight** (`:247`–`:252`). Нельзя перенести последнее приведение после умножения без изменения округления.

С fp32 weight результат pointwise multiply при fp16 или fp32 input — fp32 (вывод из стандартного torch type promotion; фактический GPU trace UNVERIFIED). Frozen norm weight requires_grad=False, но градиент по input нужен для LoRA. В decoder residual stream начинается fp32 от embeddings (`:355`), final norm выдаёт fp32 (`:398`), поэтому **реальный hidden на входе lm_head.forward ожидается fp32**, даже при fp16 обучении. Контракт ядра `hidden fp16` должен содержать явный differentiable cast hidden→fp16; backward cast обратно в исходный fp32 обеспечивается autograd. Weight хранится fp32, вычисление головы под autocast использует fp16, logits округляются в fp16.

На T4 Soup выбирает fp16/bf16 flags через `site/soup_cli/trainer/sft.py:1495`, training kwargs `:964`; T4 не поддерживает bf16. Accelerate при fp16 включает native_amp и GradScaler (`site/accelerate/accelerator.py:564`, `:583`), оборачивает model.forward autocast + convert_outputs_to_fp32 (`:1818`). CUDA autocast dtype=float16 — `site/accelerate/utils/modeling.py:2087`. Рекурсивное преобразование **выходов** fp16/bf16→fp32 — `site/accelerate/utils/operations.py:901`, `:930`. Оно не меняет dtype внутренних hidden до головы.

Для fused CE нужно воспроизвести `z_ref = float32(round_fp16(matmul(fp16(hidden), fp16(weight))))`, затем fp32 softmax/CE/statistics. Простое fp32 накопление dot и непосредственный softmax до fp16 округления — иной numerical path. Точное совпадение cuBLAS/Triton матмулов и quality UNVERIFIED. Backward обязан применять upstream grad_loss (включая GradScaler), а не предполагать scale=1; статистика detached. Не вычислять backward с маленькими unscaled fp16 значениями заранее, как streaming-образец, без проверки underflow.

## 6. PEFT Linear4bit + LoRA и dequant

`site/peft/tuners/lora/bnb.py:513`: проверить args, убрать adapter_names/variant kwargs; disabled/merged ветки вызывают base_layer (`:518`); обычная ветка base_layer(x), затем clone (`:527`, `:533`). Для каждого active adapter — A, B, dropout, scaling (`:535`). Если **autocast выключен**, запомнить result.dtype и привести x к A.weight.dtype (`:543`). Vanilla: `B(A(dropout(x))) * scaling` (`:549`); без autocast привести output к result.dtype (`:550`); затем result+output (`:552`). При включённом autocast эти явные casts пропускаются. Нельзя переставлять dropout относительно A, делить один dropout mask между разными адаптерами или менять RNG порядок.
Scaling обычного LoRA alpha/r=32/16=2, не alpha/sqrt(r): `site/peft/tuners/lora/layer.py:274`–`:277`. Dropout default 0.05 — `site/soup_cli/config/schema.py:91`.

PEFT vanilla forward **не вызывает dequantize_bnb_weight сам**: это работа base_layer. PEFT helper dequantize для merge/unmerge вызывает `bnb.functional.dequantize_4bit(weight.data, weight.quant_state)` (`site/peft/utils/integrations.py:92`, `:102`; merge `site/peft/tuners/lora/bnb.py:384`). Не путать merge с каждым training forward.

Официальный тег bnb 0.50.2, не локальная инсталляция:
- [bitsandbytes/nn/modules.py:533,565–591](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/nn/modules.py#L565): первый вход fp32 может переключить compute_dtype в fp32; forward приводит вход к compute dtype, вызывает matmul_4bit, возвращает input dtype. Поэтому заданный bnb_4bit_compute_dtype=fp16 сам по себе не гарантирует каждый внутренний путь fp16.
- [bitsandbytes/autograd/_functions.py:274,294,348–351](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/autograd/_functions.py#L274): forward использует bitsandbytes::gemm_4bit, backward по A явно dequantize_4bit(B,state)→grad_output.dtype и matmul. dB отсутствует.
- [bitsandbytes/backends/cuda/ops.py:778,830,845–902](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.50.2/bitsandbytes/backends/cuda/ops.py#L830): custom fused GEMM либо dequant+F.linear fallback; при M>1536 CUDA выбирает fallback. В нём nested absmax сначала деквантуется, затем 4bit W в A.dtype. Для B=4,S=512 M=2048; для коротких батчей выбор зависит от формы/устройства. Фактическая выбранная ветка на T4 UNVERIFIED.

## 7. Тесты с фейками и guards apply()

`tests/conftest.py:15`–`:19` вызывает apply при SOUP_APPLY_SOLUTION=1. `harness/check_rules.py:242` запускает pytest tests с patches; ни один GPU тест из solution этим не заменяется.

`tests/test_trainer_init.py:18` проверяет SFTTrainerWrapper constructor. `:58`–`:166` создаёт _Model **не nn.Module**, config=SimpleNamespace, parameters()=[], fake transformers/peft (`:113`, `:121`), wrapper через object.__new__ (`:156`) и прямо вызывает `_setup_transformers` (`:163`). Другие тесты повторяют это для vision (`:169`), audio (`:267`) и DPO/GRPO/IPO/KTO/BCO/ORPO/SimPO (`:385` и далее). Fake modules заменяют sys.modules и не обязаны иметь Qwen classes, Trainer, __spec__ или version. Tests callback вызывают SoupTrainerCallback с MagicMock state/args (`tests/test_callback.py:10`); config/loader/formats проверяют чистую подготовку данных. Реальный model forward, TRL metrics и rewind numerical path эта регрессия не доказывает.

Обязательные guard решения:
- Идемпотентный apply: не оборачивать классы/методы повторно; сохранять original для fallback, не кешировать fake model как реальный.
- Setup wrapper сначала вызывает **оригинальный** `_setup_transformers`; после него ранний return если self.model не torch.nn.Module. Не обходить оригинал при fake, иначе сломаются проверяемые вызовы resize/tokenizer.
- Ленивые импорты и проверка доступности реальных Qwen/TRL классов: fake transformers — SimpleNamespace, его нельзя считать пакетом. Узкая обработка ImportError/AttributeError при регистрации; не скрывать ошибки настоящего CUDA forward общим except.
- Fast path: конкретный Qwen2 causal model/real tensors/CUDA, supported dtypes/shape, bias-free frozen dense head, labels, нулевые virtual tokens, отсутствие unsupported loss options/softcap/shift_labels/parallelism; иначе original.
- Trainer bridge активировать только при успешной регистрации данного model fast path; cpu/fake/unsupported/no labels используют original. Сохранять сигнатуру forward и **kwargs для num_items и HF introspection.
- Обычные generation/no-label и внешняя eval не должны требовать side-channel статистику. `harness/eval_adapter.py:1` явно не импортирует solution и проверяет стандартный saved LoRA.

## 8. Формы и config

| Величина | Значение / степень подтверждения |
|---|---|
| B | 4 (`task/task.yaml:13`); без смешивания строк между batch |
| S | <=512 (`task/task.yaml:9`), right-pad до max длины **этого batch**, не обязательно 512 |
| N | B*S; для loss последняя позиция каждой строки ignored, но для entropy нужна |
| H | **ASSUMED 1536** для Qwen2.5-1.5B-Instruct; config.json checkpoint локально отсутствует |
| V | 151936 в README kit и default Qwen2Config (`site/transformers/models/qwen2/configuration_qwen2.py:61`); фактическую shape head проверить при setup |
| Слои | **ASSUMED 28** для выбранной базы, не выводить из общего default 32 |
| Intermediate | **ASSUMED 8960**, не общий default 22016 |
| heads / KV heads | **ASSUMED 12 / 2**, не defaults 32 / 32 |
| eps | **ASSUMED 1e-6** у checkpoint; подтверждён default `configuration_qwen2.py:70`, передаётся каждому norm из config (`modeling_qwen2.py:331`, `:266`) |
| tied head | ASSUMED true по README kit; реальное alias требуется проверить; default Qwen2Config false |

Padding доказан `site/trl/trainer/sft_trainer.py:199`: pad right для ids, labels=-100, attention_mask=0; helper max output_shape — `site/trl/trainer/utils.py:114`–`:159`. `pad_to_multiple_of=None` (`site/trl/trainer/sft_config.py:204`), padding_free=False (`:194`). Soup готовит assistant-only pretokenized labels: `site/soup_cli/data/sft_format.py:5`, `:95`; setup выбирает formatter на `site/soup_cli/trainer/sft.py:898`. TRL сохраняет labels, если они уже существуют (`site/trl/trainer/sft_trainer.py:167`); дополнительно применяет assistant mask (`:223`). Поэтому не строить labels заново из attention_mask и не считать completion_only_loss единственным источником answer mask.

Общий Qwen2Config **не** config Qwen2.5-1.5B: `site/transformers/models/qwen2/configuration_qwen2.py:61` задаёт H4096/32 слоя. Жёстко подставлять их в ядро нельзя. Размеры брать из реальной head.weight/hidden/config.

## Контракт для интеграции

Для **каждой** позиции [b,t], включая prompt и t=S-1, fused head должен получать тот же hidden, который пошёл бы в lm_head (после финальной RMSNorm); явно учитывать fp16 autocast и fp16 округление logits перед fp32 CE/statistics. Никакой [B,S,V] аллокации.

1. Targets: pad(labels,(0,1),-100)[...,1:], flatten [N]. Последний target каждой строки -100. Не резать всю sequence до S-1, если тем самым теряется entropy последнего непаддингового токена.
2. `per_token_loss[N] fp32`: CE по fp32-представлению fp16-rounded logits, 0 для target=-100; differentiable по hidden, frozen weight без dW. Gradient loss = sum(per_token_loss)/num_items_in_batch, или mean по valid targets при None. Upstream GradScaler multiplier применяется в backward. D берётся от Trainer, не заменяется локальной константой batch_size.
3. `entropy[N] fp32`, `argmax[N] int64`, `lse[N] fp32`: detached. Entropy — по **всем** позициям независимо от targets; argmax по всему V, tie behaviour как torch.argmax (первый индекс). LSE внутренний, имя метрики не добавлять.
4. Trainer entropy: sum(entropy.reshape(B,S)*attention_mask)/attention_mask.sum(), gather(mean). Accuracy: argmax[:, :-1] против labels[:,1:], valid mask, correct/valid после gather; 0 если valid=0. Cumulative num_tokens += attention_mask.sum() в train. Это три разных маски/счётчика.
5. Rewind: fresh ChunkedStep-compatible `.token_loss[B,S]`, detached, последний столбец 0; `.token_entropy[B,S]`, `.correct` и `.labelled` scalar. Использовать **тот же STEP_ATTR**, который читает Rewind (`site/soup_cli/utils/stream_loss_chunk.py:298` и `site/soup_cli/trainer/rewind_hf.py:55`). Existing `row_losses_from_token_losses` даст means/counts. Не менять sampler/FIFO/on_micro_batch.
6. Вернуть model output с loss и logits=None, остальными ожидаемыми полями. Trainer bridge обязан восстановить TRL metrics и передать свежий step до возврата в Rewind. Preserve args/settings, return_outputs и fallback; готовая фабрика Soup показывает организацию bridge, но её raw-head statistics требуют корректировки для baseline AMP.
7. RMSNorm контракт отдельно: fp32 variance+rsqrt, cast нормализованного x **до** умножения на fp32 weight; вернуть dtype torch promotion (здесь fp32), градиент по input; frozen weight не требует dgamma. eps брать с instance. Выход всегда fp16 нарушил бы baseline.

## Риски

- Главный numerical риск: пропуск round_fp16 logits между dot и CE; fp32 logits dot сами по себе не эквивалентны baseline. Нужно сравнить loss/grad/entropy/accuracy на T4, затем checks 5–7 и held-out eval.
- Hidden перед головой ожидается fp32; плановый kernel принимает fp16. Нужен differentiable cast и проверка промежуточных dtypes; без GPU этот вывод остаётся статическим, не измеренным.
- Existing streaming helper считает entropy/rewind raw-head log_softmax; его комментарии не учитывают external convert_outputs_to_fp32 baseline. Не копировать формулы dtype механически.
- Tied alias и H/layers/eps checkpoint не подтверждены локальным config.json. Runtime guards/shape inspection обязательны; config defaults другой модели не подходят.
- All-ignored batch: baseline training CE может NaN, accuracy=0 и row mean=0. Это три разных соглашения, не менять их неявно.
- Внешний AMP GradScaler, сигнатура forward и model_accepts_loss_kwargs чувствительны к patch. Потеря **kwargs может изменить D/градиент при accumulation.
- Не сбрасывать last_step означает запись статистики предыдущего batch. Исключения recorder могут отключить rewind вместо падения train; успешный exit не доказывает check 5.
- No-logits output ломает неизменённый TRL. Bridge обязателен; class patch должен учитывать динамическое создание Rewind subclass в setup.
- LoRA dropout и RNG нельзя менять. Frozen base не означает отсутствие градиента по hidden/norm input; eval без solution проверит реальные adapter weights.
- bnb 0.50.2 имеет custom GEMM и fallback по M/dtype/arch; утверждение «каждый forward всегда dequant+linear fp16» неверно. Профиль нужен на реальном pinned runtime.
- Здесь **не запускались** 79 regression tests или GPU tests: нет исполняемого torch/triton/GPU. Acceptance отчёта проверяется статически; аккаунт/Orca settlement остаются UNVERIFIED из-за EPERM.
