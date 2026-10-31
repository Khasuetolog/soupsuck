# Codex-5: проверка README, SUBMISSION и публичного notebook draft

Дата: 2026-10-31, 16:10 UTC. Назначенный snapshot:
`c3781d468b7412a7aa4bf91caf9ca9f53dae696b`.
README.md (516 строк) и SUBMISSION.md (62 строки) прочитаны полностью.
Рабочий HEAD менялся другими исполнителями; проверенные README, SUBMISSION,
template notebook и best_config.json побайтово совпадали с назначенным snapshot.
Изменён только этот отчёт. Commit, push, публикация и GPU-запуски не выполнялись.

**Вердикт:** итоговые измерения подтверждены исходными артефактами, однако
submission пока нельзя воспроизвести буквально без заполнения ссылок/SHA и
уточнения финального notebook. Ниже перечислены предложения, а не внесённые правки.

## Предлагаемые исправления

| ID / приоритет | Место | Найдено и предлагаемая правка |
| --- | --- | --- |
| R01 / P0 | README:30–31, 346, 352–354; SUBMISSION:1, 5–7, 13, 42, 50; draft cell 1 | Остались `<KAGGLE_LINK>`, `<FINAL_COMMIT>`, team/contact placeholders; draft содержит `COMMIT = "__FINAL_COMMIT__"`. `git checkout <FINAL_COMMIT>` даже не проходит `bash -n`: угловые скобки интерпретируются оболочкой. Перед сдачей вставить утверждённый полный SHA, настоящий saved-version URL и согласованные данные команды; сверить SHA в notebook и обеих документах. Не угадывать состав/контакты команды. |
| R02 / P1 | README:345–354, 402–406; SUBMISSION:28–30 | Доказательства финального full2 обозначены как local/not in Git, а публичная ссылка ещё отсутствует. Указанного final evidence directory нет в этом checkout; raw-файлы найдены в другом scratch checkout. Анонимное открытие адреса репозитория из draft вернуло HTTP 404: публичный доступ не подтверждён, что совместимо с приватным репозиторием. Дать жюри доступ к source и работающую ссылку на сохранённую версию notebook с result/checks/metrics/logs; проверить доступ без авторизации владельца. Не коммитить runs_kaggle или credentials. |
| R03 / P1 | README:349–351; templates/kaggle_notebook.ipynb и draft, cell 5 | Оба notebook запускают `python harness/run_task.py`, то есть default `--repeats 1`. README правильно требует ручную замену на `--repeats 2`, но текущий draft при обычном Run All не воспроизводит двухповторную финальную процедуру. Перед финальным сохранением выполнить явно описанную в README замену и проверить `result.task.repeats == 2`, две base и две solution записи. Если сохраняется требование «draft отличается от template только REPO_URL/COMMIT», изменение training cell нужно согласованно внести в оба notebook, либо отдельно зафиксировать разрешённую финальную правку. |
| R04 / P1 | README:358–375; SUBMISSION:39 | Установка фиксирует шесть train-пакетов, но не создаёт описанный Python/CUDA/torch/Triton runtime и не фиксирует datasets. Доказательства используют Python 3.13.15, torch 2.11.0+cu128, Triton 3.6.0, datasets 4.8.5; backward требует CUDA `torch.mm(..., out_dtype=torch.float32)`. Для чистой Linux-машины добавить проверенную инструкцию создания среды/образ либо явное предварительное требование именно этой среды, с проверкой версий до запуска. SUBMISSION должен прямо ссылаться на installation block README. Подходящий wheel/index не проверялся, поэтому команда установки CUDA wheel здесь не выдумывается. |
| R05 / P1 | README:7–22, 60, 478–479; SUBMISSION:17–29 | Все quoted GPU/test результаты относятся к `96a890c57a6c9dd28ca92649e27a7a992aa3de51`, а текущий source содержит более поздние изменения в solution/mlp.py и solution/precompile.py, главным образом opt-in EXP-013/015. Текст честно называет измеренный SHA; важно не заполнить `<FINAL_COMMIT>` новым SHA и представить старые проверки проверками нового кода. Зафиксировать границу измеренного source и final SHA; при выборе более нового solution провести требуемую финальную проверку этого SHA. Этот аудит не заменяет такую проверку. |
| R06 / P2 | README:436; experiments/best_config.json:promotion_rule | Строка EXP-001/004 верно переносит wall delta −1.80 s и CI [−5.96, −1.60], но peak delta в той же paired wave2 равна **+4 MiB**, а не 0: A=3765, E=3769, три пары, CI памяти [4, 4]. Заменить 0 на +4 и указать три пары/source d5dc84b. Также согласовать формулировку promotion policy: README говорит «не увеличивает память», best_config требует строго меньший peak, но EXP-010 принят при одинаковых 3769 MiB. Нельзя менять фактические числа ради соответствия policy. |
| R07 / P2 | README:36–50; SUBMISSION:61–62 | Start-probe `f81a736` действительно содержит приблизительно приведённые времена, но его `start_probe.json` имеет **ok=false**, `missing_phases=["import.bitsandbytes"]`; внешний status тоже не является успешной полной проверкой probe. Явно назвать разбиение частичным наблюдением и привести это ограничение либо повторить исправленный probe. «Quantization 2.2 s» — span между первой/последней quantization event (~2.165 s), тогда как union вызовов ~0.361 s; указать, что именно измерено. Утверждение о неизменяемых ~24 s каждого запуска сильнее evidence: наблюдалось 23.965 s до первого шага именно в этом instrumented run; финальный full2 имеет wall−loop 25.77/25.87 s. Сформулировать наблюдение без обещания фиксированного нижнего предела. |
| R08 / P2 | README:24–28, 427–450; SUBMISSION:33–35 | «Every optimization decision» / «measurements for every optimization» не соответствует самой таблице: EXP-002 — synthetic GEMM microbench, EXP-011 — launch sweep/оценка, EXP-007 вообще не измерен. 4123 MiB действительно есть в старом recompute full run ce1ff09; +20 s получается из **разных Kaggle sessions**, 93.95−73.54=20.41 s, и не является paired A/B. Уточнить область paired claims и пометить старое recompute сравнение как historical cross-session observation. |
| R09 / P2 | README:374 | `E/runs/result.json:versions` не является существующим repo path, и E нигде не определён. Заменить на `runs/result.json` в опубликованных output-файлах финальной notebook version и определить ссылку/путь к evidence. Числа runtime сами по себе совпадают с raw versions. |
| R10 / P2 | experiments/reports/EXP-001, 004, 006, 008–015; experiments/registry.jsonl; README:429–445 | Отчёты остались историческими worker handoff: EXP-001:234 говорит GPU UNVERIFIED, EXP-004:167–170 не имеет A/B решения, EXP-006:227 — pending, EXP-008:183 и EXP-009:158–159 — UNVERIFIED, EXP-010:171 — BEST не менялся, EXP-011:155 — winners UNVERIFIED, EXP-012:3–7 — counts NOT RUN, EXP-013:160 и EXP-015 — GPU pending. Между тем README уже цитирует последующие результаты. В tracked registry только 23 записи двух старых source SHA (6e62dbf/d5dc84b), без следующих A/B. Добавить к соответствующим отчётам краткие финальные evidence addenda с source SHA, командой, реальным outcome и публичным артефактом; поздние registry данные переносить только после удаления личных идентификаторов. Исторические ошибки не стирать. |
| R11 / P3 | README:24–25; SUBMISSION:32–33 | Диапазон base 77.7–91.4 s соответствует выбранным поздним runs (77.66 и 91.37), но не всему доступному evidence: tracked registry содержит base 76.96 s, старый recompute full — 97.75 s. Указать область «selected recent sessions» либо привести полный диапазон с явно выбранной группой runs. |
| R12 / P3 | README:250, 423, 494 | «Offloads ... (next section)» указывает на следующий раздел NF4, тогда как offload описан выше: заменить на ссылку на Asynchronous MLP activation offload. Total 4768.1 ms — сумма уже округлённых строк; raw total 4767.985 ms округляется в 4768.0: можно пояснить метод округления. В строке measurement environment лучше явно написать «no warm compiler cache»; сейчас отрицание в длинном перечне допускает двусмысленное чтение. |

## Подтверждённые числовые данные

Исходные GPU-логи **прочитаны**, новые GPU-прогоны не выполнялись. Ниже пути
относительны каталогу evidence, личные абсолютные пути намеренно опущены.

`F = runs_kaggle/20261031T133425573730Z_full2_96a890c_promote010/out/`.
В этом checkout F отсутствует; проверена копия F в предоставленном scratch tree.
`F/status.json`: ok=true, expected_commit=verified_commit=`96a890c57a6c9dd28ca92649e27a7a992aa3de51`,
exit codes gpu/pip/prepare_data/pytest_solution/harness/checks/score равны 0.
SHA256 `F/runs/result.json`:
`a90cb45a30f30d7775e054ea64e268ef33cf82639ac0d11ac8cc894609612016`.
`F/runs/checks.json.result_sha256` совпадает с этим hash; все 10 check.ok=true.

| Заявление README/SUBMISSION | Результат сверки |
| --- | --- |
| Wall base 91.37/83.85, mean 87.61 s; solution 65.11/65.87, mean 65.49 s | Совпадает с F/runs/result.json и best_config.json. Отношение 1.33776149 → 1.338. |
| Loop 60.75/39.67 s, 1.53× | Совпадает: base 58.99/62.51, solution 39.34/40.00; ratio 1.53138. |
| Median step 0.94–0.99 / 0.51–0.52 s | Совпадает: 0.9407/0.9857 и 0.5090/0.5208. |
| Peak 14345/3769 MiB; ratio 3.806 | Совпадает; ratio 3.80604935. |
| Eval loss 2.04966/2.04964, −0.001% | Совпадает с summary/evals и check 7. |
| Own share min 13.3%, check 4; loss first 0.02%, mean 0.01%, check 6 | Подтверждено F/checks.log и checks.json. |
| 518 passed, 1 skipped; Soup 79/79 | Подтверждено F/pytest_solution.log, F/checks.log и отдельным unit_96a890c_promote010: 518/1 и 79. Skip объяснён rejected EXP-008. Это результаты старого измеренного SHA, не новый запуск аудитора. |
| 13 строк kernel table, включая ms и обе доли | Все 13/13 строк совпадают с metrics. Timed loop raw 35.368/35.994 s → 35.37/35.99. RoPE суммарно ~0.92%; NF4 ~7.0%. Total rounding — R12. |
| 52890 real / 91856 padded / 33608 labelled, 42.4% и 63.4% skip | Подтверждено ok=true token_stats.json из tokstats_f81a736_exp012 (20261031T152441672330Z): 64 batches, 256 rows; 42.4207455% и 63.4122975%. |
| Layer profile 83/136/57/48/30/27 ms | Подтверждено layerprof_80f8e2e_s043: dequant 83.365; GEMM 136.348; event waits 56.542; SDPA 48.386; LoRA 30.323; norms+RoPE 26.878 ms. Это инструментированный старый source, не attribution финального full2. |
| EXP-010 −3.33 [−3.66, −2.86] s; first 6.53→2.93 s; peak 0 | Совпадает с ab_457cf41_exp006-010/summary.json, C vs A, четыре пары. |
| EXP-006 −0.06 [−0.53, +0.25] s; peak 0 | Совпадает с тем же summary, B vs A, raw median −0.055. |
| EXP-008 +0.25 [−0.62, +0.55]; EXP-009 +2.06 [+0.97, +2.38], peak −82 | Совпадает с ab_aaca49c_exp008-009/summary.json, шесть пар; raw +0.245/CI −0.615..0.55 и +2.06/CI 0.965..2.375. |
| EXP-013 −0.41 [−1.31, +1.16], first −0.68; EXP-015 −0.25 [−0.50, +0.30], peak 0 | Совпадает с ab_f81a736_exp013-015/summary.json, четыре пары. |
| EXP-002 4–66× slower | Подтверждено EXP-002.md:123–125 и исходным bench: 4.1219–66.0133× против decode+cuBLAS; не full-run A/B. |
| EXP-011 ≤6.5%, <0.2 s/run | Shared-config maximum speedup 1.064566; взвешенная оценка по kernel times первого финального run ~0.107 s. Это оценка, не измеренный whole-run выигрыш; memory peak в bench не подтверждён (physical_peak_verified=false). |
| EXP-007 +70·k MiB | Оценка размера fp16 gate/up при N2048/I8960, не измеренный peak; таблица честно пишет not measured. |
| Fixed task 256 rows, batch4, max512, epoch1, seed1, r16/alpha32 | Совпадает с task/prepare_data.py и task/task.yaml. |

## Notebook: статическая проверка

Проверен read-only `<scratch>/public-nb/soup-kernel-challenge-final.ipynb`.

- JSON корректен; 8 cells в template и draft.
- Все source кроме cell 1 совпадают побайтово после нормализации nbformat source
  (source может быть строкой или списком строк). Все прочие cell fields и
  top-level metadata совпадают.
- В cell 1 отличаются только значения REPO_URL и COMMIT. AST после замены этих
  двух значений совпадает; REPO_URL согласуется с адресом в SUBMISSION.
- Template хранит source cell 1 списком, draft строкой: это допустимое
  сериализационное отличие nbformat, не изменение поведения.
- Outputs пусты, execution_count=null во всех code cells: это **draft**, не
  доказательство выполненного notebook. Обычный Python внутри cells проходит
  AST parse; IPython `!` отдельно распознан как notebook shell syntax и не
  исполнялся. Проверка не доказывает runtime-успех.
- Литеральных credentials/tokens/email/персональных developer paths не найдено;
  GITHUB_TOKEN и HF_TOKEN — имена Kaggle Secrets, не значения. `/kaggle/working`
  — стандартный путь runtime. Реальные авторизационные файлы не открывались.
- COMMIT placeholder и одно повторение требуют действий R01/R03. `!`-команды
  сами по себе не делают notebook fail-fast; при сохранении необходимо проверить
  actual check output и checks.json, а не только успешный статус Run All.

SHA256 template:
`58c941504ecdb49e9ee10918905eb0b6615e27725feab015c01e91150edf7972`.
SHA256 проверенного draft:
`d1766a7e1e3d7abdf75f290a5ac0930ce155b80ea26f49b8a686bdffcde277e7`.

## Пути, команды и secret scan

- 30/30 локальных Markdown-ссылок существуют: 26 в README, 4 в SUBMISSION.
  Все упомянутые Python entrypoints, solution modules, task/harness/tests,
  LICENSE и NOTICE существуют. Все упомянутые git SHA разрешаются локально.
- 10 внешних source/reference ссылок README успешно открылись через read-only
  web lookup, включая pinned PEFT/bnb pages, papers, model/dataset cards.
  Публичная доступность source-репозитория — отдельное замечание R02.
- `runs/result.json` создаёт run_task.py, `data/*.jsonl` — prepare_data.py;
  отсутствие этих generated outputs в Git нормально. `~/.kaggle/kaggle.json`
  — внешний prerequisite, не пропавший repo file. Неопределённый E — R09.
- Bash blocks README проходят `bash -n`; SUBMISSION block ожидаемо не проходит
  из-за незаполненного checkout placeholder. Runner `run --help` выполнен
  успешно; `run`, `push-code`, modes unit/solution/full2 и --dry-run существуют.
  `--repeats 2`, checks/score arguments проверены по исходникам harness.
  Ни pip install, ни training, ни push-code/run с upload не выполнялись.
- `git ls-files -z` + `rg -l` по tracked files: 123 файла в исходном snapshot,
  повторная проверка 124 tracked files после чужого commit. Регулярные выражения
  для GitHub/HF/OpenAI/AWS/Google/Slack/Kaggle token forms, private-key headers,
  credential URLs, quoted token/password/secret assignments, Bearer/JWT,
  emails, macOS/Linux/Windows developer paths дали **0 совпадений**.
  Tracked `.env`, kaggle.json, auth.json, credentials, private key files не найдены.
- Публичные repository/dataset handles и upstream copyright attribution
  присутствуют намеренно; они не являются найденными секретами. В отчёт не
  перенесены персональные scratch paths, Kaggle account IDs или auth values.
  Regex scan не является доказательством отсутствия произвольного секрета и
  не охватывает Git history, ignored files либо внешние аккаунты.

## Порядок принятия предложений

Сначала закрыть R01–R05: определить final SHA и доступ, заполнить submission,
получить двухповторный notebook с доступными outputs и проверенной средой.
Затем исправить измеримые несоответствия R06–R10 и небольшие уточнения R11–R12.
Изменения остаются за координатором/владельцами соответствующих файлов.
