#!/usr/bin/env python3
"""EXP-012: native Soup/TRL token and padding accounting, CPU only, no model.

Use the pinned Soup formatter, TRL dataset preparation/collator and the real
HF/Accelerate dataloader, including Soup's Rewind sampler wrapper. No Trainer
or model constructor runs: a data-only Trainer shell supplies these methods.
The only differences from training arguments are CPU placement and disabled
mixed precision; data, masks, padding, sampler, seed and batching stay fixed.

JSON contains every batch; the text table is short. Counts distinguish raw
labels from shifted causal targets. Our head forward computes entropy on all
real tokens; only head backward compacts to nonignored shifted targets.
--self-check is stdlib-only. Actual measurements require the pinned packages
and the official prepared data/train.jsonl; no solution module is imported.
"""

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
PINNED = {'soup-cli': '0.77.0', 'transformers': '5.16.1', 'trl': '0.29.1',
          'peft': '0.20.0', 'accelerate': '1.14.0', 'datasets': '4.8.5'}
COUNT_KEYS = ('sequences', 'real_tokens', 'padded_tokens', 'capacity_tokens',
              'completion_tokens', 'causal_target_tokens')


def fractions(counts):
    real, padded, capacity = (counts[k] for k in ('real_tokens', 'padded_tokens', 'capacity_tokens'))
    causal = counts['causal_target_tokens']
    if not 0 <= causal <= counts['completion_tokens'] <= real <= padded <= capacity:
        raise ValueError('Invalid token count ordering')
    return dict(padding_tokens=padded-real, padding_ratio=(padded-real)/padded,
                utilization_512=real/capacity, padded_utilization_512=padded/capacity,
                decoder_skip_fraction=(padded-real)/padded, head_forward_skip_fraction=(padded-real)/padded,
                head_backward_skip_fraction=(padded-causal)/padded,
                head_backward_skip_vs_real_fraction=(real-causal)/real)


def batch_statistics(mask, labels, max_length=512):
    if not mask or len(mask) != len(labels) or not mask[0]:
        raise ValueError('Expected a nonempty collated batch with matching labels')
    width = len(mask[0])
    lengths, completion, causal = [], [], []
    for attention, target in zip(mask, labels):
        if len(attention) != width or len(target) != width or width > max_length:
            raise ValueError('Invalid collator shape or max_length')
        length, padding = 0, False
        for active, label in zip(attention, target):
            if active not in (0, 1) or (padding and active):
                raise ValueError('Expected binary right padding')
            if active:
                length += 1
            else:
                padding = True
                if label != -100:
                    raise ValueError('Padding labels must be ignored')
        if not length:
            raise ValueError('Empty sequence after official preparation')
        lengths.append(length)
        completion.append(sum(t != -100 for t in target))
        causal.append(sum(t != -100 for t in target[1:]))
    if width != max(lengths):
        raise ValueError('Expected the native dynamic S_max, without pad_to_multiple_of')
    counts = dict(sequences=len(mask), real_tokens=sum(lengths), padded_tokens=len(mask)*width,
                  capacity_tokens=len(mask)*max_length, completion_tokens=sum(completion),
                  causal_target_tokens=sum(causal))
    return dict(**counts, **fractions(counts), sequence_max=width,
                sequence_lengths=lengths, completion_tokens_per_row=completion,
                causal_target_tokens_per_row=causal)


def totals(steps):
    counts = {k: sum(s[k] for s in steps) for k in COUNT_KEYS}
    if not steps:
        raise ValueError('No batches were produced')
    return dict(steps=len(steps), **counts, **fractions(counts))


def print_table(report, limit=8):
    if not report.get('ok'):
        print('Token statistics failed:', report.get('error', 'incomplete'))
        return
    print('step | real | B*Smax | pad % | real/(B*512) % | labels | causal targets')
    steps = report['steps']
    selected = steps if len(steps) <= limit else steps[:limit//2] + steps[-limit//2:]
    previous = 0
    for step in selected:
        if previous and step['step'] > previous + 1:
            print(f'... {step["step"]-previous-1} steps; full data in token_stats.json ...')
        print(f'{step["step"]:4d} | {step["real_tokens"]:4d} | {step["padded_tokens"]:6d} | '
              f'{100*step["padding_ratio"]:5.1f} | {100*step["utilization_512"]:15.1f} | '
              f'{step["completion_tokens"]:6d} | {step["causal_target_tokens"]:14d}')
        previous = step['step']
    value = report['totals']
    print(f'TOTAL | {value["real_tokens"]} | {value["padded_tokens"]} | '
          f'{100*value["padding_ratio"]:.2f} | {100*value["utilization_512"]:.2f} | '
          f'{value["completion_tokens"]} | {value["causal_target_tokens"]}')
    print('Skip fractions (token positions, not wall time):', json.dumps({
        k: value[k] for k in ('decoder_skip_fraction', 'head_forward_skip_fraction',
                             'head_backward_skip_fraction', 'head_backward_skip_vs_real_fraction')}))
    print('Phase wall seconds:', json.dumps(report.get('timings', {}), sort_keys=True))


def check_reference(report, path):
    """Optional exact cross-check against saved full2 row order and token logs."""
    evidence = json.loads(Path(path).read_text())
    checked = []
    for side in ('base', 'solution'):
        for run in evidence.get('runs', {}).get(side, []):
            rewind = run.get('rewind', [])
            cumulative = run.get('log', {}).get('num_tokens', [])
            if not rewind or not cumulative:
                continue
            if len(rewind) != len(report['steps']) or len(cumulative) != len(report['steps']):
                raise ValueError('Reference run has a different number of steps')
            real = 0
            for step, recorded, tokens in zip(report['steps'], rewind, cumulative):
                real += step['real_tokens']
                if (step['row_indices'] != recorded[0] or step['causal_target_tokens_per_row'] != recorded[1]
                        or real != int(tokens)):
                    raise ValueError(f'Reference mismatch in {run["run"]} step {step["step"]}')
            checked.append(run['run'])
    if not checked:
        raise ValueError('Reference contains no complete rewind/token evidence')
    return dict(ok=True, runs=checked, checks=['row order', 'causal targets per row', 'cumulative real tokens'])


def source_hash(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def measure(args, report):
    started = time.perf_counter()
    versions = {name: metadata.version(name) for name in PINNED}
    for name, expected in PINNED.items():
        if versions[name] != expected:
            raise RuntimeError(f'Exact pipeline requires {name}=={expected}, got {versions[name]}')
    import torch
    from accelerate import Accelerator, DataLoaderConfiguration
    from rich.console import Console
    from transformers import AutoTokenizer, TrainingArguments, set_seed
    from transformers.trainer import Trainer
    from trl import SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from soup_cli.config.loader import load_config
    from soup_cli.data.loader import load_dataset
    from soup_cli.data.sft_format import build_format_row
    from soup_cli.trainer.rewind_hf import RewindState, make_rewind_trainer_class
    from soup_cli.trainer.sft import SFTTrainerWrapper, _map_text_sft_rows, _ensure_assistant_masks
    from soup_cli.utils.run_provenance import base_revision_kwargs
    from soup_cli.utils.seeding import training_seed_kwargs
    from soup_cli.utils.warmup import resolve_trainer_warmup_steps

    if torch.__version__.split('+')[0] != '2.11.0':
        raise RuntimeError(f'Exact sampler requires torch==2.11.0, got {torch.__version__}')
    report['versions'] = dict(versions, torch=torch.__version__)
    timings = report['timings']
    timings['imports_s'] = time.perf_counter() - started
    cfg = load_config(args.config, raise_unreadable=True)
    tcfg = cfg.training
    fixed = (cfg.base == 'Qwen/Qwen2.5-1.5B-Instruct' and cfg.task == 'sft'
             and cfg.data.format == 'alpaca' and cfg.data.max_length == 512 and cfg.data.val_split == 0
             and tcfg.batch_size == 4 and tcfg.gradient_accumulation_steps == 1
             and tcfg.seed == 1 and tcfg.epochs == 1 and not tcfg.packing and not tcfg.multipack
             and tcfg.rewind_log and not tcfg.gradient_checkpointing)
    if not fixed:
        raise ValueError('Config is outside the fixed task pipeline supported by this measurement')
    if args.train_file:
        cfg.data.train = str(args.train_file.resolve())
    train_path = Path(cfg.data.train)
    if not train_path.is_file():
        raise FileNotFoundError('Prepare the official data first with task/prepare_data.py')
    report['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    report['train_file_sha256'] = hashlib.sha256(train_path.read_bytes()).hexdigest()
    report['settings'] = dict(base=cfg.base, max_length=512, batch_size=4, gradient_accumulation_steps=1,
        seed=1, data_seed=training_seed_kwargs(tcfg)['data_seed'], epochs=1, packing=False, padding_free=False,
        train_on_responses_only=cfg.data.train_on_responses_only,
        mask_history=cfg.data.mask_history, train_on_eot=tcfg.train_on_eot, rewind_log=True)
    console = Console()
    phase = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=False, **base_revision_kwargs(cfg.base))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    timings['tokenizer_load_s'] = time.perf_counter() - phase
    report['tokenizer'] = dict(name=tokenizer.name_or_path, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, revision=tokenizer.init_kwargs.get('_commit_hash'),
        chat_template_sha256=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        backend_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest())

    phase = time.perf_counter()
    dataset = load_dataset(cfg.data)
    timings['soup_data_load_s'] = time.perf_counter() - phase
    if len(dataset['train']) != 256 or dataset.get('val'):
        raise ValueError('Official task requires exactly 256 formatted rows and no validation split')
    phase = time.perf_counter()
    formatter = build_format_row(tokenizer=tokenizer, data_cfg=cfg.data, console=console, training_cfg=tcfg)
    tokenized = _map_text_sft_rows(dataset['train'], format_row=formatter, split='train', max_length=512)
    tokenized = _ensure_assistant_masks(tokenized)
    timings['soup_tokenization_s'] = time.perf_counter() - phase

    with tempfile.TemporaryDirectory(prefix='soup-token-stats-') as work:
        # CPU/device and precision fields serve measurement only. All data-facing
        # fields/defaults come from the same TrainingArguments -> Soup SFTConfig conversion.
        training = TrainingArguments(output_dir=work, use_cpu=True, fp16=False, bf16=False,
            per_device_train_batch_size=tcfg.batch_size, num_train_epochs=tcfg.epochs,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.lr, warmup_steps=resolve_trainer_warmup_steps(tcfg.warmup_ratio),
            weight_decay=tcfg.weight_decay, max_grad_norm=tcfg.max_grad_norm, optim=tcfg.optimizer,
            lr_scheduler_type=tcfg.scheduler, logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps, save_total_limit=3,
            remove_unused_columns=False, report_to='none', **training_seed_kwargs(tcfg))
        training = SFTTrainerWrapper._as_sft_config(training, cfg.data.max_length, packing=tcfg.packing)
        if (training.padding_free or training.packing or training.shuffle_dataset
                or training.pad_to_multiple_of is not None or training.train_sampling_strategy != 'random'):
            raise ValueError('Unexpected packing/padding/shuffle/sampler defaults; exact pipeline needs review')
        cls = make_rewind_trainer_class(SFTTrainer)
        shell = cls.__new__(cls)  # never calls Trainer.__init__ or a model constructor
        shell.args, shell.processing_class, shell._is_vlm = training, tokenizer, False
        phase = time.perf_counter()
        prepared = SFTTrainer._prepare_dataset(shell, tokenized, tokenizer, training, False, None, 'train')
        timings['trl_dataset_prepare_s'] = time.perf_counter() - phase
        sample = next(iter(prepared))
        completion_only = ('prompt' in sample and 'completion' in sample
                           if training.completion_only_loss is None else training.completion_only_loss)
        pad_token = training.pad_token or tokenizer.pad_token or tokenizer.eos_token
        collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.convert_tokens_to_ids(pad_token),
            completion_only_loss=completion_only, padding_free=False,
            pad_to_multiple_of=training.pad_to_multiple_of)
        config = training.accelerator_config.to_dict()
        loader_config = DataLoaderConfiguration(**{k: config[k] for k in
            ('split_batches', 'dispatch_batches', 'even_batches', 'use_seedable_sampler', 'non_blocking')},
            data_seed=training.data_seed)
        shell.accelerator = Accelerator(cpu=True, dataloader_config=loader_config)
        if shell.accelerator.num_processes != 1:
            raise ValueError('Exact official task measurement is single-process')
        shell.train_dataset, shell.data_collator, shell._train_batch_size = prepared, collator, 4

        class Sink:
            def __init__(self):
                self.rows = deque()
            def record_batch(self, **entry):
                self.rows.append(entry['rows'])

        sink = Sink()
        state = RewindState(sink)
        shell._soup_rewind_state = state
        # Trainer.__init__ reseeds after model/LoRA setup. No model RNG work is
        # emulated: restart the same seed before native dataloader construction.
        set_seed(training.seed)
        phase = time.perf_counter()
        loader = shell.get_train_dataloader()
        if hasattr(loader, 'set_epoch'):
            loader.set_epoch(0)
        sampler = loader.batch_sampler.sampler
        if (type(sampler).__name__ != 'IndexRecordingSampler'
                or not isinstance(getattr(sampler, '_inner', None), torch.utils.data.RandomSampler)):
            raise RuntimeError('Native Rewind/RandomSampler composition changed; refusing an inexact order')
        report['collator'] = dict(type=type(collator).__name__, completion_only_loss=completion_only,
            padding_free=collator.padding_free, pad_to_multiple_of=collator.pad_to_multiple_of)
        report['sampling'] = dict(trainer=cls.__name__, sampler=type(sampler).__name__,
            wrapped_sampler=type(sampler._inner).__name__,
            use_seedable_sampler=loader_config.use_seedable_sampler, data_seed=training.data_seed,
            epoch=0, world_size=1, batch_size=training.per_device_train_batch_size,
            drop_last=training.dataloader_drop_last, workers=training.dataloader_num_workers,
            pin_memory=training.dataloader_pin_memory, strategy=training.train_sampling_strategy)
        for number, batch in enumerate(loader, 1):
            if any(t.device.type != 'cpu' for t in batch.values() if isinstance(t, torch.Tensor)):
                raise RuntimeError('A collated tensor left the CPU')
            step = batch_statistics(batch['attention_mask'].tolist(), batch['labels'].tolist())
            state.on_micro_batch(step=number, row_loss=[0.0]*step['sequences'],
                                 row_tokens=step['causal_target_tokens_per_row'])
            if state.failed or state.dropped or not sink.rows:
                raise RuntimeError('Native Rewind sampler row accounting failed')
            step.update(step=number, row_indices=sink.rows.popleft())
            report['steps'].append(step)
        timings['collate_iterate_account_s'] = time.perf_counter() - phase
    report['totals'] = totals(report['steps'])
    if report['totals']['steps'] != 64 or report['totals']['sequences'] != 256:
        raise ValueError('Official one-epoch batch count changed')
    order = [i for step in report['steps'] for i in step['row_indices']]
    if sorted(order) != list(range(256)):
        raise ValueError('Rows were skipped, repeated or reordered outside the native permutation')
    report['sampling']['order_sha256'] = hashlib.sha256(json.dumps(order).encode()).hexdigest()
    timings['dataset_build_s'] = sum(timings[k] for k in
        ('soup_data_load_s', 'soup_tokenization_s', 'trl_dataset_prepare_s'))
    report['sources'] = {name: source_hash(fn) for name, fn in {
        'soup_format': build_format_row, 'soup_map': _map_text_sft_rows,
        'trl_prepare': SFTTrainer._prepare_dataset, 'trl_collator': DataCollatorForLanguageModeling.torch_call,
        'hf_dataloader': Trainer._get_dataloader, 'hf_sampler': Trainer._get_train_sampler,
        'soup_rewind': make_rewind_trainer_class}.items()}
    if torch.cuda.is_initialized():
        raise RuntimeError('Token measurement unexpectedly initialized CUDA')
    if any(name == 'solution' or name.startswith('solution.') for name in sys.modules):
        raise RuntimeError('Token measurement must not import solution patches')
    report['cuda_initialized'] = False
    if args.reference_result:
        report['reference_check'] = check_reference(report, args.reference_result)
    report['ok'] = True


def self_check():
    step = batch_statistics([[1, 1, 1, 0], [1, 1, 1, 1]],
                            [[-100, 2, 3, -100], [-100, -100, 5, 6]])
    assert (step['real_tokens'], step['padded_tokens'], step['completion_tokens'],
            step['causal_target_tokens']) == (7, 8, 4, 4)
    assert step['padding_ratio'] == 1/8 and step['utilization_512'] == 7/1024
    assert step['head_forward_skip_fraction'] != step['head_backward_skip_fraction']
    shifted = batch_statistics([[1, 1], [1, 0]], [[1, 2], [3, -100]])
    assert shifted['completion_tokens'] == 3 and shifted['causal_target_tokens'] == 1
    assert totals([step, shifted])['padding_ratio'] == 2/12
    for mask, labels in [([[0, 1]], [[-100, 2]]), ([[1, 0]], [[2, 3]]), ([[1, 2]], [[2, 3]])]:
        try:
            batch_statistics(mask, labels)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid mask/label geometry accepted')
    return dict(ok=True, checks=['dynamic padding', 'completion vs causal labels',
        'weighted totals', 'right-padding safety', 'head forward/backward distinction'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('out'))
    parser.add_argument('--config', type=Path, default=ROOT/'task/task.yaml')
    parser.add_argument('--train-file', type=Path)
    parser.add_argument('--reference-result', type=Path, help='Optional saved full2 result row/token oracle')
    parser.add_argument('--commit', default='')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.commit and not re.fullmatch('[0-9a-f]{40}', args.commit):
        parser.error('--commit requires a full lowercase SHA')
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
        return 0
    # Set before torch/Accelerate imports; only this measurement process changes.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['ACCELERATE_USE_CPU'] = 'true'
    sys.modules['unsloth'] = None
    sys.modules['unsloth_zoo'] = None
    report = dict(schema=1, ok=False, commit=args.commit or None, steps=[], timings={},
        created_utc=datetime.now(timezone.utc).isoformat(), model_loaded=False,
        semantics='decoder/head forward: real tokens; head backward: shifted nonignored labels',
        scope='official 256-row public task; no GPU, model load or solution patches')
    started = time.perf_counter()
    try:
        measure(args, report)
    except (Exception, SystemExit) as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        report['timings']['total_wall_s'] = time.perf_counter()-started
        args.out.mkdir(parents=True, exist_ok=True)
        target = args.out/'token_stats.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temporary.replace(target)
    print_table(report)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
