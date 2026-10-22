"""Inspect completed training and run explicitly configured standalone evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('stage', choices=['inspect', 'evaluate', 'probe', 'check-eval'])
parser.add_argument('run_dir', type=Path)
parser.add_argument('--train-seed', type=int, required=True)
args = parser.parse_args()
run = args.run_dir
policy = run / 'cube_hjepa_l3_object.ckpt'


def finite(value):
    if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
        assert torch.isfinite(value).all(), 'Nonfinite checkpoint tensor'
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


if args.stage == 'inspect':
    cfg = OmegaConf.load(run / 'config.yaml')
    assert cfg.seed == args.train_seed
    assert cfg.trainer.max_epochs == 8 and cfg.loader.batch_size == 128
    assert cfg.trainer.devices == 1 and cfg.trainer.precision == 'bf16-mixed'
    assert cfg.trainer.gradient_clip_val == 1 and cfg.trainer.get('max_steps', -1) == -1
    checkpoint = torch.load(run / 'lightning_resume/last.ckpt', map_location='cpu', weights_only=False)
    assert checkpoint['global_step'] == 83912 and checkpoint['epoch'] == 7
    assert checkpoint['optimizer_states'] and checkpoint['lr_schedulers']
    finite(checkpoint)
    del checkpoint
    model = torch.load(policy, map_location='cpu', weights_only=False)
    assert model.num_levels == 3
    finite(model.state_dict())
    assert (run / 'normalizer.pt').is_file()
    save(run / 'inspection.json', dict(ok=True, global_step=83912, completed_epochs=8,
                                      seed=args.train_seed, policy=str(policy)))
else:
    # Match the prior standalone 52% control, independent of training entry defaults.
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    precision = dict(matmul=torch.get_float32_matmul_precision(),
                     tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_tf32=torch.backends.cudnn.allow_tf32,
                     benchmark=torch.backends.cudnn.benchmark)
    assert precision == dict(matmul='highest', tf32=False, cudnn_tf32=True, benchmark=False)
    if args.stage in ('evaluate', 'check-eval'):
        from planning_eval import load_eval_config, run_planning_eval
        cfg = load_eval_config('cube_l3')
        cfg.policy = str(policy)
        cfg.seed = 42
        out = run / 'planning_eval/seed42_highest'
        cfg.output.dir = str(out)
        task = Path(cfg.load_eval_trajs_path)
        assert hashlib.sha256(task.read_bytes()).hexdigest() == '8880d7d72f0d044dcc3df0deab162f2488ff603a4cb01d472c5272d80a01a5e7'
        assert cfg.eval.num_eval == 50 and cfg.eval.eval_budget == 30
        assert cfg.eval.goal_offset_steps == 20
        assert [cfg.hierarchical_plan_config[f'level{i}'].horizon for i in (1, 2, 3)] == [2, 2, 1]
        assert [cfg.solver.solvers[f'level{i}'].num_samples for i in (1, 2, 3)] == [25, 800, 800]
        assert [cfg.solver.solvers[f'level{i}'].action_noise for i in (1, 2, 3)] == [0, 0.01, 0.01]
        assert all(cfg.solver.solvers[f'level{i}'].n_steps == 90 for i in (1, 2, 3))
        print(json.dumps(dict(training_seed=args.train_seed, evaluation_seed=42,
                              precision=precision, config=OmegaConf.to_container(cfg, resolve=True))), flush=True)
        if args.stage == 'evaluate':
            assert policy.is_file() and not out.exists()
            out.mkdir(parents=True)
            save(out / 'precision.json', precision)
            run_planning_eval(cfg)
            metrics = OmegaConf.load(out / 'metrics.yaml')
            successes = list(metrics.episode_successes)
            assert len(successes) == 50
            for i in range(50):
                chunk = json.loads((out / f'chunks/tasks_{i:03d}-{i+1:03d}.json').read_text())
                assert chunk['episode_successes'] == [successes[i]]
            assert math.isclose(float(metrics.success_rate), 2 * sum(successes))
            save(out / 'verified.json', dict(ok=True, training_seed=args.train_seed,
                 evaluation_seed=42, task_count=50, success_count=sum(successes),
                 success_rate=2 * sum(successes), precision=precision))
    else:
        from final_probing_decoding_eval import load_probing_config, run_final_probing_decoding_eval
        train_cfg = OmegaConf.load(run / 'config.yaml')
        cfg = load_probing_config(train_cfg.final_probing_decoding_eval.config_name)
        cfg = OmegaConf.merge(cfg, train_cfg.final_probing_decoding_eval.overrides)
        cfg.policy = str(policy)
        cfg.seed = args.train_seed
        out = run / 'final_probing_decoding_eval'
        assert policy.is_file() and not out.exists()
        run_final_probing_decoding_eval(cfg, train_cfg=train_cfg, output_dir=out)
        save(out / 'precision.json', precision)
