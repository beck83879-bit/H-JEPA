"""Seed initialization before entering the otherwise unchanged upstream trainer."""
import hashlib
import json
import os
import gc
import time
from pathlib import Path

import lightning as pl
import hydra
import torch
import main_hjepa
from omegaconf import OmegaConf

seed = int(os.environ['HJEPA_TRAIN_SEED'])
pl.seed_everything(seed, workers=True)
original_create = main_hjepa.create_world_model


def create_and_record(cfg):
    assert int(cfg.seed) == seed
    model, losses = original_create(cfg)
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    record = {
        'seed': seed, 'seeded_before_construction': True,
        'initial_model_sha256': digest.hexdigest(),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'matmul': torch.get_float32_matmul_precision(),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'main_source': main_hjepa.__file__,
    }
    Path(os.environ['HJEPA_INIT_RECORD']).write_text(json.dumps(record, indent=2) + '\n')
    print('INITIALIZATION', json.dumps(record), flush=True)
    return model, losses


main_hjepa.create_world_model = create_and_record


class RuntimeRecord(pl.Callback):
    def on_train_start(self, trainer, pl_module):
        assert trainer.num_training_batches == 13292, trainer.num_training_batches
        assert trainer.estimated_stepping_batches == 119628
        assert trainer.world_size == 1 and trainer.accumulate_grad_batches == 1

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % 50:
            return
        record = dict(time=time.time(), seed=seed, pid=os.getpid(),
                      step=trainer.global_step, epoch=trainer.current_epoch,
                      allocated_gib=torch.cuda.memory_allocated() / 1024**3,
                      reserved_gib=torch.cuda.memory_reserved() / 1024**3,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
                      gc_enabled=gc.isenabled(), gc_counts=gc.get_count())
        path = Path(os.environ['HJEPA_INIT_RECORD']).parent / 'runtime.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(record) + '\n')


original_manager_init = main_hjepa.ResumableManager.__init__


def manager_init(self, *args, **kwargs):
    kwargs['trainer'].callbacks.append(RuntimeRecord())
    return original_manager_init(self, *args, **kwargs)


main_hjepa.ResumableManager.__init__ = manager_init


def dispatch(cfg):
    if os.environ.get('HJEPA_PREFLIGHT_CONFIG'):
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
                       os.environ['HJEPA_PREFLIGHT_CONFIG'])
        return
    return main_hjepa.run.__wrapped__(cfg)


hydra.main(version_base=None, config_path=str(Path(main_hjepa.__file__).parent / 'config/train'),
           config_name=None)(dispatch)()
