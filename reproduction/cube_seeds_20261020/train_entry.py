"""Seed initialization before entering the otherwise unchanged upstream trainer."""
import hashlib
import json
import os
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


def dispatch(cfg):
    if os.environ.get('HJEPA_PREFLIGHT_CONFIG'):
        OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)),
                       os.environ['HJEPA_PREFLIGHT_CONFIG'])
        return
    return main_hjepa.run.__wrapped__(cfg)


hydra.main(version_base=None, config_path=str(Path(main_hjepa.__file__).parent / 'config/train'),
           config_name=None)(dispatch)()
