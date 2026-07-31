"""
Fast T1 SS flow training without expensive snapshot sampling.
Monkey-patches the snapshot method to skip it, then runs training.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import numpy as np
import random
from easydict import EasyDict as edict
from trellis2 import models, datasets, trainers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    random.seed(0)

    # Load config
    with open(args.config) as f:
        config = json.load(f)
    cfg = edict(config)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Build dataset
    dataset = getattr(datasets, cfg.dataset.name)(args.data_dir, **cfg.dataset.args)

    # Build model
    model_dict = {
        name: getattr(models, model.name)(**model.args).cuda()
        for name, model in cfg.models.items()
    }

    # Model summary
    num_trainable = sum(p.numel() for p in model_dict['denoiser'].parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Save model summary
    with open(os.path.join(args.output_dir, 'denoiser_model_summary.txt'), 'w') as f:
        for name, param in model_dict['denoiser'].named_parameters():
            f.write(f'{name:<72}{str(param.shape):<32}{str(param.dtype):<16}{param.requires_grad}\n')
        f.write(f'\nNumber of parameters: {sum(p.numel() for p in model_dict["denoiser"].parameters())}\n')
        f.write(f'Number of trainable parameters: {num_trainable}\n')

    # Build trainer
    trainer = getattr(trainers, cfg.trainer.name)(
        model_dict, dataset, **cfg.trainer.args,
        output_dir=args.output_dir, load_dir=args.output_dir, step=None
    )

    # Monkey-patch snapshot to be a no-op (skip expensive sampling)
    trainer.snapshot = lambda *a, **kw: None
    trainer.snapshot_dataset = lambda *a, **kw: None

    # Run training
    trainer.run()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
