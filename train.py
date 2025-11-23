"""
Author: Lynn Ye
Created on: 2025/11/10
Brief:
"""
import argparse
import warnings

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
import torch

from ttm.config import config, model_config
from ttm.data_preparation.data_module import UCDataModule
from ttm.module.uc_module import configure_callbacks, UCModule
from ttm.module.chord_module import ChordModule   # NEW: chord module


pl.seed_everything(config['seed'])
warnings.filterwarnings('ignore')

feature_shorthand_map = {
    'unconditional': 'uc',
    'chord': 'ch',        
}


def load_uc_state_dict(ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    return state_dict


def train(args, use_wandb=True, resume_wandb_id=None, resume_ckpt=None):
    # Data
    data_module = UCDataModule(args.data_dir, feature=args.feature)

    # Model
    if args.feature == 'unconditional':
        model = UCModule(model_config['unconditional'])

    elif args.feature == 'chord':
        chord_cfg = model_config['chord']

        if args.uc_ckpt_path is not None and args.uc_ckpt_path != '':
            uc_state_dict = load_uc_state_dict(args.uc_ckpt_path, device=args.device if args.device != 'auto' else 'cpu')
        else:
            uc_state_dict = None

        model = ChordModule(
            chord_cfg,
            uc_state_dict=uc_state_dict,
            freeze_uc=args.freeze_uc,
        )
    else:
        raise ValueError(f'Invalid feature type: {args.feature}')

    # Logger
    name = '' if args.name == '' else '-' + args.name
    wandb_logger = WandbLogger(
        project='tap_music',
        name=f'{feature_shorthand_map.get(args.feature, args.feature)}{name}',
        save_dir=args.train_dir,
        id=resume_wandb_id,
        resume='allow'
    )

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=args.train_dir,
        logger=wandb_logger if use_wandb else CSVLogger(args.train_dir),
        max_epochs=args.nepoch,
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        check_val_every_n_epoch=args.eval_n_epoch,
        callbacks=configure_callbacks(
            save_dir=args.train_dir,
            prefix=args.feature,
            monitor='val_top5_acc' if args.feature in ['unconditional', 'chord'] else 'val_loss'
        ),
        accelerator=args.device,
    )

    # Train
    trainer.fit(model, data_module, ckpt_path=resume_ckpt)


def create_argparser():
    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument('--train_dir', type=str, help='Workspace directory.')
    parser.add_argument('--feature', type=str, default='unconditional',
                        help='Feature type. One of: unconditional, chord.')
    parser.add_argument('--data_dir', type=str, help='Data directory.')
    parser.add_argument('--nepoch', type=int, default=1000, help='Num epochs.')
    parser.add_argument('--eval_n_epoch', type=int, default=20, help='Eval per n epochs.')
    parser.add_argument('--name', type=str, default='', help='Custom name (short) to be displayed on wandb')
    parser.add_argument('--device', type=str, default='auto', help='Specify device (cpu, gpu, auto)')

    parser.add_argument(
        '--uc_ckpt_path',
        type=str,
        default='',
        help='Path to a pretrained UC checkpoint.'
             'If empty, chord model is trained from scratch.'
    )
    parser.add_argument(
        '--freeze_uc',
        type=bool,
        default=False,
        help='If set, freeze the UC core inside the chord model.'
    )

    return parser


def main():
    parser = create_argparser()
    args = parser.parse_args()
    if args.feature not in config.keys():
        raise Exception(f"{args.feature} config not found. Update config.yaml")

    train(args)


def debug_main():
    print("\x1B[33m[Warning]\033[0m You are running debug train")
    parser = create_argparser()
    args = parser.parse_args()

    # Example: debug unconditional
    # args.feature = 'unconditional'
    # args.train_dir = '/Users/kurono/Desktop/10701 final/tap_the_music/output/debug_uc'
    # args.data_dir = '/Users/kurono/Desktop/10701 final/tap_the_music/output'
    # args.name = 'debug_uc'
    # args.device = 'cpu'

    # Example: debug chord from scratch
    args.feature = 'chord'
    args.train_dir = 'data/chord/debug_ch'
    args.data_dir = 'data/chord'
    args.name = 'debug_ch'
    args.device = 'cpu'
    args.uc_ckpt_path = ''       # set to UC ckpt path to warm-start
    args.freeze_uc = False

    if args.feature not in config.keys():
        raise Exception(f"{args.feature} is not supported")
    train(args, use_wandb=False)

if __name__ == "__main__":
    main()
    # debug_main()
