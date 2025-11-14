"""
Author: Lynn Ye
Created on: 2025/11/10
Brief:
"""
import argparse
import warnings

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from ttm.config import config, model_config
from ttm.data_preparation.data_module import UCDataModule
from ttm.module.uc_module import configure_callbacks, UCModule

pl.seed_everything(config['seed'])
warnings.filterwarnings('ignore')


def train(args, use_wandb=True, resume_wandb_id=None, resume_ckpt=None):
    # Data
    data_module = UCDataModule(args.data_dir, feature=args.feature)

    # Model
    if args.feature == 'unconditional':
        model = UCModule(model_config['unconditional'])
    else:
        raise ValueError('Invalid feature type.')

    # Logger
    wandb_logger = WandbLogger(
        project='tap_music',
        name=f'{args.feature}-{args.name}',
        save_dir=args.train_dir,
        id=resume_wandb_id,
        resume='allow'
    )

    # Trainer
    trainer = pl.Trainer(
        default_root_dir=args.train_dir,
        logger=wandb_logger if use_wandb else CSVLogger(args.train_dir),
        log_every_n_steps=50,
        reload_dataloaders_every_n_epochs=True,
        check_val_every_n_epoch=args.eval_n_epoch,
        callbacks=configure_callbacks(save_dir=args.train_dir, prefix=args.feature, monitor='ppl'),
        accelerator=args.device
        # gpus=training_configs[args.feature]['gpus'],
    )

    # Train
    trainer.fit(model, data_module, ckpt_path=resume_ckpt)


def create_argparser():
    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument('--train_dir', type=str, help='Workspace directory.')
    parser.add_argument('--feature', type=str, default='unconditional', help='Feature type.')
    parser.add_argument('--data_dir', type=str, help='Data directory.')
    parser.add_argument('--nepoch', type=int, default=1000, help='Num epochs.')
    parser.add_argument('--eval_n_epoch', type=int, default=20, help='Eval per n epochs.')
    parser.add_argument('--name', type=str, default='', help='Custom name (short) to be displayed on wandb')
    parser.add_argument('--device', type=str, default='auto', help='Specify device if you want')

    return parser


def main():
    parser = create_argparser()
    args = parser.parse_args()
    if args.feature not in config.keys():
        raise Exception(f"{args.model_type} config not found. Update config.yaml")

    train(args)


def debug_main():
    print("\x1B[33m[Warning]\033[0m You are running debug train")
    parser = create_argparser()
    args = parser.parse_args()

    # TODO Maybe use dotenv for convenient debug
    args.feature = 'unconditional'
    args.train_dir = '/Users/kurono/Desktop/10701 final/tap_the_music/output/debug'
    args.data_dir = '/Users/kurono/Desktop/10701 final/tap_the_music/output'
    args.name = 'debug'
    args.device = 'cpu'

    if args.feature not in config.keys():
        raise Exception(f"{args.feature} is not supported")
    train(args, use_wandb=False,
          # resume_wandb_id='r82bduac',
          # resume_ckpt='/Users/kurono/Desktop/15798 final/mbt/trdata/beat_ckpt/last.ckpt'
          )


if __name__ == "__main__":
    main()
    # debug_main()
