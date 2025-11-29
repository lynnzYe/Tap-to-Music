"""
Author: YZQ
Created on: 2025/11/27
Brief: Train HandFiLM model by finetuning from pretrained unconditional model
"""
import argparse
import warnings
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from ttm.config import config, model_config
from yzq.data_module import HandDataModule
from yzq.model import HandFiLMModule, configure_callbacks

pl.seed_everything(config.get('seed', 42))
warnings.filterwarnings('ignore')


def train(args):
    """Train HandFiLM model."""
    
    # Data Module
    dm = HandDataModule(
        data_dir=args.data_dir,
        feature_type='pop909hand',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Model config
    m_config = {
        **model_config.get('unconditional', {}),
        'learning_rate': args.lr,
        'num_hands': 2,
        'hand_emb_dim': args.hand_emb_dim,
        'film_hidden_dim': args.film_hidden_dim,
        'multi_layer_film': args.multi_layer_film,
        'hand_dropout': args.hand_dropout,
        'label_smoothing': args.label_smoothing,
    }
    
    # Model - finetune from pretrained
    model = HandFiLMModule(
        m_config=m_config,
        pretrained_path=args.pretrained_path,
        freeze_backbone=args.freeze_backbone
    )
    
    # Logger
    if args.use_wandb:
        logger = WandbLogger(
            project='tap-to-music-hand',
            name=args.name,
            save_dir=args.output_dir,
        )
    else:
        logger = CSVLogger(args.output_dir, name='hand_film')
    
    # Callbacks
    callbacks = configure_callbacks(
        save_dir=args.output_dir,
        prefix='hand_film',
        monitor='val_loss',
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        accelerator=args.device,
    )
    
    # Train
    print("=" * 60)
    print("Training HandFiLM Model")
    print("=" * 60)
    print(f"Pretrained checkpoint: {args.pretrained_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    trainer.fit(model, dm)
    
    # Test
    print("\n" + "=" * 60)
    print("Testing...")
    print("=" * 60)
    trainer.test(model, dm)
    
    print("\nTraining complete!")
    return model, trainer


def main():
    parser = argparse.ArgumentParser(description='Train HandFiLM model')
    
    # Paths
    parser.add_argument('--pretrained_path', type=str, 
                        default='uc/last-trained-on-maestro-asap.ckpt',
                        help='Path to pretrained UCModule checkpoint')
    parser.add_argument('--data_dir', type=str, 
                        default='yzq/output',
                        help='Directory containing pop909hand-*.pkl files')
    parser.add_argument('--output_dir', type=str, 
                        default='yzq/checkpoints/hand_film',
                        help='Output directory for checkpoints')
    
    # Training params
    parser.add_argument('--max_epochs', type=int, default=50,
                        help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    
    # Model params
    parser.add_argument('--hand_emb_dim', type=int, default=32,
                        help='Hand embedding dimension')
    parser.add_argument('--film_hidden_dim', type=int, default=128,
                        help='FiLM MLP hidden dimension')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze pretrained backbone weights')
    parser.add_argument('--multi_layer_film', action='store_true',
                        help='Apply FiLM at input level too (multi-layer)')
    parser.add_argument('--hand_dropout', type=float, default=0.1,
                        help='Dropout on hand embedding')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing for cross-entropy loss')
    
    # Other
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, gpu, mps)')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--name', type=str, default='hand_film',
                        help='Run name for wandb')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    train(args)


if __name__ == "__main__":
    main()

