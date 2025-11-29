"""
Hyperparameter sweep for Hand and Window FiLM models.
Compares different configurations and logs results.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime
import warnings

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from ttm.config import config, model_config
from yzq.data_module import HandDataModule, WindowDataModule
from yzq.model import HandFiLMModule, WindowFiLMModule, configure_callbacks

warnings.filterwarnings('ignore')

# Hyperparameter configurations to compare
HAND_CONFIGS = [
    # Baseline (what we've been using)
    {
        'name': 'hand_baseline',
        'hand_emb_dim': 32,
        'film_hidden_dim': 128,
        'lr': 3e-4,
        'multi_layer_film': True,
        'hand_dropout': 0.1,
        'label_smoothing': 0.1,
    },
    # Smaller model
    {
        'name': 'hand_small',
        'hand_emb_dim': 16,
        'film_hidden_dim': 64,
        'lr': 3e-4,
        'multi_layer_film': False,
        'hand_dropout': 0.1,
        'label_smoothing': 0.1,
    },
    # Larger model
    {
        'name': 'hand_large',
        'hand_emb_dim': 64,
        'film_hidden_dim': 256,
        'lr': 1e-4,
        'multi_layer_film': True,
        'hand_dropout': 0.15,
        'label_smoothing': 0.1,
    },
    # Higher learning rate
    {
        'name': 'hand_high_lr',
        'hand_emb_dim': 32,
        'film_hidden_dim': 128,
        'lr': 1e-3,
        'multi_layer_film': True,
        'hand_dropout': 0.1,
        'label_smoothing': 0.1,
    },
    # No label smoothing
    {
        'name': 'hand_no_smooth',
        'hand_emb_dim': 32,
        'film_hidden_dim': 128,
        'lr': 3e-4,
        'multi_layer_film': True,
        'hand_dropout': 0.1,
        'label_smoothing': 0.0,
    },
    # Higher dropout
    {
        'name': 'hand_high_dropout',
        'hand_emb_dim': 32,
        'film_hidden_dim': 128,
        'lr': 3e-4,
        'multi_layer_film': True,
        'hand_dropout': 0.3,
        'label_smoothing': 0.1,
    },
]

WINDOW_CONFIGS = [
    # Baseline
    {
        'name': 'window_baseline',
        'window_emb_dim': 32,
        'film_hidden_dim': 128,
        'lr': 3e-4,
        'multi_layer_film': True,
        'window_dropout': 0.1,
        'label_smoothing': 0.1,
    },
    # Smaller model
    {
        'name': 'window_small',
        'window_emb_dim': 16,
        'film_hidden_dim': 64,
        'lr': 3e-4,
        'multi_layer_film': False,
        'window_dropout': 0.1,
        'label_smoothing': 0.1,
    },
    # Larger model
    {
        'name': 'window_large',
        'window_emb_dim': 64,
        'film_hidden_dim': 256,
        'lr': 1e-4,
        'multi_layer_film': True,
        'window_dropout': 0.15,
        'label_smoothing': 0.1,
    },
    # Higher learning rate
    {
        'name': 'window_high_lr',
        'window_emb_dim': 32,
        'film_hidden_dim': 128,
        'lr': 1e-3,
        'multi_layer_film': True,
        'window_dropout': 0.1,
        'label_smoothing': 0.1,
    },
]


def train_hand_config(cfg, args):
    """Train a single hand config and return results."""
    print(f"\n{'='*60}")
    print(f"Training Hand Config: {cfg['name']}")
    print(f"{'='*60}")
    print(json.dumps(cfg, indent=2))
    
    pl.seed_everything(config.get('seed', 42))
    
    # Data
    dm = HandDataModule(
        data_dir=args.data_dir,
        feature_type='pop909hand',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Model config
    m_config = {
        **model_config.get('unconditional', {}),
        'learning_rate': cfg['lr'],
        'num_hands': 2,
        'hand_emb_dim': cfg['hand_emb_dim'],
        'film_hidden_dim': cfg['film_hidden_dim'],
        'multi_layer_film': cfg['multi_layer_film'],
        'hand_dropout': cfg['hand_dropout'],
        'label_smoothing': cfg['label_smoothing'],
    }
    
    # Model
    model = HandFiLMModule(
        m_config=m_config,
        pretrained_path=args.pretrained_path,
        freeze_backbone=False
    )
    
    # Output dir
    output_dir = Path(args.output_dir) / cfg['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = CSVLogger(str(output_dir), name='logs')
    
    # Callbacks
    callbacks = configure_callbacks(
        save_dir=str(output_dir),
        prefix=cfg['name'],
        monitor='val_loss',
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        accelerator=args.device,
        enable_progress_bar=True,
    )
    
    # Train
    trainer.fit(model, dm)
    
    # Test
    test_results = trainer.test(model, dm)
    
    # Extract results
    results = {
        'config_name': cfg['name'],
        **cfg,
        'test_loss': test_results[0].get('test_loss', None),
        'test_ppl': test_results[0].get('test_ppl', None),
        'test_top1_acc': test_results[0].get('test_top1_acc', None),
        'test_top5_acc': test_results[0].get('test_top5_acc', None),
    }
    
    # For hand model, also get per-hand accuracy
    if 'test_left_acc' in test_results[0]:
        results['test_left_acc'] = test_results[0]['test_left_acc']
        results['test_right_acc'] = test_results[0]['test_right_acc']
    
    return results


def train_window_config(cfg, args):
    """Train a single window config and return results."""
    print(f"\n{'='*60}")
    print(f"Training Window Config: {cfg['name']}")
    print(f"{'='*60}")
    print(json.dumps(cfg, indent=2))
    
    pl.seed_everything(config.get('seed', 42))
    
    # Data
    dm = WindowDataModule(
        data_dir=args.data_dir,
        feature_type='pop909window',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        window_size=8
    )
    
    # Model config
    m_config = {
        **model_config.get('unconditional', {}),
        'learning_rate': cfg['lr'],
        'window_emb_dim': cfg['window_emb_dim'],
        'film_hidden_dim': cfg['film_hidden_dim'],
        'multi_layer_film': cfg['multi_layer_film'],
        'window_dropout': cfg['window_dropout'],
        'label_smoothing': cfg['label_smoothing'],
        'window_mean': 63.0,
        'window_std': 5.0,
    }
    
    # Model
    model = WindowFiLMModule(
        m_config=m_config,
        pretrained_path=args.pretrained_path,
        freeze_backbone=False
    )
    
    # Output dir
    output_dir = Path(args.output_dir) / cfg['name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger
    logger = CSVLogger(str(output_dir), name='logs')
    
    # Callbacks
    callbacks = configure_callbacks(
        save_dir=str(output_dir),
        prefix=cfg['name'],
        monitor='val_loss',
        mode='min'
    )
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=str(output_dir),
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        accelerator=args.device,
        enable_progress_bar=True,
    )
    
    # Train
    trainer.fit(model, dm)
    
    # Test
    test_results = trainer.test(model, dm)
    
    # Extract results
    results = {
        'config_name': cfg['name'],
        **cfg,
        'test_loss': test_results[0].get('test_loss', None),
        'test_ppl': test_results[0].get('test_ppl', None),
        'test_top1_acc': test_results[0].get('test_top1_acc', None),
        'test_top5_acc': test_results[0].get('test_top5_acc', None),
    }
    
    return results


def print_results_table(results, model_type):
    """Print results in a nice table format."""
    print(f"\n{'='*80}")
    print(f"{model_type.upper()} MODEL RESULTS COMPARISON")
    print(f"{'='*80}")
    
    # Header
    header = f"{'Config':<20} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Loss':<10} {'PPL':<10}"
    print(header)
    print("-" * 80)
    
    # Sort by Top-5 accuracy (descending)
    sorted_results = sorted(results, key=lambda x: x.get('test_top5_acc', 0) or 0, reverse=True)
    
    for r in sorted_results:
        top1 = r.get('test_top1_acc', 0) or 0
        top5 = r.get('test_top5_acc', 0) or 0
        loss = r.get('test_loss', 0) or 0
        ppl = r.get('test_ppl', 0) or 0
        
        row = f"{r['config_name']:<20} {top1*100:>10.2f}% {top5*100:>10.2f}% {loss:>10.4f} {ppl:>10.2f}"
        print(row)
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter sweep')
    
    # Paths
    parser.add_argument('--pretrained_path', type=str, 
                        default='uc/last-trained-on-maestro-asap.ckpt')
    parser.add_argument('--data_dir', type=str, default='yzq/output')
    parser.add_argument('--output_dir', type=str, default='yzq/checkpoints/sweep')
    
    # Training
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Epochs per config (default 30 for faster sweep)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    
    # What to run
    parser.add_argument('--model', type=str, choices=['hand', 'window', 'both'],
                        default='both', help='Which model to sweep')
    parser.add_argument('--config', type=str, default=None,
                        help='Run specific config by name (e.g., hand_baseline)')
    
    args = parser.parse_args()
    
    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Hand model sweep
    if args.model in ['hand', 'both']:
        hand_results = []
        configs_to_run = HAND_CONFIGS
        
        # Filter if specific config requested
        if args.config:
            configs_to_run = [c for c in configs_to_run if c['name'] == args.config]
        
        for cfg in configs_to_run:
            try:
                result = train_hand_config(cfg, args)
                hand_results.append(result)
            except Exception as e:
                print(f"Error training {cfg['name']}: {e}")
                continue
        
        if hand_results:
            print_results_table(hand_results, 'hand')
            all_results.extend(hand_results)
    
    # Window model sweep
    if args.model in ['window', 'both']:
        window_results = []
        configs_to_run = WINDOW_CONFIGS
        
        # Filter if specific config requested
        if args.config:
            configs_to_run = [c for c in configs_to_run if c['name'] == args.config]
        
        for cfg in configs_to_run:
            try:
                result = train_window_config(cfg, args)
                window_results.append(result)
            except Exception as e:
                print(f"Error training {cfg['name']}: {e}")
                continue
        
        if window_results:
            print_results_table(window_results, 'window')
            all_results.extend(window_results)
    
    # Save all results to JSON
    if all_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = Path(args.output_dir) / f'sweep_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

