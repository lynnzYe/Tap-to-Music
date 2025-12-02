"""
Test different window sizes for window_avg feature.
Generates datasets and trains models with different window sizes.
"""
import argparse
import json
import os
from pathlib import Path
import warnings
import pickle
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from ttm.config import config, model_config
from ttm.data_preparation.utils import get_note_sequence_from_midi, midi_to_tap
from yzq.model import WindowFiLMModule, configure_callbacks

warnings.filterwarnings('ignore')

# Window sizes to test
WINDOW_SIZES = [2, 4, 8, 16, 32]


def compute_window_avg(labels, window_size):
    """Compute bidirectional window average of pitches (past n + future n)."""
    n = len(labels)
    window_avg = np.zeros(n, dtype=float)
    
    for i in range(n):
        past_start = max(0, i - window_size)
        future_end = min(n, i + 1 + window_size)
        
        past_notes = labels[past_start:i]
        future_notes = labels[i+1:future_end]
        
        combined = np.concatenate([past_notes, future_notes])
        
        if len(combined) == 0:
            window_avg[i] = labels[i]
        else:
            window_avg[i] = np.mean(combined)
    
    return window_avg


def generate_window_data(pop909_dir, output_dir, window_size, split_ratios=(0.8, 0.1, 0.1)):
    """Generate dataset with specific window size."""
    from pathlib import Path
    import random
    from tqdm import tqdm
    
    pop909_dir = Path(pop909_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"pop909window{window_size}"
    
    # Collect all MIDI files
    midi_files = []
    pop909_main = pop909_dir / "POP909"
    if pop909_main.exists():
        for song_dir in sorted(pop909_main.iterdir()):
            if song_dir.is_dir():
                for midi_file in song_dir.glob("*.mid"):
                    midi_files.append(midi_file)
    
    print(f"Found {len(midi_files)} MIDI files for window_size={window_size}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(midi_files)
    
    n = len(midi_files)
    train_end = int(n * split_ratios[0])
    val_end = train_end + int(n * split_ratios[1])
    
    splits = {
        'train': midi_files[:train_end],
        'validation': midi_files[train_end:val_end],
        'test': midi_files[val_end:]
    }
    
    # Process each split
    for split_name, files in splits.items():
        data = []
        for midi_path in tqdm(files, desc=f"Processing {split_name} (ws={window_size})"):
            try:
                note_sequence = get_note_sequence_from_midi(str(midi_path))
                if len(note_sequence) < 2:
                    continue
                features, labels = midi_to_tap(note_sequence)
                
                # Compute window average with specified size
                window_avg = compute_window_avg(labels, window_size)
                features = np.column_stack([features, window_avg])
                
                data.append((features, labels))
            except Exception as e:
                continue
        
        # Save
        output_path = output_dir / f"{prefix}-{split_name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} samples to {output_path}")
    
    # Compute statistics for this window size
    all_window_avg = []
    for features, _ in data:
        all_window_avg.extend(features[:, 4])
    
    arr = np.array(all_window_avg)
    stats = {
        'window_size': window_size,
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }
    print(f"Window size {window_size}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    return stats


class WindowDatasetCustom:
    """Custom dataset loader for different window sizes."""
    def __init__(self, data_dir, split, window_size):
        self.prefix = f"pop909window{window_size}"
        pkl_path = Path(data_dir) / f"{self.prefix}-{split}.pkl"
        print(f"Loading {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.split = split
        self.max_length = 128
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        import torch
        from ttm.config import MIN_PIANO_PITCH
        
        features, labels = self.data[idx]
        features = features.copy()
        labels = labels.copy()
        
        # Normalize pitch
        features[:, 0] -= MIN_PIANO_PITCH
        labels -= MIN_PIANO_PITCH
        
        # Pad if needed for train/val
        if self.split != 'test' and len(features) < self.max_length:
            pad_len = self.max_length - len(features)
            window_avg_for_pad = features[0, 4] if len(features) > 0 else 0
            pad_row = np.array([[88, 0, 0, 0, window_avg_for_pad]])
            pad_block = np.repeat(pad_row, pad_len, axis=0)
            features = np.concatenate([features, pad_block], axis=0)
            labels = np.concatenate([labels, np.full(pad_len, 88)])
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class WindowDataModuleCustom(pl.LightningDataModule):
    """DataModule for custom window sizes."""
    def __init__(self, data_dir, window_size, batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = WindowDatasetCustom(self.data_dir, 'train', self.window_size)
            self.val_dataset = WindowDatasetCustom(self.data_dir, 'validation', self.window_size)
        if stage == 'test' or stage is None:
            self.test_dataset = WindowDatasetCustom(self.data_dir, 'test', self.window_size)
    
    def train_dataloader(self):
        import torch
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        import torch
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate to handle variable length sequences."""
        import torch
        # Pad sequences to same length within batch
        max_len = max(x[0].shape[0] for x in batch)
        features_batch = []
        labels_batch = []
        for features, labels in batch:
            if features.shape[0] < max_len:
                pad_len = max_len - features.shape[0]
                window_avg_pad = features[0, 4].item() if features.shape[0] > 0 else 0
                pad_features = torch.tensor([[88, 0, 0, 0, window_avg_pad]] * pad_len, dtype=features.dtype)
                features = torch.cat([features, pad_features], dim=0)
                labels = torch.cat([labels, torch.full((pad_len,), 88, dtype=labels.dtype)])
            features_batch.append(features)
            labels_batch.append(labels)
        return torch.stack(features_batch), torch.stack(labels_batch)
    
    def test_dataloader(self):
        import torch
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False,
            num_workers=self.num_workers, drop_last=False
        )


def train_window_size(window_size, stats, args):
    """Train model with specific window size."""
    print(f"\n{'='*60}")
    print(f"Training Window Size: {window_size}")
    print(f"{'='*60}")
    
    pl.seed_everything(42)
    
    # Data
    dm = WindowDataModuleCustom(
        data_dir=args.data_dir,
        window_size=window_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Model config
    m_config = {
        **model_config.get('unconditional', {}),
        'learning_rate': 3e-4,
        'window_emb_dim': 32,
        'film_hidden_dim': 128,
        'multi_layer_film': True,
        'window_dropout': 0.1,
        'label_smoothing': 0.1,
        'window_mean': stats['mean'],
        'window_std': stats['std'],
    }
    
    # Model
    model = WindowFiLMModule(
        m_config=m_config,
        pretrained_path=args.pretrained_path,
        freeze_backbone=False
    )
    
    # Output
    output_dir = Path(args.output_dir) / f"window_size_{window_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logger - use Wandb if enabled, otherwise CSVLogger
    if args.use_wandb:
        import wandb
        logger = WandbLogger(
            project=args.wandb_project,
            name=f'window_size_{window_size}',
            save_dir=str(output_dir),
            config={
                'window_size': window_size,
                'window_mean': stats['mean'],
                'window_std': stats['std'],
                'batch_size': args.batch_size,
                'max_epochs': args.max_epochs,
                'learning_rate': 3e-4,
                'window_emb_dim': 32,
                'film_hidden_dim': 128,
            },
            tags=['window_size_test', f'ws{window_size}'],
        )
    else:
        logger = CSVLogger(str(output_dir), name='logs')
    
    # Callbacks
    callbacks = configure_callbacks(
        save_dir=str(output_dir),
        prefix=f'ws{window_size}',
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
    
    results = {
        'window_size': window_size,
        'window_mean': stats['mean'],
        'window_std': stats['std'],
        'test_loss': test_results[0].get('test_loss', None),
        'test_ppl': test_results[0].get('test_ppl', None),
        'test_top1_acc': test_results[0].get('test_top1_acc', None),
        'test_top5_acc': test_results[0].get('test_top5_acc', None),
    }
    
    # Finish wandb run
    if args.use_wandb:
        import wandb
        wandb.finish()
    
    return results


def print_results_table(results):
    """Print results table."""
    print(f"\n{'='*80}")
    print("WINDOW SIZE COMPARISON RESULTS")
    print(f"{'='*80}")
    
    header = f"{'Window Size':<15} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Loss':<10} {'PPL':<10} {'Std':<10}"
    print(header)
    print("-" * 80)
    
    # Sort by Top-5 accuracy
    sorted_results = sorted(results, key=lambda x: x.get('test_top5_acc', 0) or 0, reverse=True)
    
    for r in sorted_results:
        top1 = (r.get('test_top1_acc', 0) or 0) * 100
        top5 = (r.get('test_top5_acc', 0) or 0) * 100
        loss = r.get('test_loss', 0) or 0
        ppl = r.get('test_ppl', 0) or 0
        std = r.get('window_std', 0) or 0
        
        row = f"{r['window_size']:<15} {top1:>10.2f}% {top5:>10.2f}% {loss:>10.4f} {ppl:>10.2f} {std:>10.2f}"
        print(row)
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test different window sizes')
    
    parser.add_argument('--pop909_dir', type=str, default='POP909-Dataset')
    parser.add_argument('--data_dir', type=str, default='yzq/output')
    parser.add_argument('--output_dir', type=str, default='yzq/checkpoints/window_sizes')
    parser.add_argument('--pretrained_path', type=str, 
                        default='uc/last-trained-on-maestro-asap.ckpt')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--window_sizes', type=str, default='2,4,8,16,32',
                        help='Comma-separated window sizes to test')
    parser.add_argument('--skip_data_gen', action='store_true',
                        help='Skip data generation if already done')
    # Wandb options
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='tap-to-music-window-size',
                        help='Wandb project name')
    parser.add_argument('--wandb_api_key', type=str, default=None,
                        help='Wandb API key (can also use WANDB_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Setup wandb API key if provided
    if args.wandb_api_key:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key
    
    # Login to wandb if using it
    if args.use_wandb:
        import wandb
        if not os.environ.get('WANDB_API_KEY') and not args.wandb_api_key:
            print("Warning: No WANDB_API_KEY set. Please login manually or provide --wandb_api_key")
        wandb.login()
    
    # Parse window sizes
    window_sizes = [int(x) for x in args.window_sizes.split(',')]
    print(f"Testing window sizes: {window_sizes}")
    
    # Generate datasets for each window size
    stats_dict = {}
    if not args.skip_data_gen:
        for ws in window_sizes:
            stats = generate_window_data(args.pop909_dir, args.data_dir, ws)
            stats_dict[ws] = stats
    else:
        # Use default stats if skipping
        for ws in window_sizes:
            stats_dict[ws] = {'mean': 63.0, 'std': 5.0, 'window_size': ws}
    
    # Train models for each window size
    all_results = []
    for ws in window_sizes:
        try:
            result = train_window_size(ws, stats_dict[ws], args)
            all_results.append(result)
        except Exception as e:
            print(f"Error with window size {ws}: {e}")
            continue
    
    # Print results
    if all_results:
        print_results_table(all_results)
        
        # Save results
        results_file = Path(args.output_dir) / 'window_size_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()

