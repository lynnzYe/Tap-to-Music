"""
Ablation study for Hand + Window FiLM conditioning.

Compares:
1. Unconditional baseline (no conditioning)
2. Hand-only FiLM (condition on left/right hand)
3. Window-only FiLM (condition on window_avg pitch)
4. Hand + Window FiLM (condition on both)

Author: YZQ
Created on: 2025/12/02
Brief: Ablation study for Hand + Window FiLM conditioning
       Compares:
       - Unconditional baseline (no conditioning)
       - Hand-only FiLM (condition on left/right hand)
       - Window-only FiLM (condition on window_avg pitch)
       - Hand + Window FiLM (condition on both)
"""
import argparse
import json
import os
from pathlib import Path
import warnings
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from ttm.config import config, model_config, MIN_PIANO_PITCH, MAX_PIANO_PITCH
from ttm.data_preparation.utils import get_note_sequence_from_midi, midi_to_tap
from ttm.model.uc_model import UCTapLSTM
from yzq.model import (
    HandFiLMModule, WindowFiLMModule, 
    configure_callbacks, parse_model_config
)
from yzq.data_preparation import predict_hands_for_midi, build_hannds_model

warnings.filterwarnings('ignore')


# ============================================================
# Combined Hand + Window Model
# ============================================================

class CombinedTapLSTM(nn.Module):
    """
    FiLM-conditioned LSTM that uses BOTH hand and window_avg features.
    Applies FiLM conditioning from both sources using additive combination.
    """
    def __init__(
        self,
        n_pitches=88,
        pitch_emb_dim=32,
        dt_dim=1,
        vel_dim=1,
        dur_dim=1,
        hidden=128,
        layers=2,
        dropout=0.15,
        linear_dim=128,
        # Hand conditioning params
        num_hands=2,
        hand_emb_dim=32,
        # Window conditioning params
        window_emb_dim=32,
        # Shared FiLM params
        film_hidden_dim=128,
        multi_layer_film=True,
        cond_dropout=0.1,
    ):
        super().__init__()
        
        # Core UCTapLSTM
        self.uc_core = UCTapLSTM(
            n_pitches=n_pitches,
            pitch_emb_dim=pitch_emb_dim,
            dt_dim=dt_dim,
            vel_dim=vel_dim,
            dur_dim=dur_dim,
            hidden=hidden,
            layers=layers,
            dropout=dropout,
            linear_dim=linear_dim,
        )
        
        self.hidden = hidden
        self.linear_dim = linear_dim
        self.multi_layer_film = multi_layer_film
        
        # Hand embedding
        self.hand_emb = nn.Embedding(num_hands, hand_emb_dim)
        self.hand_dropout = nn.Dropout(cond_dropout)
        
        # Window projection (continuous -> embedding)
        self.window_proj = nn.Linear(1, window_emb_dim)
        self.window_dropout = nn.Dropout(cond_dropout)
        
        # FiLM MLP for hand
        self.hand_film_mlp = nn.Sequential(
            nn.Linear(hand_emb_dim, film_hidden_dim),
            nn.GELU(),
            nn.Dropout(cond_dropout),
            nn.Linear(film_hidden_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(film_hidden_dim, hidden * 2),
        )
        
        # FiLM MLP for window
        self.window_film_mlp = nn.Sequential(
            nn.Linear(window_emb_dim, film_hidden_dim),
            nn.GELU(),
            nn.Dropout(cond_dropout),
            nn.Linear(film_hidden_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(film_hidden_dim, hidden * 2),
        )
        
        # Initialize FiLM to identity
        for mlp in [self.hand_film_mlp, self.window_film_mlp]:
            nn.init.constant_(mlp[-1].weight, 0)
            nn.init.constant_(mlp[-1].bias[:hidden], 1)  # gamma = 1
            nn.init.constant_(mlp[-1].bias[hidden:], 0)  # beta = 0
        
        # Input-level FiLM (optional)
        if multi_layer_film:
            self.hand_input_film = nn.Sequential(
                nn.Linear(hand_emb_dim, film_hidden_dim),
                nn.GELU(),
                nn.Linear(film_hidden_dim, linear_dim * 2),
            )
            self.window_input_film = nn.Sequential(
                nn.Linear(window_emb_dim, film_hidden_dim),
                nn.GELU(),
                nn.Linear(film_hidden_dim, linear_dim * 2),
            )
            for mlp in [self.hand_input_film, self.window_input_film]:
                nn.init.constant_(mlp[-1].weight, 0)
                nn.init.constant_(mlp[-1].bias[:linear_dim], 1)
                nn.init.constant_(mlp[-1].bias[linear_dim:], 0)
        
        self.out_head = self.uc_core.out_head
    
    def forward(self, x, hand=None, window_avg=None, hx=None):
        """
        Args:
            x: (B, T, 4 or 6) - features [pitch, dt, dur, vel] or [pitch, dt, dur, vel, hand, window_avg]
            hand: (B, T) int - hand labels (0=left, 1=right), optional if in x
            window_avg: (B, T) float - window average, optional if in x
            hx: hidden state
        """
        # Extract features
        if x.size(-1) == 6:
            x_uc = x[..., :4]
            hand = x[..., 4].long()
            window_avg = x[..., 5].float()
        elif x.size(-1) == 5:
            # Assume hand or window is provided separately
            x_uc = x[..., :4]
            if hand is None:
                hand = x[..., 4].long()
            if window_avg is None:
                window_avg = x[..., 4].float()
        else:
            x_uc = x[..., :4]
        
        # Get embeddings
        hand_emb = self.hand_emb(hand)  # (B, T, hand_emb_dim)
        hand_emb = self.hand_dropout(hand_emb)
        
        window_emb = self.window_proj(window_avg.unsqueeze(-1))  # (B, T, window_emb_dim)
        window_emb = self.window_dropout(window_emb)
        
        # Forward through input layers
        pitch_idx = x_uc[..., 0].long()
        dt = x_uc[..., 1]
        dur = x_uc[..., 2]
        vel = x_uc[..., 3]
        
        pe = self.uc_core.pitch_emb(pitch_idx)
        feats = [pe, dt.unsqueeze(-1), dur.unsqueeze(-1), vel.unsqueeze(-1)]
        x_in = torch.cat(feats, dim=-1).to(self.uc_core.input_linear.weight.dtype)
        x_linear = self.uc_core.input_linear(x_in)
        
        # Apply input-level FiLM
        if self.multi_layer_film:
            # Hand FiLM on input
            hand_input_params = self.hand_input_film(hand_emb)
            h_gamma_in, h_beta_in = torch.chunk(hand_input_params, 2, dim=-1)
            
            # Window FiLM on input
            window_input_params = self.window_input_film(window_emb)
            w_gamma_in, w_beta_in = torch.chunk(window_input_params, 2, dim=-1)
            
            # Additive combination of FiLM parameters
            gamma_in = (h_gamma_in + w_gamma_in) / 2
            beta_in = (h_beta_in + w_beta_in) / 2
            
            x_linear = gamma_in * x_linear + beta_in
        
        # LSTM forward
        out_lstm, (h, c) = self.uc_core.lstm(x_linear, hx)
        
        # Output-level FiLM
        hand_film_params = self.hand_film_mlp(hand_emb)
        h_gamma, h_beta = torch.chunk(hand_film_params, 2, dim=-1)
        
        window_film_params = self.window_film_mlp(window_emb)
        w_gamma, w_beta = torch.chunk(window_film_params, 2, dim=-1)
        
        # Additive combination of FiLM parameters
        gamma = (h_gamma + w_gamma) / 2
        beta = (h_beta + w_beta) / 2
        
        out_cond = gamma * out_lstm + beta
        
        y_pitch = self.out_head(out_cond)
        return y_pitch, (h, c)
    
    @classmethod
    def from_pretrained_uc(cls, uc_state_dict, **kwargs):
        """Create CombinedTapLSTM from pretrained UCTapLSTM."""
        temp_uc = UCTapLSTM()
        temp_sd = temp_uc.state_dict()
        
        cleaned_uc_sd = {}
        for k, v in uc_state_dict.items():
            new_k = k.replace("model.", "")
            if new_k in temp_sd and temp_sd[new_k].shape == v.shape:
                cleaned_uc_sd[new_k] = v
        
        model = cls(
            n_pitches=temp_uc.n_pitches - 1,
            pitch_emb_dim=temp_uc.pitch_emb_dim,
            hidden=temp_uc.hidden,
            layers=temp_uc.layers,
            dropout=temp_uc.dropout,
            linear_dim=temp_uc.linear_dim,
            **kwargs
        )
        
        model.uc_core.load_state_dict(cleaned_uc_sd, strict=False)
        return model


class CombinedFiLMModule(pl.LightningModule):
    """PyTorch Lightning module for Combined Hand+Window FiLM model."""
    
    def __init__(self, m_config, pretrained_path=None, freeze_backbone=False):
        super().__init__()
        self.config = m_config
        self.save_hyperparameters()
        
        if pretrained_path:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('state_dict', checkpoint)
            
            self.model = CombinedTapLSTM.from_pretrained_uc(
                state_dict,
                num_hands=m_config.get('num_hands', 2),
                hand_emb_dim=m_config.get('hand_emb_dim', 32),
                window_emb_dim=m_config.get('window_emb_dim', 32),
                film_hidden_dim=m_config.get('film_hidden_dim', 128),
                multi_layer_film=m_config.get('multi_layer_film', True),
                cond_dropout=m_config.get('cond_dropout', 0.1),
            )
            
            if freeze_backbone:
                for name, param in self.model.named_parameters():
                    if 'film' not in name and 'hand' not in name and 'window' not in name:
                        param.requires_grad = False
        else:
            self.model = CombinedTapLSTM(**parse_model_config(m_config, CombinedTapLSTM))
        
        self.test_results = []
    
    def forward(self, x, hand=None, window_avg=None, hx=None):
        return self.model(x, hand, window_avg, hx)
    
    def configure_optimizers(self):
        lr = float(self.config.get('learning_rate', 3e-4))
        
        pretrained_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(k in name for k in ['film', 'hand', 'window']):
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        param_groups = [
            {'params': pretrained_params, 'lr': lr * 0.1},
            {'params': new_params, 'lr': lr}
        ]
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        optimizer = torch.optim.AdamW(param_groups, betas=(0.8, 0.8), eps=1e-4, weight_decay=1e-3)
        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        features, labels = batch
        pitch_logits, _ = self(features)
        
        B, L, C = pitch_logits.shape
        label_smoothing = self.config.get('label_smoothing', 0.1)
        ce = F.cross_entropy(
            pitch_logits.reshape(B * L, C),
            labels.reshape(B * L).long(),
            reduction='mean',
            ignore_index=88,
            label_smoothing=label_smoothing
        )
        
        # Compute training accuracy
        with torch.no_grad():
            preds = pitch_logits.argmax(dim=-1)
            mask = labels != 88
            correct = (preds == labels) & mask
            train_acc = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
            
            top5_preds = pitch_logits.topk(5, dim=-1).indices
            top5_correct = (top5_preds == labels.unsqueeze(-1)).any(dim=-1) & mask
            train_top5_acc = top5_correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        self.log('train_loss', ce, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_acc', train_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_top5_acc', train_top5_acc, logger=True, on_step=False, on_epoch=True)
        return {'loss': ce}
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch
        pitch_logits, _ = self(features)
        
        B, L, C = pitch_logits.shape
        ce = F.cross_entropy(
            pitch_logits.reshape(B * L, C),
            labels.reshape(B * L).long(),
            reduction='mean',
            ignore_index=88
        )
        
        preds = pitch_logits.argmax(dim=-1)
        mask = labels != 88
        correct = (preds == labels) & mask
        top1_acc = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        top5_preds = pitch_logits.topk(5, dim=-1).indices
        top5_correct = (top5_preds == labels.unsqueeze(-1)).any(dim=-1) & mask
        top5_acc = top5_correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        ppl = torch.exp(ce)
        
        self.log('val_loss', ce, prog_bar=True, logger=True)
        self.log('val_ppl', ppl, prog_bar=True, logger=True)
        self.log('val_top1_acc', top1_acc, prog_bar=True, logger=True)
        self.log('val_top5_acc', top5_acc, prog_bar=True, logger=True)
        
        return {'val_loss': ce}
    
    def test_step(self, batch, batch_idx):
        features, labels = batch
        pitch_logits, _ = self(features)
        
        B, L, C = pitch_logits.shape
        ce = F.cross_entropy(
            pitch_logits.reshape(B * L, C),
            labels.reshape(B * L).long(),
            reduction='mean',
            ignore_index=88
        )
        
        preds = pitch_logits.argmax(dim=-1)
        mask = labels != 88
        correct = (preds == labels) & mask
        top1_acc = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        top5_preds = pitch_logits.topk(5, dim=-1).indices
        top5_correct = (top5_preds == labels.unsqueeze(-1)).any(dim=-1) & mask
        top5_acc = top5_correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        self.test_results.append({
            'loss': ce.item(),
            'top1_acc': top1_acc.item(),
            'top5_acc': top5_acc.item(),
            'n_samples': mask.sum().item()
        })
        
        return {'test_loss': ce}
    
    def on_test_epoch_end(self):
        if not self.test_results:
            return
        
        total_samples = sum(r['n_samples'] for r in self.test_results)
        avg_loss = sum(r['loss'] * r['n_samples'] for r in self.test_results) / total_samples
        avg_top1 = sum(r['top1_acc'] * r['n_samples'] for r in self.test_results) / total_samples
        avg_top5 = sum(r['top5_acc'] * r['n_samples'] for r in self.test_results) / total_samples
        avg_ppl = np.exp(avg_loss)
        
        self.log('test_loss', avg_loss)
        self.log('test_ppl', avg_ppl)
        self.log('test_top1_acc', avg_top1)
        self.log('test_top5_acc', avg_top5)
        
        print(f"\n{'='*60}")
        print(f"Combined FiLM Test Results")
        print(f"{'='*60}")
        print(f"  test_loss: {avg_loss:.4f}")
        print(f"  test_ppl: {avg_ppl:.4f}")
        print(f"  test_top1_acc: {avg_top1:.4f}")
        print(f"  test_top5_acc: {avg_top5:.4f}")
        print(f"{'='*60}")
        
        self.test_results = []


# ============================================================
# Combined Dataset
# ============================================================

def compute_window_avg(pitches, window_size):
    """Compute bidirectional window average (past n + future n)."""
    n = len(pitches)
    window_avg = np.zeros(n, dtype=float)
    
    for i in range(n):
        past_start = max(0, i - window_size)
        future_end = min(n, i + 1 + window_size)
        
        past_notes = pitches[past_start:i]
        future_notes = pitches[i+1:future_end]
        
        combined = np.concatenate([past_notes, future_notes])
        
        if len(combined) == 0:
            window_avg[i] = pitches[i]
        else:
            window_avg[i] = np.mean(combined)
    
    return window_avg


def prepare_combined_dataset(pop909_dir, output_dir, hannds_checkpoint, window_size=8,
                              split_ratios=(0.8, 0.1, 0.1), device='cpu'):
    """
    Prepare dataset with both hand and window_avg features.
    Features: [pitch, log_dt, log_dur, vel, hand, window_avg] (6 features)
    """
    from tqdm import tqdm
    import random
    
    pop909_dir = Path(pop909_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build HANNDs model for hand prediction
    print("Loading HANNDs model...")
    hannds_model = build_hannds_model(device, hannds_checkpoint)
    
    # Collect MIDI files
    midi_files = []
    pop909_main = pop909_dir / "POP909"
    if pop909_main.exists():
        for song_dir in sorted(pop909_main.iterdir()):
            if song_dir.is_dir():
                for midi_file in song_dir.glob("*.mid"):
                    midi_files.append(midi_file)
    
    print(f"Found {len(midi_files)} MIDI files")
    
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
    
    prefix = "pop909combined"
    
    for split_name, files in splits.items():
        data = []
        for midi_path in tqdm(files, desc=f"Processing {split_name}"):
            try:
                # Get basic features
                note_sequence = get_note_sequence_from_midi(str(midi_path))
                if len(note_sequence) < 2:
                    continue
                features, labels = midi_to_tap(note_sequence)
                
                # Get hand labels
                notes, hand_labels_str = predict_hands_for_midi(hannds_model, device, str(midi_path))
                hand_labels = np.array([0 if h == 'L' else 1 for h in hand_labels_str])
                
                # Align hand labels with features
                if len(hand_labels) != len(features):
                    # Simple fallback: use pitch-based heuristic
                    hand_labels = (features[:, 0] >= 60).astype(int)
                
                # Compute window average
                window_avg = compute_window_avg(labels, window_size)
                
                # Combine: [pitch, dt, dur, vel, hand, window_avg]
                combined_features = np.column_stack([features, hand_labels, window_avg])
                
                data.append((combined_features, labels))
                
            except Exception as e:
                continue
        
        # Save
        output_path = output_dir / f"{prefix}-{split_name}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} samples to {output_path}")


class CombinedDataset(torch.utils.data.Dataset):
    """Dataset with both hand and window_avg features."""
    
    def __init__(self, data_dir, split, feature_type='pop909combined'):
        self.split = split
        pkl_path = Path(data_dir) / f"{feature_type}-{split}.pkl"
        print(f"Loading {pkl_path}")
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.max_length = 128
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features, labels = self.data[idx]
        features = features.copy()
        labels = labels.copy()
        
        # Normalize pitch
        features[:, 0] -= MIN_PIANO_PITCH
        labels -= MIN_PIANO_PITCH
        
        # Pad if needed
        if self.split != 'test' and len(features) < self.max_length:
            pad_len = self.max_length - len(features)
            hand_pad = features[0, 4] if len(features) > 0 else 0
            window_pad = features[0, 5] if len(features) > 0 else 0
            pad_row = np.array([[88, 0, 0, 0, hand_pad, window_pad]])
            pad_block = np.repeat(pad_row, pad_len, axis=0)
            features = np.concatenate([features, pad_block], axis=0)
            labels = np.concatenate([labels, np.full(pad_len, 88)])
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class CombinedDataModule(pl.LightningDataModule):
    """DataModule for combined hand+window dataset."""
    
    def __init__(self, data_dir, feature_type='pop909combined', batch_size=32, num_workers=0):
        super().__init__()
        self.data_dir = data_dir
        self.feature_type = feature_type
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CombinedDataset(self.data_dir, 'train', self.feature_type)
            self.val_dataset = CombinedDataset(self.data_dir, 'validation', self.feature_type)
        if stage == 'test' or stage is None:
            self.test_dataset = CombinedDataset(self.data_dir, 'test', self.feature_type)
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self):
        # Use batch_size=1 to handle variable length sequences
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False,
            num_workers=self.num_workers, drop_last=False
        )
    
    def _collate_fn(self, batch):
        """Custom collate to handle variable length sequences in train/val."""
        max_len = max(x[0].shape[0] for x in batch)
        features_batch = []
        labels_batch = []
        for features, labels in batch:
            if features.shape[0] < max_len:
                pad_len = max_len - features.shape[0]
                hand_pad = features[0, 4].item() if features.shape[0] > 0 else 0
                window_pad = features[0, 5].item() if features.shape[0] > 0 else 0
                pad_row = torch.tensor([[88, 0, 0, 0, hand_pad, window_pad]] * pad_len, dtype=features.dtype)
                features = torch.cat([features, pad_row], dim=0)
                labels = torch.cat([labels, torch.full((pad_len,), 88, dtype=labels.dtype)])
            features_batch.append(features)
            labels_batch.append(labels)
        return torch.stack(features_batch), torch.stack(labels_batch)


# ============================================================
# Ablation Runner
# ============================================================

def run_ablation(args):
    """Run full ablation study comparing all model variants."""
    
    results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model configurations to test
    ablation_configs = {
        'hand_only': {
            'model_class': 'HandFiLMModule',
            'data_module_class': 'hand',
            'description': 'Hand FiLM only (left/right hand conditioning)',
        },
        'window_only': {
            'model_class': 'WindowFiLMModule',
            'data_module_class': 'window',
            'description': 'Window FiLM only (pitch context conditioning)',
        },
        'combined': {
            'model_class': 'CombinedFiLMModule',
            'data_module_class': 'combined',
            'description': 'Combined Hand+Window FiLM (additive)',
        },
    }
    
    # Filter configs if specific ones requested
    if args.configs:
        config_names = args.configs.split(',')
        ablation_configs = {k: v for k, v in ablation_configs.items() if k in config_names}
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY: Hand + Window FiLM Conditioning")
    print(f"{'='*70}")
    print(f"Configs to test: {list(ablation_configs.keys())}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"{'='*70}\n")
    
    for config_name, config in ablation_configs.items():
        print(f"\n{'='*60}")
        print(f"Training: {config_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        pl.seed_everything(42)
        
        try:
            # Setup data module
            if config['data_module_class'] == 'hand':
                from yzq.data_module import HandDataModule
                dm = HandDataModule(args.data_dir, feature_type='pop909hand',
                                   batch_size=args.batch_size, num_workers=args.num_workers)
            elif config['data_module_class'] == 'window':
                from yzq.data_module import WindowDataModule
                dm = WindowDataModule(args.data_dir, feature_type='pop909window',
                                     batch_size=args.batch_size, num_workers=args.num_workers,
                                     window_size=8)
            else:  # combined
                dm = CombinedDataModule(args.data_dir, feature_type='pop909combined',
                                       batch_size=args.batch_size, num_workers=args.num_workers)
            
            # Setup model
            m_config = {
                **model_config.get('unconditional', {}),
                'learning_rate': args.lr,
                'num_hands': 2,
                'hand_emb_dim': 32,
                'window_emb_dim': 32,
                'film_hidden_dim': 128,
                'multi_layer_film': True,
                'label_smoothing': 0.1,
                'cond_dropout': 0.1,
            }
            
            if config['model_class'] == 'HandFiLMModule':
                model = HandFiLMModule(
                    m_config=m_config,
                    pretrained_path=args.pretrained_path,
                    freeze_backbone=args.freeze_backbone
                )
            elif config['model_class'] == 'WindowFiLMModule':
                model = WindowFiLMModule(
                    m_config=m_config,
                    pretrained_path=args.pretrained_path,
                    freeze_backbone=args.freeze_backbone
                )
            else:  # CombinedFiLMModule
                model = CombinedFiLMModule(
                    m_config=m_config,
                    pretrained_path=args.pretrained_path,
                    freeze_backbone=args.freeze_backbone
                )
            
            # Logger
            run_dir = output_dir / config_name
            run_dir.mkdir(parents=True, exist_ok=True)
            
            if args.use_wandb:
                logger = WandbLogger(
                    project='tap-to-music-ablation',
                    name=config_name,
                    save_dir=str(run_dir),
                    config={'config_name': config_name, **m_config}
                )
            else:
                logger = CSVLogger(str(run_dir), name='logs')
            
            # Trainer
            callbacks = configure_callbacks(
                save_dir=str(run_dir),
                prefix=config_name,
                monitor='val_loss',
                mode='min'
            )
            
            trainer = pl.Trainer(
                default_root_dir=str(run_dir),
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
            
            # Record results
            result = {
                'config_name': config_name,
                'description': config['description'],
                'test_loss': test_results[0].get('test_loss'),
                'test_ppl': test_results[0].get('test_ppl'),
                'test_top1_acc': test_results[0].get('test_top1_acc'),
                'test_top5_acc': test_results[0].get('test_top5_acc'),
            }
            results.append(result)
            
            if args.use_wandb:
                import wandb
                wandb.finish()
                
        except Exception as e:
            print(f"Error with {config_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print results table
    print(f"\n{'='*80}")
    print("ABLATION STUDY RESULTS")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Top-1 Acc':<12} {'Top-5 Acc':<12} {'Loss':<10} {'PPL':<10}")
    print(f"{'-'*80}")
    
    for r in sorted(results, key=lambda x: x.get('test_top5_acc', 0) or 0, reverse=True):
        top1 = (r.get('test_top1_acc', 0) or 0) * 100
        top5 = (r.get('test_top5_acc', 0) or 0) * 100
        loss = r.get('test_loss', 0) or 0
        ppl = r.get('test_ppl', 0) or 0
        print(f"{r['config_name']:<25} {top1:>10.2f}% {top5:>10.2f}% {loss:>10.4f} {ppl:>10.2f}")
    
    print(f"{'='*80}")
    
    # Save results
    results_file = output_dir / 'ablation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ablation study for Hand+Window FiLM')
    
    parser.add_argument('--pop909_dir', type=str, default='POP909-Dataset')
    parser.add_argument('--data_dir', type=str, default='yzq/output')
    parser.add_argument('--output_dir', type=str, default='yzq/checkpoints/ablation')
    parser.add_argument('--pretrained_path', type=str, 
                        default='uc/unconditional-epoch=979-val_loss=2.5250-val_top5_acc=0.6619.ckpt')
    parser.add_argument('--hannds_checkpoint', type=str,
                        default='hannds/pretrained/model_checkpoint.pt')
    parser.add_argument('--max_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_api_key', type=str, default=None)
    parser.add_argument('--prepare_data', action='store_true',
                        help='Prepare combined dataset before training')
    parser.add_argument('--configs', type=str, default=None,
                        help='Comma-separated config names to run (default: all)')
    parser.add_argument('--window_size', type=int, default=8)
    
    args = parser.parse_args()
    
    # Setup wandb
    if args.wandb_api_key:
        os.environ['WANDB_API_KEY'] = args.wandb_api_key
    
    if args.use_wandb:
        import wandb
        wandb.login()
    
    # Prepare combined dataset if requested
    if args.prepare_data:
        print("Preparing combined dataset...")
        prepare_combined_dataset(
            args.pop909_dir,
            args.data_dir,
            args.hannds_checkpoint,
            window_size=args.window_size,
            device=args.device
        )
    
    # Run ablation
    run_ablation(args)


if __name__ == "__main__":
    main()

