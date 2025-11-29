"""
Author: YZQ
Created on: 2025/11/27
Brief: FiLM-conditioned models for finetuning with hand feature
       
       FiLM (Feature-wise Linear Modulation) applies:
       output = gamma * features + beta
       where gamma and beta are learned from the conditioning signal (hand).
       
       Reference: https://github.com/lynnzYe/Tap-to-Music/blob/9bf3016803e6038b302265dabbd68414c9a71b0a/ttm/model/chord_model.py
       
       Models:
       - HandTapLSTM: LSTM with FiLM conditioning on hand feature
       - HandFiLMModule: PyTorch Lightning module for training/testing with wandb
"""
import inspect
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from ttm.config import model_config
from ttm.model.uc_model import UCTapLSTM


def parse_model_config(config_dict, model_type):
    """Parse config dict to only include valid model parameters."""
    sig = inspect.signature(model_type.__init__)
    allowed_keys = [p for p in sig.parameters if p != "self"]
    filtered_config = {k: v for k, v in config_dict.items() if k in allowed_keys}
    return filtered_config


class HandTapLSTM(nn.Module):
    """
    LSTM model with FiLM conditioning on hand feature.
    
    Similar to ChordTapLSTM but conditioned on hand (0=left, 1=right).
    Uses FiLM (Feature-wise Linear Modulation) to modulate LSTM output.
    
    Input: (B, T, 5) - [pitch, dt, dur, vel, hand]
    Output: (B, T, 89) - pitch logits
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
        num_hands=2,            # 0=left, 1=right
        hand_emb_dim=16,
        film_hidden_dim=64,
        # New improvements
        multi_layer_film=False,  # Apply FiLM at input level too
        hand_dropout=0.1,        # Dropout on hand embedding
    ):
        super().__init__()
        
        # Wrap UCTapLSTM as core (same pattern as ChordTapLSTM)
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
        self.num_hands = num_hands
        self.hand_emb_dim = hand_emb_dim
        self.film_hidden_dim = film_hidden_dim
        self.multi_layer_film = multi_layer_film
        
        # Hand embedding with larger capacity
        self.hand_emb = nn.Embedding(num_hands, hand_emb_dim)
        self.hand_dropout = nn.Dropout(hand_dropout)
        
        # FiLM-like conditioning: map hand embedding → (gamma, beta)
        self.film_mlp = nn.Sequential(
            nn.Linear(hand_emb_dim, film_hidden_dim),
            nn.GELU(),  # GELU often works better than ReLU
            nn.Dropout(hand_dropout),
            nn.Linear(film_hidden_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(film_hidden_dim, hidden * 2),  # gamma and beta
        )
        
        # Optional: Multi-layer FiLM (also modulate input)
        if multi_layer_film:
            input_dim = pitch_emb_dim + 3  # pitch_emb + dt + dur + vel
            self.input_film_mlp = nn.Sequential(
                nn.Linear(hand_emb_dim, film_hidden_dim),
                nn.GELU(),
                nn.Linear(film_hidden_dim, hidden * 2),  # For input_linear output
            )
        
        # Use same output head as uc_core
        self.out_head = self.uc_core.out_head
        
        # Initialize FiLM to identity (gamma=1, beta=0)
        self._init_film_weights()
    
    def _init_film_weights(self):
        """Initialize FiLM layers to identity transformation (gamma=1, beta=0)."""
        # Initialize last layer of film_mlp to output gamma=1, beta=0
        with torch.no_grad():
            last_layer = self.film_mlp[-1]
            last_layer.weight.zero_()
            # First half of bias (gamma) = 1, second half (beta) = 0
            last_layer.bias.zero_()
            last_layer.bias[:self.hidden] = 1.0  # gamma
            
            if self.multi_layer_film:
                last_layer = self.input_film_mlp[-1]
                last_layer.weight.zero_()
                last_layer.bias.zero_()
                last_layer.bias[:self.hidden] = 1.0
    
    def _uc_forward_features(self, x, hand_emb=None, hx=None):
        """Get LSTM features from unconditional core (without output head)."""
        pitch_idx = x[..., 0].long()
        dt = x[..., 1]
        dur = x[..., 2]
        vel = x[..., 3]
        
        pe = self.uc_core.pitch_emb(pitch_idx)
        feats = [pe, dt.unsqueeze(-1), dur.unsqueeze(-1), vel.unsqueeze(-1)]
        x_in = torch.cat(feats, dim=-1).to(self.uc_core.input_linear.weight.dtype)
        
        x_in = self.uc_core.input_linear(x_in)  # (B, T, hidden)
        
        # Multi-layer FiLM: modulate input before LSTM
        if self.multi_layer_film and hand_emb is not None:
            input_film_params = self.input_film_mlp(hand_emb)
            gamma_in, beta_in = torch.chunk(input_film_params, 2, dim=-1)
            x_in = gamma_in * x_in + beta_in
        
        out_uc, (h, c) = self.uc_core.lstm(x_in, hx)  # out_uc: (B, T, hidden)
        
        return out_uc, (h, c)
    
    def forward(self, x, hand_id=None, hx=None):
        """
        Forward pass.
        
        Args:
            x: (B, T, 5) with hand as 5th feature, or (B, T, 4) if hand_id provided
            hand_id: Optional (B, T) hand labels if not in x
            hx: Optional LSTM hidden state
        
        Returns:
            y_pitch: (B, T, 89) - pitch logits
            (h, c): LSTM hidden state
        """
        # Extract hand from input if included as 5th feature
        if x.size(-1) == 5 and hand_id is None:
            x_uc = x[..., :4]
            hand_id = x[..., 4].long()
        else:
            x_uc = x[..., :4]
            assert hand_id is not None, "hand_id must be provided if x does not include it as 5th feature."
            hand_id = hand_id.long()
        
        # Get hand embedding with dropout
        hand_emb = self.hand_emb(hand_id)  # (B, T, hand_emb_dim)
        hand_emb = self.hand_dropout(hand_emb)
        
        # Get LSTM features from unconditional core (with optional multi-layer FiLM)
        out_uc, (h, c) = self._uc_forward_features(x_uc, hand_emb=hand_emb, hx=hx)
        
        # Generate FiLM parameters
        film_params = self.film_mlp(hand_emb)  # (B, T, hidden * 2)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # Each (B, T, hidden)
        
        # Apply FiLM modulation
        out_cond = gamma * out_uc + beta  # (B, T, hidden)
        
        # Output projection - returns pitch logits only
        y_pitch = self.out_head(out_cond)  # (B, T, 89)
        
        return y_pitch, (h, c)
    
    @classmethod
    def from_pretrained_uc(
        cls,
        uc_state_dict,
        num_hands=2,
        hand_emb_dim=16,
        film_hidden_dim=64,
        multi_layer_film=False,
        hand_dropout=0.1,
        **uc_kwargs,
    ):
        """
        Create HandTapLSTM from pretrained UCTapLSTM state dict.
        
        Args:
            uc_state_dict: State dict from trained UCModule/UCTapLSTM
            num_hands: Number of hand classes (default 2)
            hand_emb_dim: Hand embedding dimension
            film_hidden_dim: FiLM MLP hidden dimension
            multi_layer_film: Apply FiLM at input level too
            hand_dropout: Dropout on hand embedding
            **uc_kwargs: Arguments for UCTapLSTM
        
        Returns:
            HandTapLSTM with pretrained weights loaded
        """
        # Create temporary UCTapLSTM to get architecture
        temp_uc = UCTapLSTM(**uc_kwargs)
        temp_sd = temp_uc.state_dict()
        
        # Clean state dict (remove 'model.' prefix if present)
        cleaned_uc_sd = {}
        for k, v in uc_state_dict.items():
            new_k = k
            if new_k.startswith("model."):
                new_k = new_k[len("model."):]
            if new_k in temp_sd and temp_sd[new_k].shape == v.shape:
                cleaned_uc_sd[new_k] = v
        
        # Create HandTapLSTM with same architecture
        model = cls(
            n_pitches=temp_uc.n_pitches - 1,  # Remove +1 that was added
            pitch_emb_dim=temp_uc.pitch_emb_dim,
            dt_dim=temp_uc.dt_dim,
            vel_dim=temp_uc.vel_dim,
            dur_dim=temp_uc.dur_dim,
            hidden=temp_uc.hidden,
            layers=temp_uc.layers,
            dropout=temp_uc.dropout,
            linear_dim=temp_uc.linear_dim,
            num_hands=num_hands,
            hand_emb_dim=hand_emb_dim,
            film_hidden_dim=film_hidden_dim,
            multi_layer_film=multi_layer_film,
            hand_dropout=hand_dropout,
        )
        
        # Load pretrained weights into uc_core
        model.uc_core.load_state_dict(cleaned_uc_sd, strict=False)
        
        return model


class HandFiLMModule(pl.LightningModule):
    """
    PyTorch Lightning module for training/testing HandTapLSTM with wandb logging.
    
    Supports:
    - Training from scratch
    - Finetuning from pretrained unconditional model
    - Test evaluation with wandb logging
    """
    
    def __init__(self, m_config, pretrained_path=None, freeze_backbone=False):
        """
        Args:
            m_config: Model configuration dict
            pretrained_path: Path to pretrained UCModule checkpoint (optional)
            freeze_backbone: Whether to freeze pretrained weights during finetuning
        """
        super().__init__()
        self.config = m_config
        self.save_hyperparameters()
        
        # Store test results for wandb
        self.test_results = []
        
        if pretrained_path is not None:
            # Load pretrained unconditional model and convert to FiLM model
            self.model = self._load_pretrained(pretrained_path, freeze_backbone)
        else:
            # Train from scratch
            self.model = HandTapLSTM(**parse_model_config(m_config, HandTapLSTM))
    
    def _load_pretrained(self, pretrained_path, freeze_backbone):
        """Load pretrained UCTapLSTM and convert to HandTapLSTM."""
        # Load checkpoint
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Get UC model kwargs from config
        uc_kwargs = parse_model_config(self.config, UCTapLSTM)
        
        # Create HandTapLSTM from pretrained
        film_model = HandTapLSTM.from_pretrained_uc(
            state_dict,
            num_hands=self.config.get('num_hands', 2),
            hand_emb_dim=self.config.get('hand_emb_dim', 32),
            film_hidden_dim=self.config.get('film_hidden_dim', 128),
            multi_layer_film=self.config.get('multi_layer_film', False),
            hand_dropout=self.config.get('hand_dropout', 0.1),
            **uc_kwargs
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for name, param in film_model.named_parameters():
                if 'film' not in name and 'hand_emb' not in name:
                    param.requires_grad = False
        
        return film_model
    
    def forward(self, x, hand_id=None, hx=None):
        """Forward pass - returns pitch logits only."""
        return self.model(x, hand_id, hx)
    
    def configure_optimizers(self):
        lr = float(self.config.get('learning_rate', 3e-4))
        
        # Use different learning rates for pretrained vs new parameters
        pretrained_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'film' in name or 'hand_emb' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        # Higher LR for new parameters, lower for pretrained
        param_groups = [
            {'params': pretrained_params, 'lr': lr * 0.1},  # Lower LR for pretrained
            {'params': new_params, 'lr': lr}  # Normal LR for new FiLM params
        ]
        
        # Filter out empty groups
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.8, 0.8),
            eps=1e-4,
            weight_decay=1e-3,
        )
        
        return [optimizer]
    
    def training_step(self, batch, batch_idx):
        features, labels = batch
        
        # Forward pass - returns pitch logits
        pitch_logits, _ = self(features)
        
        # Cross entropy loss with label smoothing (ignore pad token 88)
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
            preds = pitch_logits.argmax(dim=-1)  # (B, L)
            mask = labels != 88  # ignore pad tokens
            correct = (preds == labels) & mask
            train_acc = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
            
            # Top-5 accuracy
            top5_preds = pitch_logits.topk(5, dim=-1).indices  # (B, L, 5)
            top5_correct = (top5_preds == labels.unsqueeze(-1)).any(dim=-1) & mask
            train_top5_acc = top5_correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        # Logging
        self.log('train_loss', ce, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_acc', train_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_top5_acc', train_top5_acc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        return {'loss': ce}
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch
        
        # Forward pass - returns pitch logits
        pitch_logits, _ = self(features)
        
        # Compute metrics
        metrics = self._compute_metrics(pitch_logits, labels, features)
        
        # Logging
        logs = {
            'val_loss': metrics['loss'],
            'val_ppl': metrics['ppl'],
            'val_top5_acc': metrics['top5_acc'],
            'val_top1_acc': metrics['top1_acc'],
            'val_left_acc': metrics['left_acc'],
            'val_right_acc': metrics['right_acc'],
        }
        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return {'loss': metrics['loss'], 'logs': logs}
    
    def test_step(self, batch, batch_idx):
        """Test step with detailed metrics for wandb."""
        features, labels = batch
        
        # Forward pass - returns pitch logits
        pitch_logits, _ = self(features)
        
        # Compute metrics
        metrics = self._compute_metrics(pitch_logits, labels, features)
        
        # Store for wandb logging
        self.test_results.append(metrics)
        
        # Logging
        logs = {
            'test_loss': metrics['loss'],
            'test_ppl': metrics['ppl'],
            'test_top5_acc': metrics['top5_acc'],
            'test_top1_acc': metrics['top1_acc'],
            'test_left_acc': metrics['left_acc'],
            'test_right_acc': metrics['right_acc'],
        }
        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        return logs
    
    def on_test_epoch_end(self):
        """Aggregate test results and log to wandb."""
        if not self.test_results:
            return
        
        # Aggregate metrics
        avg_metrics = {
            'test_loss': torch.stack([r['loss'] for r in self.test_results]).mean(),
            'test_ppl': torch.stack([r['ppl'] for r in self.test_results]).mean(),
            'test_top5_acc': torch.stack([r['top5_acc'] for r in self.test_results]).mean(),
            'test_top1_acc': torch.stack([r['top1_acc'] for r in self.test_results]).mean(),
            'test_left_acc': torch.stack([r['left_acc'] for r in self.test_results]).mean(),
            'test_right_acc': torch.stack([r['right_acc'] for r in self.test_results]).mean(),
        }
        
        # Log to wandb if available
        if self.logger is not None:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        'test/loss': avg_metrics['test_loss'].item(),
                        'test/perplexity': avg_metrics['test_ppl'].item(),
                        'test/top5_accuracy': avg_metrics['test_top5_acc'].item(),
                        'test/top1_accuracy': avg_metrics['test_top1_acc'].item(),
                        'test/left_hand_accuracy': avg_metrics['test_left_acc'].item(),
                        'test/right_hand_accuracy': avg_metrics['test_right_acc'].item(),
                    })
                    
                    # Create summary table
                    wandb.run.summary['test_loss'] = avg_metrics['test_loss'].item()
                    wandb.run.summary['test_perplexity'] = avg_metrics['test_ppl'].item()
                    wandb.run.summary['test_top5_accuracy'] = avg_metrics['test_top5_acc'].item()
                    wandb.run.summary['test_top1_accuracy'] = avg_metrics['test_top1_acc'].item()
                    wandb.run.summary['test_left_hand_accuracy'] = avg_metrics['test_left_acc'].item()
                    wandb.run.summary['test_right_hand_accuracy'] = avg_metrics['test_right_acc'].item()
            except ImportError:
                pass
        
        # Print summary
        print("\n" + "=" * 60)
        print("Test Results Summary")
        print("=" * 60)
        for k, v in avg_metrics.items():
            print(f"  {k}: {v.item():.4f}")
        print("=" * 60)
        
        # Clear results
        self.test_results = []
    
    def _compute_metrics(self, pitch_logits, labels, features):
        """Compute all metrics for a batch."""
        # Mask for non-pad tokens
        pad_mask = labels != 88
        
        # Cross entropy loss
        B, L, C = pitch_logits.shape
        ce = F.cross_entropy(
            pitch_logits.reshape(B * L, C),
            labels.reshape(B * L).long(),
            reduction='mean',
            ignore_index=88
        )
        
        # Perplexity
        ppl = torch.exp(ce)
        
        # Top-5 accuracy
        top5 = torch.topk(pitch_logits, 5, dim=-1).indices
        hits5 = (top5 == labels.unsqueeze(-1)).any(dim=-1).float() * pad_mask.float()
        mask = pad_mask.bool()
        top5_acc = hits5[mask].sum() / (mask.sum() + 1e-8)
        
        # Top-1 accuracy
        pred = torch.argmax(pitch_logits, dim=-1)
        correct = (pred == labels).float()
        top1_acc = (correct * pad_mask.float()).sum() / (mask.sum() + 1e-8)
        
        # Per-hand accuracy
        hand_feature = features[..., 4].long()
        left_mask = (hand_feature == 0) & pad_mask
        right_mask = (hand_feature == 1) & pad_mask
        
        left_acc = (correct * left_mask.float()).sum() / (left_mask.sum() + 1e-8)
        right_acc = (correct * right_mask.float()).sum() / (right_mask.sum() + 1e-8)
        
        return {
            'loss': ce,
            'ppl': ppl,
            'top5_acc': top5_acc,
            'top1_acc': top1_acc,
            'left_acc': left_acc,
            'right_acc': right_acc,
        }


def configure_callbacks(save_dir, prefix='hand_film', monitor='val_loss', mode='min'):
    """Configure training callbacks."""
    name = f"{prefix}-{{epoch}}-{{val_loss:.4f}}-{{val_top5_acc:.4f}}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        filename=name,
        save_last=True,
        dirpath=save_dir
    )
    return [checkpoint_callback]


# =============================================================================
# Window Average FiLM Model
# =============================================================================

class WindowTapLSTM(nn.Module):
    """
    LSTM model with FiLM conditioning on window average pitch.
    
    Unlike HandTapLSTM which uses discrete hand labels (0/1),
    this uses continuous window_avg values (normalized pitch average).
    
    Input: (B, T, 5) - [pitch, dt, dur, vel, window_avg]
    Output: (B, T, 89) - pitch logits
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
        # Window conditioning params
        window_emb_dim=32,
        film_hidden_dim=128,
        multi_layer_film=False,
        window_dropout=0.1,
        # Normalization params (from data statistics)
        window_mean=63.0,
        window_std=5.0,
    ):
        super().__init__()
        
        # Wrap UCTapLSTM as core
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
        self.window_emb_dim = window_emb_dim
        self.film_hidden_dim = film_hidden_dim
        self.multi_layer_film = multi_layer_film
        
        # Normalization parameters
        self.register_buffer('window_mean', torch.tensor(window_mean))
        self.register_buffer('window_std', torch.tensor(window_std))
        
        # Continuous embedding: project normalized window_avg to embedding
        self.window_proj = nn.Sequential(
            nn.Linear(1, window_emb_dim),
            nn.GELU(),
            nn.Linear(window_emb_dim, window_emb_dim),
        )
        self.window_dropout = nn.Dropout(window_dropout)
        
        # FiLM conditioning MLP
        self.film_mlp = nn.Sequential(
            nn.Linear(window_emb_dim, film_hidden_dim),
            nn.GELU(),
            nn.Dropout(window_dropout),
            nn.Linear(film_hidden_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(film_hidden_dim, hidden * 2),  # gamma and beta
        )
        
        # Optional multi-layer FiLM
        if multi_layer_film:
            self.input_film_mlp = nn.Sequential(
                nn.Linear(window_emb_dim, film_hidden_dim),
                nn.GELU(),
                nn.Linear(film_hidden_dim, hidden * 2),
            )
        
        self.out_head = self.uc_core.out_head
        self._init_film_weights()
    
    def _init_film_weights(self):
        """Initialize FiLM layers to identity transformation."""
        with torch.no_grad():
            last_layer = self.film_mlp[-1]
            last_layer.weight.zero_()
            last_layer.bias.zero_()
            last_layer.bias[:self.hidden] = 1.0
            
            if self.multi_layer_film:
                last_layer = self.input_film_mlp[-1]
                last_layer.weight.zero_()
                last_layer.bias.zero_()
                last_layer.bias[:self.hidden] = 1.0
    
    def _normalize_window(self, window_avg):
        """Normalize window_avg to zero mean, unit variance."""
        return (window_avg - self.window_mean) / (self.window_std + 1e-8)
    
    def _uc_forward_features(self, x, window_emb=None, hx=None):
        """Get LSTM features from unconditional core."""
        pitch_idx = x[..., 0].long()
        dt = x[..., 1]
        dur = x[..., 2]
        vel = x[..., 3]
        
        pe = self.uc_core.pitch_emb(pitch_idx)
        feats = [pe, dt.unsqueeze(-1), dur.unsqueeze(-1), vel.unsqueeze(-1)]
        x_in = torch.cat(feats, dim=-1).to(self.uc_core.input_linear.weight.dtype)
        
        x_in = self.uc_core.input_linear(x_in)
        
        if self.multi_layer_film and window_emb is not None:
            input_film_params = self.input_film_mlp(window_emb)
            gamma_in, beta_in = torch.chunk(input_film_params, 2, dim=-1)
            x_in = gamma_in * x_in + beta_in
        
        out_uc, (h, c) = self.uc_core.lstm(x_in, hx)
        return out_uc, (h, c)
    
    def forward(self, x, window_avg=None, hx=None):
        """
        Forward pass.
        
        Args:
            x: (B, T, 5) with window_avg as 5th feature, or (B, T, 4) if window_avg provided
            window_avg: Optional (B, T) window average values
            hx: Optional LSTM hidden state
        
        Returns:
            y_pitch: (B, T, 89) - pitch logits
            (h, c): LSTM hidden state
        """
        if x.size(-1) == 5 and window_avg is None:
            x_uc = x[..., :4]
            window_avg = x[..., 4]
        else:
            x_uc = x[..., :4]
            assert window_avg is not None
        
        # Normalize and embed window_avg
        window_norm = self._normalize_window(window_avg)
        window_emb = self.window_proj(window_norm.unsqueeze(-1))
        window_emb = self.window_dropout(window_emb)
        
        # Get LSTM features
        out_uc, (h, c) = self._uc_forward_features(x_uc, window_emb=window_emb, hx=hx)
        
        # FiLM modulation
        film_params = self.film_mlp(window_emb)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        out_cond = gamma * out_uc + beta
        
        y_pitch = self.out_head(out_cond)
        return y_pitch, (h, c)
    
    @classmethod
    def from_pretrained_uc(
        cls,
        uc_state_dict,
        window_emb_dim=32,
        film_hidden_dim=128,
        multi_layer_film=False,
        window_dropout=0.1,
        window_mean=63.0,
        window_std=5.0,
        **uc_kwargs,
    ):
        """Create WindowTapLSTM from pretrained UCTapLSTM state dict."""
        temp_uc = UCTapLSTM(**uc_kwargs)
        temp_sd = temp_uc.state_dict()
        
        cleaned_uc_sd = {}
        for k, v in uc_state_dict.items():
            new_k = k
            if new_k.startswith("model."):
                new_k = new_k[len("model."):]
            if new_k in temp_sd and temp_sd[new_k].shape == v.shape:
                cleaned_uc_sd[new_k] = v
        
        model = cls(
            n_pitches=temp_uc.n_pitches - 1,
            pitch_emb_dim=temp_uc.pitch_emb_dim,
            dt_dim=temp_uc.dt_dim,
            vel_dim=temp_uc.vel_dim,
            dur_dim=temp_uc.dur_dim,
            hidden=temp_uc.hidden,
            layers=temp_uc.layers,
            dropout=temp_uc.dropout,
            linear_dim=temp_uc.linear_dim,
            window_emb_dim=window_emb_dim,
            film_hidden_dim=film_hidden_dim,
            multi_layer_film=multi_layer_film,
            window_dropout=window_dropout,
            window_mean=window_mean,
            window_std=window_std,
        )
        
        model.uc_core.load_state_dict(cleaned_uc_sd, strict=False)
        return model


class WindowFiLMModule(pl.LightningModule):
    """
    PyTorch Lightning module for training/testing WindowTapLSTM.
    """
    
    def __init__(
        self,
        m_config,
        pretrained_path=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.config = m_config
        self.save_hyperparameters()
        
        if pretrained_path:
            self.model = self._load_pretrained(pretrained_path, freeze_backbone)
        else:
            self.model = WindowTapLSTM(**parse_model_config(m_config, WindowTapLSTM))
        
        self.test_results = []
    
    def _load_pretrained(self, pretrained_path, freeze_backbone):
        """Load pretrained UCTapLSTM and convert to WindowTapLSTM."""
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        uc_kwargs = parse_model_config(self.config, UCTapLSTM)
        
        film_model = WindowTapLSTM.from_pretrained_uc(
            state_dict,
            window_emb_dim=self.config.get('window_emb_dim', 32),
            film_hidden_dim=self.config.get('film_hidden_dim', 128),
            multi_layer_film=self.config.get('multi_layer_film', False),
            window_dropout=self.config.get('window_dropout', 0.1),
            window_mean=self.config.get('window_mean', 63.0),
            window_std=self.config.get('window_std', 5.0),
            **uc_kwargs
        )
        
        if freeze_backbone:
            for name, param in film_model.named_parameters():
                if 'film' not in name and 'window' not in name:
                    param.requires_grad = False
        
        return film_model
    
    def forward(self, x, window_avg=None, hx=None):
        return self.model(x, window_avg, hx)
    
    def configure_optimizers(self):
        lr = float(self.config.get('learning_rate', 3e-4))
        
        pretrained_params = []
        new_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'film' in name or 'window' in name:
                    new_params.append(param)
                else:
                    pretrained_params.append(param)
        
        param_groups = [
            {'params': pretrained_params, 'lr': lr * 0.1},
            {'params': new_params, 'lr': lr}
        ]
        param_groups = [g for g in param_groups if len(g['params']) > 0]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.8, 0.8),
            eps=1e-4,
            weight_decay=1e-3,
        )
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
            preds = pitch_logits.argmax(dim=-1)  # (B, L)
            mask = labels != 88  # ignore pad tokens
            correct = (preds == labels) & mask
            train_acc = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
            
            # Top-5 accuracy
            top5_preds = pitch_logits.topk(5, dim=-1).indices  # (B, L, 5)
            top5_correct = (top5_preds == labels.unsqueeze(-1)).any(dim=-1) & mask
            train_top5_acc = top5_correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
        
        self.log('train_loss', ce, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_acc', train_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('train_top5_acc', train_top5_acc, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        return {'loss': ce}
    
    def validation_step(self, batch, batch_idx):
        features, labels = batch
        pitch_logits, _ = self(features)
        metrics = self._compute_metrics(pitch_logits, labels, features)
        
        logs = {
            'val_loss': metrics['loss'],
            'val_ppl': metrics['ppl'],
            'val_top5_acc': metrics['top5_acc'],
            'val_top1_acc': metrics['top1_acc'],
        }
        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return {'loss': metrics['loss'], 'logs': logs}
    
    def test_step(self, batch, batch_idx):
        features, labels = batch
        pitch_logits, _ = self(features)
        metrics = self._compute_metrics(pitch_logits, labels, features)
        
        self.test_results.append(metrics)
        
        logs = {
            'test_loss': metrics['loss'],
            'test_ppl': metrics['ppl'],
            'test_top5_acc': metrics['top5_acc'],
            'test_top1_acc': metrics['top1_acc'],
        }
        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return logs
    
    def on_test_epoch_end(self):
        if not self.test_results:
            return
        
        avg_metrics = {
            'test_loss': torch.stack([r['loss'] for r in self.test_results]).mean(),
            'test_ppl': torch.stack([r['ppl'] for r in self.test_results]).mean(),
            'test_top5_acc': torch.stack([r['top5_acc'] for r in self.test_results]).mean(),
            'test_top1_acc': torch.stack([r['top1_acc'] for r in self.test_results]).mean(),
        }
        
        print("\n" + "=" * 60)
        print("Window FiLM Test Results Summary")
        print("=" * 60)
        for k, v in avg_metrics.items():
            print(f"  {k}: {v.item():.4f}")
        print("=" * 60)
        
        self.test_results = []
    
    def _compute_metrics(self, pitch_logits, labels, features):
        pad_mask = labels != 88
        
        B, L, C = pitch_logits.shape
        ce = F.cross_entropy(
            pitch_logits.reshape(B * L, C),
            labels.reshape(B * L).long(),
            reduction='mean',
            ignore_index=88
        )
        
        ppl = torch.exp(ce)
        
        top5 = torch.topk(pitch_logits, 5, dim=-1).indices
        hits5 = (top5 == labels.unsqueeze(-1)).any(dim=-1).float() * pad_mask.float()
        mask = pad_mask.bool()
        top5_acc = hits5[mask].sum() / (mask.sum() + 1e-8)
        
        pred = torch.argmax(pitch_logits, dim=-1)
        correct = (pred == labels).float()
        top1_acc = (correct * pad_mask.float()).sum() / (mask.sum() + 1e-8)
        
        return {
            'loss': ce,
            'ppl': ppl,
            'top5_acc': top5_acc,
            'top1_acc': top1_acc,
        }


def train_with_wandb(
    data_dir,
    pretrained_path,
    output_dir,
    project_name='tap-to-music-hand',
    run_name=None,
    max_epochs=100,
    batch_size=32,
    freeze_backbone=False,
):
    """
    Train HandFiLMModule with wandb logging.
    
    Args:
        data_dir: Path to data directory with pop909hand-*.pkl files
        pretrained_path: Path to pretrained UCModule checkpoint
        output_dir: Output directory for checkpoints
        project_name: Wandb project name
        run_name: Wandb run name (auto-generated if None)
        max_epochs: Maximum training epochs
        batch_size: Batch size
        freeze_backbone: Whether to freeze pretrained backbone
    
    Returns:
        Trained model and trainer
    """
    import wandb
    from pytorch_lightning.loggers import WandbLogger
    from yzq.data_module import HandDataModule
    
    # Initialize wandb
    wandb_logger = WandbLogger(
        project=project_name,
        name=run_name,
        save_dir=output_dir,
    )
    
    # Config
    m_config = {
        **model_config.get('unconditional', {}),
        'learning_rate': 3e-4,
        'num_hands': 2,
        'hand_emb_dim': 16,
        'film_hidden_dim': 64,
    }
    
    # Data module
    dm = HandDataModule(
        data_dir=data_dir,
        feature_type='pop909hand',
        batch_size=batch_size,
        num_workers=4
    )
    
    # Model
    module = HandFiLMModule(
        m_config=m_config,
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone
    )
    
    # Callbacks
    callbacks = configure_callbacks(output_dir, prefix='hand_film')
    
    # Trainer
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        accelerator='auto',
    )
    
    # Train
    trainer.fit(module, dm)
    
    # Test
    trainer.test(module, dm)
    
    # Finish wandb
    wandb.finish()
    
    return module, trainer


def main():
    """Test the FiLM model."""
    print("Testing HandTapLSTM...")
    
    # Create model
    model = HandTapLSTM(
        n_pitches=88,
        pitch_emb_dim=32,
        hidden=128,
        layers=2,
        dropout=0.15,
        linear_dim=128,
        num_hands=2,
        hand_emb_dim=16,
        film_hidden_dim=64,
    )
    
    # Test forward pass
    B, T = 4, 64
    x = torch.zeros(B, T, 5)
    x[..., 0] = torch.randint(0, 88, (B, T))  # pitch
    x[..., 1] = torch.rand(B, T)  # dt
    x[..., 2] = torch.rand(B, T)  # dur
    x[..., 3] = torch.rand(B, T) * 127  # vel
    x[..., 4] = torch.randint(0, 2, (B, T))  # hand
    
    y, (h, c) = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape (pitch logits): {y.shape}")
    print(f"Hidden state shape: h={h.shape}, c={c.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    film_params = sum(p.numel() for n, p in model.named_parameters() if 'film' in n or 'hand_emb' in n)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"FiLM parameters: {film_params:,}")
    print(f"Backbone parameters: {total_params - film_params:,}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
