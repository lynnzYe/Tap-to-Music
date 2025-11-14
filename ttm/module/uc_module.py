"""
Author: Lynn Ye
Created on: 2025/11/13
Brief: 
"""
import inspect

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ttm.config import model_config
from ttm.data_preparation.data_module import UCDataModule
from ttm.model.uc_model import UCTapLSTM


def parse_model_config(config_dict, model_type):
    sig = inspect.signature(model_type.__init__)
    allowed_keys = [p for p in sig.parameters if p != "self"]
    filtered_config = {k: v for k, v in config_dict.items() if k in allowed_keys}
    return filtered_config


def configure_optimizers(module, lr=1e-3, step_size=50):
    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=lr,
        betas=(0.8, 0.8),
        eps=1e-4,
        weight_decay=1e-3,
    )
    # scheduler_lrdecay = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=step_size,
    #     gamma=0.1
    # )
    return [optimizer]


def configure_callbacks(*, save_dir, prefix='', monitor='val_f1', mode='max'):
    name = prefix + '-{epoch}-{val_loss:.4f}-{val_f1:.4f}'.replace('val_f1', monitor)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=3,
        filename=name,
        save_last=True,
        dirpath=save_dir
    )
    # earlystop_callback = pl.callbacks.EarlyStopping(
    #     monitor=monitor,
    #     patience=200,
    #     mode=mode,
    # )
    return [checkpoint_callback]


class UCModule(pl.LightningModule):
    def __init__(self, m_config):
        super().__init__()
        self.config = m_config
        self.model = UCTapLSTM(**parse_model_config(m_config, UCTapLSTM))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self, lr=float(self.config.get('learning_rate', 3e-4)))

    def training_step(self, batch, batch_idx):
        # Data
        features, labels = batch

        # Forward pass
        pitch_logits, _ = self(features)

        # Cross entropy for 88 + 1(pad) classes
        B, L, C = pitch_logits.shape
        ce = F.cross_entropy(pitch_logits.reshape(B * L, C), labels.reshape(B * L).long(), reduction='mean',
                             ignore_index=88)

        # Logging
        logs = {'train_loss': ce}
        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return {'loss': ce, 'logs': logs}

    def validation_step(self, batch, batch_idx):
        # Data
        features, labels = batch

        # Forward pass
        pitch_logits, _ = self(features)

        # Cross entropy for 88 + 1(pad) classes
        pad_mask = labels != 88

        B, L, C = pitch_logits.shape
        ce = F.cross_entropy(pitch_logits.reshape(B * L, C), labels.reshape(B * L).long(), reduction='mean',
                             ignore_index=88)
        ppl = torch.exp(ce)  # expect initial perplexity = |V|, 88+1

        top5 = torch.topk(pitch_logits, 5, dim=-1).indices
        hits = (top5 == labels.unsqueeze(-1)).any(dim=-1).float() * pad_mask.float()
        mask = pad_mask.bool()
        hits_masked = hits[mask]
        top5_acc = hits_masked.sum() / (mask.sum() + 1e-8)  # expect 0.056 for uniform distribution

        # Logging
        logs = {'val_loss': ce,
                'ppl': ppl,
                'val_top5_acc': top5_acc
                }
        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return {'loss': ce, 'logs': logs}


def check_training_step():
    trainer = pl.Trainer(
        default_root_dir='/Users/kurono/Desktop/10701 final/tap_the_music/output/debug',
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=1,
        accelerator='cpu'
    )
    model = UCModule(model_config['unconditional'])
    data = UCDataModule('/Users/kurono/Desktop/10701 final/tap_the_music/output')
    trainer.fit(model, data)


def main():
    check_training_step()
    print("Hello, world!")


if __name__ == "__main__":
    main()
