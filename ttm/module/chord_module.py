import inspect

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from ttm.config import model_config
from ttm.data_preparation.data_module import UCDataModule
from ttm.model.uc_model import UCTapLSTM
from ttm.model.chord_model import ChordTapLSTM


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
    return [optimizer]

class ChordModule(pl.LightningModule):
    def __init__(self, m_config, uc_state_dict=None, freeze_uc=False):

        super().__init__()
        self.config = m_config

        if uc_state_dict is not None:
            uc_params = parse_model_config(m_config, UCTapLSTM)
            self.model = ChordTapLSTM.from_pretrained_uc(
                uc_state_dict=uc_state_dict,
                **uc_params
            )
        else:
            self.model = ChordTapLSTM(**parse_model_config(m_config, ChordTapLSTM))

        if freeze_uc:
            for name, p in self.model.uc_core.named_parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return configure_optimizers(self, lr=float(self.config.get('learning_rate', 3e-4)))

    def training_step(self, batch, batch_idx):
        features, labels = batch  
        pitch_logits, _ = self(features)

        B, L, C = pitch_logits.shape
        ce = F.cross_entropy(
            pitch_logits.reshape(B * L, C),
            labels.reshape(B * L).long(),
            reduction='mean',
            ignore_index=88
        )

        logs = {'train_loss': ce}
        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return {'loss': ce, 'logs': logs}

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        pitch_logits, _ = self(features)

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
        hits = (top5 == labels.unsqueeze(-1)).any(dim=-1).float() * pad_mask.float()
        mask = pad_mask.bool()
        hits_masked = hits[mask]
        top5_acc = hits_masked.sum() / (mask.sum() + 1e-8)

        logs = {
            'val_loss': ce,
            'ppl': ppl,
            'val_top5_acc': top5_acc
        }
        self.log_dict(logs, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        return {'loss': ce, 'logs': logs}


def check_chord_training_step():
    trainer = pl.Trainer(
        default_root_dir='data/chord_debug',
        log_every_n_steps=50,
        check_val_every_n_epoch=1,
        reload_dataloaders_every_n_epochs=1,
        accelerator='cpu'
    )

    chord_cfg = model_config['chord']
    # model = ChordModule(chord_cfg)

    ckpt = torch.load('/path/to/uc/last.ckpt', map_location='cpu')
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    model = ChordModule(chord_cfg, uc_state_dict=sd, freeze_uc=True)

    data = UCDataModule(
        'data/chord',
        feature='chord'
    )
    trainer.fit(model, data)


def main():
    check_chord_training_step()
    print("Hello, world!")


if __name__ == "__main__":
    main()
