"""
Author: Lynn Ye
Created on: 2025/11/13
Brief: 
"""
import torch
import torch.nn as nn


class UCTapLSTM(nn.Module):
    def __init__(self, n_pitches=88, pitch_emb_dim=32, dt_dim=1, vel_dim=1, dur_dim=1, hidden=128, layers=2, dropout=0.15,
                 linear_dim=128):
        super().__init__()
        self.n_pitches = n_pitches + 1  # add one pad
        self.pitch_emb_dim = pitch_emb_dim
        self.dt_dim = dt_dim
        self.vel_dim = vel_dim
        self.dur_dim = dur_dim
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        self.linear_dim = linear_dim
        self.pitch_emb = nn.Embedding(self.n_pitches, self.pitch_emb_dim)  # +1 for pad
        self.input_linear = nn.Linear(self.pitch_emb_dim + self.dt_dim + self.vel_dim + self.dur_dim, self.hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.out_head = nn.Sequential(
            nn.Linear(hidden, linear_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(linear_dim, 88 + 1),  # 88 piano pitches + pad token
        )

    def forward(self, x, hx=None):
        # pitch_idx: (B, T) int, dt, vel: (B, T) float
        pitch_idx = x[..., 0].long()
        dt = x[..., 1]
        dur = x[..., 2]
        vel = x[..., 3]
        pe = self.pitch_emb(pitch_idx)
        feats = [pe, dt.unsqueeze(-1), dur.unsqueeze(-1), vel.unsqueeze(-1)]

        x = torch.cat(feats, dim=-1).to(self.input_linear.weight.dtype)
        x = self.input_linear(x)  # (B,T,hidden)

        out, (h, c) = self.lstm(x, hx)  # out: (B,T,hidden)

        y_pitch = self.out_head(out)  # (B,T,88)
        y_pitch = y_pitch.squeeze(-1)
        return y_pitch, (h, c)


def main():
    pass


if __name__ == "__main__":
    main()
