import torch
import torch.nn as nn

from ttm.model.uc_model import UCTapLSTM


class ChordTapLSTM(nn.Module):
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
        # chord conditioning params
        num_chords=109,         # 108 chords + 1 "no chord"
        chord_emb_dim=32,
        film_hidden_dim=64,
        use_next_chord=False,
    ):
        super().__init__()

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
        self.use_next_chord = use_next_chord

        # Chord embedding
        self.chord_emb = nn.Embedding(num_chords, chord_emb_dim)
        if use_next_chord:
            self.next_chord_emb = nn.Embedding(num_chords, chord_emb_dim)
            chord_in_dim = chord_emb_dim * 2
        else:
            self.next_chord_emb = None
            chord_in_dim = chord_emb_dim

        # FiLM-like conditioning: map chord embedding(s) → (gamma, beta)
        self.film_mlp = nn.Sequential(
            nn.Linear(chord_in_dim, film_hidden_dim),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, hidden * 2),
        )

        self.out_head = self.uc_core.out_head

    def _uc_forward_features(self, x, hx=None):
        pitch_idx = x[..., 0].long()
        dt = x[..., 1]
        dur = x[..., 2]
        vel = x[..., 3]

        pe = self.uc_core.pitch_emb(pitch_idx)
        feats = [pe, dt.unsqueeze(-1), dur.unsqueeze(-1), vel.unsqueeze(-1)]
        x_in = torch.cat(feats, dim=-1).to(self.uc_core.input_linear.weight.dtype)

        x_in = self.uc_core.input_linear(x_in)  # (B, T, hidden)
        out_uc, (h, c) = self.uc_core.lstm(x_in, hx)  # out_uc: (B, T, hidden)

        return out_uc, (h, c)

    def forward(self, x, chord_id=None, next_chord_id=None, hx=None):
    
        if x.size(-1) == 5 and chord_id is None:
            x_uc = x[..., :4]
            chord_id = x[..., 4].long()
        else:
            x_uc = x[..., :4]
            assert chord_id is not None, "chord_id must be provided if x does not include it as 5th feature."
            chord_id = chord_id.long()

        out_uc, (h, c) = self._uc_forward_features(x_uc, hx=hx) 

        chord_emb = self.chord_emb(chord_id) 
        chord_features = chord_emb

        # if self.use_next_chord:
        #     assert next_chord_id is not None, "next_chord_id required when use_next_chord=True"
        #     next_chord_id = next_chord_id.long()
        #     next_chord_emb = self.next_chord_emb(next_chord_id)
        #     chord_features = torch.cat([chord_emb, next_chord_emb], dim=-1)  # (B, T, 2 * chord_emb_dim)

        film_params = self.film_mlp(chord_features)             
        gamma, beta = torch.chunk(film_params, 2, dim=-1)       

        out_cond = gamma * out_uc + beta              

        y_pitch = self.out_head(out_cond)                   
        return y_pitch, (h, c)

    @classmethod
    def from_pretrained_uc(
        cls,
        uc_state_dict,
        num_chords=109,
        chord_emb_dim=32,
        film_hidden_dim=64,
        use_next_chord=False,
        **uc_kwargs,
    ):
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
            num_chords=num_chords,
            chord_emb_dim=chord_emb_dim,
            film_hidden_dim=film_hidden_dim,
            use_next_chord=use_next_chord,
        )

        model.uc_core.load_state_dict(cleaned_uc_sd, strict=False)

        return model
