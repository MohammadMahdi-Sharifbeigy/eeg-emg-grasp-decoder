"""EEG-only student of the guided model: band power -> E. No kinematic input exists anywhere in this module.

The student is the v2 EEGStream, kept under the attribute name `eeg`, so its state keys are v2's `eeg.*`.
`forward` repeats EEGStream.forward step by step only to keep the output of the first temporal layer.
"""
import torch
import torch.nn as nn

from v2.electrodes import channel_names
from v2.electrodes import electrode_prior as _electrode_prior
from v2.model import EEGStream


class EEGOnlyModel(nn.Module):
    def __init__(self, cfg_model, n_electrodes=32, n_bands=4, n_muscles=5, muscle_prior=None, electrode_prior=None):
        super().__init__()
        self.causal = bool(cfg_model.get("causal", False))
        self.eeg = EEGStream(cfg_model, n_electrodes, n_bands, n_muscles, muscle_prior, electrode_prior)

    def forward(self, eeg):
        s = self.eeg
        if self.training and s.p_drop > 0:
            keep = (torch.rand(eeg.shape[0], 1, eeg.shape[2], 1, device=eeg.device) >= s.p_drop).to(eeg.dtype)
            eeg = eeg * keep                                 # whole electrodes, no rescale (as in v2)
        nodes = s.graph(s.embed(eeg) + s.identity.to(eeg.dtype))
        x = s.up(s.pool(nodes))
        enc = s.enc
        x = x + enc._positions(x.shape[1], x.device, x.dtype)
        mask = None
        if enc.causal:
            from v2.blocks import causal_mask
            mask = causal_mask(x.shape[1], x.device)
        mid = None
        for i, layer in enumerate(enc.layers):
            x = layer(x, mask)
            if i == 0:
                mid = x
        fin = enc.norm(x)
        return {"E": s.head(fin), "mid": mid, "fin": fin}

    def attention_layers(self) -> dict:
        def stack(encoder, prefix):
            return [f"{prefix}.layers.{i}.attn" for i in range(len(encoder.layers))]
        return {"electrode": stack(self.eeg.graph, "eeg.graph"), "pool": "eeg.pool",
                "temporal": stack(self.eeg.enc, "eeg.enc"), "muscle": stack(self.eeg.head.graph, "eeg.head.graph")}


def build_student(cfg, muscle_prior=None) -> EEGOnlyModel:
    mp = None if muscle_prior is None else torch.as_tensor(muscle_prior, dtype=torch.float32)
    names = channel_names(cfg)
    return EEGOnlyModel(cfg["model"], n_electrodes=len(names), n_bands=len(cfg["eeg"]["bands"]), muscle_prior=mp,
                        electrode_prior=torch.from_numpy(_electrode_prior(names)))
