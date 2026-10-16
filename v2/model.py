"""KG-GT v2: kinematic stream (output K), EEG stream (output E), gated fusion (output F).

Structure rules, each pinned by a test in tests/test_v2_model.py:
- K's parameters are trained only by K's loss: the fusion reads K and the kinematic tokens
  through .detach(), and E never touches the kinematic stream.
- Information moves between positions only through v2.attention modules (all recorded, all listed
  by attention_layers()) and the 5-tap stem convolution (fixed footprint: KinStream.stem_matrix).
- The outputs split exactly: K = K_base * gain and F = relu(K + F_eeg).
- With cfg["model"]["causal"] = True every output at token t depends on inputs at tokens <= t only
  (masked attention over time, a left-padded stem, a masked fusion); tests/test_v2_causal.py checks it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, AttentionPool
from .blocks import GraphEncoder, MuscleHead, TemporalEncoder, causal_mask
from .electrodes import electrode_prior as _electrode_prior


class KinStream(nn.Module):
    """[kinematics | history] -> stem -> attention over time -> muscle graph -> K_base, times the history gain."""

    def __init__(self, c, n_kin, n_hist, n_muscles, muscle_prior):
        super().__init__()
        d = c["d_model"]
        self.causal = bool(c.get("causal", False))
        # causal: the 5 taps cover the token and the 4 before it (left padding only), else 2 before and 2 after
        self.stem = nn.Conv1d(n_kin + n_hist, d, kernel_size=5, padding=0 if self.causal else 2)
        self.enc = TemporalEncoder(d, c["kin_layers"], c["n_heads"], c["ffn_dim"], c["dropout"], causal=self.causal)
        self.head = MuscleHead(d, n_muscles, c["node_dim"], c["muscle_gat_layers"], c["n_heads"], muscle_prior,
                               c["muscle_dropout"])
        self.gain = nn.Sequential(nn.Linear(n_hist, 16), nn.GELU(), nn.Linear(16, n_muscles))
        nn.init.zeros_(self.gain[-1].weight)
        nn.init.zeros_(self.gain[-1].bias)

    def forward(self, kin, hist):
        x = torch.cat([kin, hist], dim=-1).transpose(1, 2)
        if self.causal:
            x = F.pad(x, (self.stem.kernel_size[0] - 1, 0))
        tokens = self.enc(self.stem(x).transpose(1, 2))
        base = self.head(tokens)
        gain = torch.pow(2.0, torch.tanh(self.gain(hist)))            # bounded in [0.5, 2], exactly 1 at init
        return base * gain, tokens, base, gain

    def stem_matrix(self, t: int) -> torch.Tensor:
        """(t, t), rows sum to 1: share of each input token in each stem token (kernel energy per offset)."""
        w = self.stem.weight.detach().float().cpu().pow(2).sum(dim=(0, 1)).sqrt()      # (kernel,)
        half = w.numel() - 1 if self.causal else w.numel() // 2      # tap of the token itself
        m = torch.zeros(t, t)
        for o in range(w.numel()):
            if abs(o - half) < t:
                m += torch.diag(w[o].expand(t - abs(o - half)), o - half)
        return m / m.sum(-1, keepdim=True)


class EEGStream(nn.Module):
    """Band power per electrode -> electrode graph -> pooling over electrodes -> attention over time -> muscle graph -> E."""

    def __init__(self, c, n_electrodes, n_bands, n_muscles, muscle_prior, elec_prior):
        super().__init__()
        d, de = c["d_model"], c["electrode_dim"]
        self.p_drop = float(c["electrode_dropout"])
        self.embed = nn.Sequential(nn.Linear(n_bands, de), nn.GELU(), nn.Linear(de, de))
        self.identity = nn.Parameter(torch.randn(n_electrodes, de) * 0.02)
        self.graph = GraphEncoder(n_electrodes, de, c["electrode_gat_layers"], c["n_heads"], elec_prior, c["dropout"])
        self.pool = AttentionPool(de)
        self.up = nn.Linear(de, d)
        self.enc = TemporalEncoder(d, c["eeg_layers"], c["n_heads"], c["ffn_dim"], c["dropout"],
                                   causal=bool(c.get("causal", False)))
        self.head = MuscleHead(d, n_muscles, c["node_dim"], c["muscle_gat_layers"], c["n_heads"], muscle_prior,
                               c["muscle_dropout"])

    def forward(self, eeg):
        if self.training and self.p_drop > 0:
            keep = (torch.rand(eeg.shape[0], 1, eeg.shape[2], 1, device=eeg.device) >= self.p_drop).to(eeg.dtype)
            eeg = eeg * keep        # no 1 / (1 - p): electrodes are embedded one by one and only averaged, never summed
        nodes = self.graph(self.embed(eeg) + self.identity.to(eeg.dtype))
        tokens = self.enc(self.up(self.pool(nodes)))
        return self.head(tokens), tokens


class Fusion(nn.Module):
    """F = relu(K + gate * correction); kinematic tokens (detached) query the EEG tokens for the correction."""

    def __init__(self, c, n_muscles):
        super().__init__()
        self.causal = bool(c.get("causal", False))
        self.cross = Attention(c["d_model"], c["n_heads"], c["dropout"])
        self.out = nn.Linear(c["d_model"], n_muscles)
        self.gate = nn.Parameter(torch.zeros(n_muscles))

    def forward(self, kin_tokens, eeg_tokens, k_out):
        mask = causal_mask(kin_tokens.shape[1], kin_tokens.device) if self.causal else None
        eeg_part = self.gate * self.out(self.cross(kin_tokens.detach(), eeg_tokens, bias=mask))
        return F.relu(k_out.detach() + eeg_part), eeg_part


class KGGTv2(nn.Module):
    def __init__(self, cfg_model, n_kin=24, n_hist=8, n_electrodes=32, n_bands=4, n_muscles=5,
                 muscle_prior=None, electrode_prior=None):
        super().__init__()
        self.causal = bool(cfg_model.get("causal", False))       # every output at token t uses inputs up to t only
        self.kin = KinStream(cfg_model, n_kin, n_hist, n_muscles, muscle_prior)
        self.eeg = EEGStream(cfg_model, n_electrodes, n_bands, n_muscles, muscle_prior, electrode_prior)
        self.fusion = Fusion(cfg_model, n_muscles)

    def forward(self, kin, hist, eeg):
        k_out, kin_tokens, k_base, gain = self.kin(kin, hist)
        e_out, eeg_tokens = self.eeg(eeg)
        f_out, f_eeg = self.fusion(kin_tokens, eeg_tokens, k_out)
        return {"K": k_out, "E": e_out, "F": f_out, "K_base": k_base, "gain": gain, "F_eeg": f_eeg,
                "gate": self.fusion.gate.detach().clone()}

    def attention_layers(self) -> dict:
        """Names of every recorded attention module, by stream and stage, in forward order."""
        def stack(encoder, prefix):
            return [f"{prefix}.layers.{i}.attn" for i in range(len(encoder.layers))]
        return {
            "kin": {"temporal": stack(self.kin.enc, "kin.enc"), "muscle": stack(self.kin.head.graph, "kin.head.graph")},
            "eeg": {"electrode": stack(self.eeg.graph, "eeg.graph"), "pool": "eeg.pool",
                    "temporal": stack(self.eeg.enc, "eeg.enc"), "muscle": stack(self.eeg.head.graph, "eeg.head.graph")},
            "fusion": {"cross": "fusion.cross"},
        }


def build_model(cfg, muscle_prior=None) -> KGGTv2:
    mp = None if muscle_prior is None else torch.as_tensor(muscle_prior, dtype=torch.float32)
    return KGGTv2(cfg["model"], n_bands=len(cfg["eeg"]["bands"]), muscle_prior=mp,
                  electrode_prior=torch.from_numpy(_electrode_prior()))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stream_of(param_name: str) -> str:
    return "kin" if param_name.startswith("kin.") else "eeg"
