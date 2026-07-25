import math
import torch
import torch.nn.functional as F
import torch.nn as nn


# ============================================================================
# Positional Encodings
# ============================================================================

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_emb[:, :x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sine/cosine positional encoding (Vaswani et al. 2017).
    Naturally encodes distance, strongly encourages diagonal attention.
    """
    def __init__(self, d_model, max_len=10000, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


def _build_pos_enc(pos_type: str, d_model: int, max_len: int = 10000):
    """Factory that builds a positional encoding module from a string key.

    Config options (set under ``model.pos_encoding``):
        ``"learnable"``  – trainable embedding (default, original behaviour)
        ``"sinusoidal"`` – fixed sine/cosine (recommended for diagonal attn)
        ``"none"``       – no positional encoding
    """
    pos_type = (pos_type or "learnable").lower()
    if pos_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len)
    if pos_type == "none":
        return nn.Identity()
    # default: learnable
    return LearnablePositionalEncoding(d_model, max_len)


def _make_local_mask(T: int, window: int, device) -> torch.Tensor:
    """Return an additive attention mask (float) of shape (T, T).

    Positions outside ``[-window, +window]`` are set to ``-inf`` so that
    softmax drives their weight to 0, creating a banded diagonal pattern.
    """
    idx = torch.arange(T, device=device)
    dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()   # (T, T)
    mask = torch.zeros(T, T, device=device)
    mask[dist > window] = float("-inf")
    return mask


# ============================================================================
# CNN Encoder
# ============================================================================

class TemporalCNNEncoder(nn.Module):
    """FIX: 2-layer residual CNN (matches teammate nb04 depth).
    conv(k=7)+LN+GELU -> conv(k=5)+LN + shortcut(k=1) -> GELU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1    = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.ln1      = nn.LayerNorm(out_channels)
        self.conv2    = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.ln2      = nn.LayerNorm(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.shortcut.weight)

    def forward(self, x):
        # x: (B, T, C_in)
        res = self.shortcut(x.transpose(1, 2)).transpose(1, 2)          # (B,T,out)
        h   = F.gelu(self.ln1(self.conv1(x.transpose(1, 2)).transpose(1, 2)))
        h   = self.ln2(self.conv2(h.transpose(1, 2)).transpose(1, 2))
        return F.gelu(h + res)


# ============================================================================
# Fusion Gate
# ============================================================================

class LearnableGatedFusion(nn.Module):
    """Sigmoid gate between streams a and b.
    g_t * a + (1-g_t) * b
    """
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, d_model)
        nn.init.zeros_(self.gate.bias)

    def forward(self, a, b):
        g_t   = torch.sigmoid(self.gate(torch.cat([a, b], dim=-1)))
        fused = g_t * a + (1 - g_t) * b
        return fused, g_t


# ============================================================================
# Transformer Encoder
# ============================================================================

class CustomTransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer."""
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1   = nn.Linear(d_model, ffn_dim)
        self.linear2   = nn.Linear(ffn_dim, d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.drop1     = nn.Dropout(dropout)
        self.drop2     = nn.Dropout(dropout)
        self.act       = nn.GELU()

    def forward(self, src, return_attention=False, attn_mask=None):
        n = self.norm1(src)
        s2, attn = self.self_attn(n, n, n,
                                  attn_mask=attn_mask,
                                  need_weights=return_attention,
                                  average_attn_weights=False)
        src = src + self.drop1(s2)
        n2  = self.norm2(src)
        src = src + self.drop2(self.linear2(self.drop1(self.act(self.linear1(n2)))))
        return src, attn


class OptimizedTransformerEncoder(nn.Module):
    def __init__(self, d_model=64, n_heads=8, n_layers=4, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, return_attention=False, attn_mask=None):
        attn_maps = []
        for layer in self.layers:
            x, attn = layer(x, return_attention=return_attention, attn_mask=attn_mask)
            if return_attention:
                attn_maps.append(attn)
        return x, attn_maps


# ============================================================================
# Graph Attention
# ============================================================================

class OptimizedKG_GAT(nn.Module):
    def __init__(self, node_dim=64, kin_dim=13, out_nodes=5):
        super().__init__()
        self.out_nodes = out_nodes
        self.adj_mlp   = nn.Sequential(
            nn.Linear(kin_dim, 32), nn.GELU(),
            nn.Linear(32, out_nodes * out_nodes)
        )
        self.node_proj = nn.Linear(node_dim, node_dim)

    def forward(self, node_features, kin_features):
        B, T, N, D = node_features.shape
        adj   = F.softmax(self.adj_mlp(kin_features).view(B * T, N, N), dim=-1)
        nodes = self.node_proj(node_features).view(B * T, N, D)
        return torch.bmm(adj, nodes).view(B, T, N, D)


# ============================================================================
# Full Hybrid Model
# ============================================================================

class HybridKGGTModel(nn.Module):
    """KG-GT model - nb06 fixed version.

    Key fixes:
    1. TemporalCNNEncoder: 2-layer residual (k=7+k=5+shortcut)
    2. Gate: gates between eeg_kin_feat and kin_eeg_feat DIRECTLY
       - not their sum, so each stream gets clean gradient signal
       - gate semantics: g~1 = rely on EEG, g~0 = rely on KIN
    3. Always returns dict {prediction, gate, fused} for gate loss + plots

    Config keys (all under ``cfg["model"]``):
        ``pos_encoding``     : "learnable" | "sinusoidal" | "none"
                               Controls positional encoding type for EEG & KIN
                               streams. "sinusoidal" strongly promotes diagonal
                               attention. (default: "learnable")
        ``local_attn_window``: int | null
                               If set to an integer (e.g. 50), self-attention
                               inside the Transformer is masked to only attend
                               within ±window timesteps, enforcing locality.
                               Set to null / omit to use full attention.
                               (default: null)
        ``chunk_size``       : int | null
                               Sequence is split into chunks before the
                               attention layers to cut O(T^2) cost.
                               (default: 500)
    """
    def __init__(self, cfg, eeg_dim=None, kin_dim=None):
        super().__init__()
        mc      = cfg["model"]
        eeg_dim = eeg_dim if eeg_dim is not None else mc.get("eeg_dim", 32)
        kin_dim = kin_dim if kin_dim is not None else mc.get("kin_dim", 13)
        d_model = mc.get("d_model", 64)
        out_ch  = mc["decoder"]["out_channels"]
        self.n_heads = mc["transformer"].get("n_heads", 8)

        # --- Config-driven options ---
        self.chunk_size        = mc.get("chunk_size", 500)
        self.local_attn_window = mc.get("local_attn_window", None)  # int or None
        pos_type               = mc.get("pos_encoding", "learnable")

        self.eeg_cnn = TemporalCNNEncoder(eeg_dim, d_model)
        self.kin_cnn = TemporalCNNEncoder(kin_dim, d_model)
        self.eeg_pos = _build_pos_enc(pos_type, d_model)
        self.kin_pos = _build_pos_enc(pos_type, d_model)

        self.cross_attn_eeg_kin = nn.MultiheadAttention(d_model, self.n_heads, batch_first=True)
        self.cross_attn_kin_eeg = nn.MultiheadAttention(d_model, self.n_heads, batch_first=True)

        # FIX: LearnableGatedFusion receives TWO SEPARATE streams
        self.fusion = LearnableGatedFusion(d_model)

        # KEY FIX: inject PE onto the fused stream BEFORE the Transformer.
        # Cross-attention value projections destroy positional info, so fused
        # is position-blind without this — causing vertical-stripe attention collapse.
        self.fused_pos = _build_pos_enc(pos_type, d_model)

        tc = mc["transformer"]
        self.transformer = OptimizedTransformerEncoder(
            d_model=d_model, n_heads=self.n_heads,
            n_layers=tc.get("n_layers", 4),
            ffn_dim=tc.get("ffn_dim", 256),
            dropout=tc.get("dropout", 0.1)
        )
        self.node_expansion = nn.Linear(d_model, out_ch * d_model)
        self.gat     = OptimizedKG_GAT(node_dim=d_model, kin_dim=kin_dim, out_nodes=out_ch)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, eeg, kin, return_attention=False):
        B, T, _ = eeg.shape
        out_ch  = self.gat.out_nodes
        d_model = self.node_expansion.out_features // out_ch

        eeg_feat = self.eeg_pos(self.eeg_cnn(eeg))
        kin_feat = self.kin_pos(self.kin_cnn(kin))

        # --- Chunking Trick to avoid O(T^2) attention memory/time explosion ---
        is_chunked = False
        if self.chunk_size is not None and T > self.chunk_size and T % self.chunk_size == 0:
            num_chunks = T // self.chunk_size
            eeg_feat = eeg_feat.view(B * num_chunks, self.chunk_size, -1)
            kin_feat = kin_feat.view(B * num_chunks, self.chunk_size, -1)
            is_chunked = True

        eeg_kin_feat, eeg_kin_attn = self.cross_attn_eeg_kin(
            query=eeg_feat, key=kin_feat, value=kin_feat,
            need_weights=return_attention, average_attn_weights=False
        )
        kin_eeg_feat, kin_eeg_attn = self.cross_attn_kin_eeg(
            query=kin_feat, key=eeg_feat, value=eeg_feat,
            need_weights=return_attention, average_attn_weights=False
        )

        # FIX: gate DIRECTLY between two distinct cross-attention streams
        fused, g_t = self.fusion(eeg_kin_feat, kin_eeg_feat)

        # Re-inject positional encoding onto fused before the Transformer.
        # Cross-attention value projections discard PE, making fused position-blind.
        # Without this, self-attention collapses to vertical stripes.
        fused = self.fused_pos(fused)

        # --- Optional local attention mask for the self-attention Transformer ---
        attn_mask = None
        if self.local_attn_window is not None:
            chunk_T = fused.shape[1]  # may be chunk_size when chunking is active
            attn_mask = _make_local_mask(chunk_T, self.local_attn_window, fused.device)

        temporal, self_attn_maps = self.transformer(
            fused, return_attention=return_attention, attn_mask=attn_mask
        )

        if is_chunked:
            temporal = temporal.view(B, T, -1)
            g_t = g_t.view(B, T, -1)
            fused = fused.view(B, T, -1)

        nodes   = self.node_expansion(temporal).view(B, T, out_ch, d_model)
        refined = self.gat(nodes, kin)
        preds   = self.decoder(refined).squeeze(-1)  # (B, T, out_ch)

        output = {"prediction": preds, "gate": g_t, "fused": fused}
        if return_attention:
            output["cross_attn_eeg_kin"] = eeg_kin_attn
            output["cross_attn_kin_eeg"] = kin_eeg_attn
            output["self_attn"]          = self_attn_maps
        return output


def build_model_from_config(cfg, input_dim=None, kin_dim=None):
    return HybridKGGTModel(cfg, eeg_dim=input_dim, kin_dim=kin_dim)
