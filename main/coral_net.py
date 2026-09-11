"""CORAL-Net: Corticomuscular Ordered Regression with Aligned Latents.

NeuroAI Architecture for continuous EEG-to-EMG corticomuscular decoding on WAY-EEG-GAL:
1. Learnable Causal Filterbank Frontend (physiologically initialized FIR conv)
2. Causal State-Space Sequence Modeling (MambaEncoder with TorchScript accelerated selective scan)
3. Differentiable Conduction Lag Alignment (linear-shifted Fourier fractional phase modulation)
4. Muscle Synergy Bottleneck Decoder (non-negative activations + NMF warm-started mixing matrix)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================================================================
# TorchScript Accelerated Selective Scan for Mamba
# ============================================================================

@torch.jit.script
def selective_scan_jit(
    u: Tensor,
    deltaA: Tensor,
    deltaB: Tensor,
    C_proj: Tensor,
    D: Tensor,
) -> Tensor:
    """TorchScript accelerated selective state-space scan.
    
    Args:
        u: Input tensor of shape (B, T, D)
        deltaA: Discretized state transition tensor of shape (B, T, D, N)
        deltaB: Discretized input matrix tensor of shape (B, T, D, N)
        C_proj: Output projection tensor of shape (B, T, N)
        D: Feedthrough skip parameter of shape (D,)
        
    Returns:
        Scanned hidden states y of shape (B, T, D)
    """
    B = u.shape[0]
    T = u.shape[1]
    dim = u.shape[2]
    N = deltaA.shape[3]

    # Pre-allocate output container
    h = torch.zeros(B, dim, N, device=u.device, dtype=u.dtype)
    ys = []

    for t in range(T):
        u_t = u[:, t].unsqueeze(-1)                          # (B, dim, 1)
        h = deltaA[:, t] * h + deltaB[:, t] * u_t            # (B, dim, N)
        C_t = C_proj[:, t].unsqueeze(1)                      # (B, 1, N)
        y_t = (h * C_t).sum(-1) + D * u[:, t]                # (B, dim)
        ys.append(y_t)

    return torch.stack(ys, dim=1)                            # (B, T, dim)


# ============================================================================
# 1. Learnable Causal Filterbank Frontend
# ============================================================================

class LearnableFilterBank(nn.Module):
    """Physiologically initialized, strictly causal FIR bandpass filter bank.
    
    Initializes depthwise 1D convolutional kernels at standard EEG frequency bands:
    - Delta: [0.5, 4.0] Hz
    - Alpha: [8.0, 12.0] Hz
    - Beta:  [13.0, 30.0] Hz
    - Gamma: [30.0, 80.0] Hz
    
    Causality is enforced via left-only padding (padding = kernel_size - 1)
    ensuring that y[t] depends strictly on x[t - k] for k >= 0 with zero lookahead.
    """

    BAND_FREQS: Dict[str, Tuple[float, float]] = {
        "delta": (0.5, 4.0),
        "alpha": (8.0, 12.0),
        "beta":  (13.0, 30.0),
        "gamma": (30.0, 80.0),
    }

    def __init__(
        self,
        in_channels: int = 32,
        kernel_size: int = 125,
        fs: float = 500.0,
        d_model: int = 256,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.fs = fs
        self.n_bands = len(self.BAND_FREQS)
        self.d_model = d_model

        # Total depthwise output channels = in_channels * n_bands (e.g. 32 * 4 = 128)
        total_band_channels = self.in_channels * self.n_bands

        # Depthwise 1D Conv: groups = in_channels
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=total_band_channels,
            kernel_size=kernel_size,
            padding=0,                  # We handle causal left-padding explicitly in forward
            groups=in_channels,
            bias=False,
        )

        # Initialize filter kernels with physiological FIR bandpass
        self._init_fir_kernels()

        # Pointwise projection: (32 * 4) -> d_model
        self.pointwise_conv = nn.Conv1d(total_band_channels, d_model, kernel_size=1, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.SiLU()

    def _init_fir_kernels(self) -> None:
        """Initialize depthwise weights using scipy.signal.firwin."""
        # Ensure odd kernel size for symmetric linear phase FIR prototype
        k_size = self.kernel_size if self.kernel_size % 2 == 1 else self.kernel_size + 1
        
        band_kernels = []
        for low, high in self.BAND_FREQS.values():
            # Standard windowed sinc FIR bandpass
            fir = scipy.signal.firwin(
                k_size,
                [low, high],
                pass_zero=False,
                fs=self.fs,
                window="hamming",
            )
            if len(fir) > self.kernel_size:
                fir = fir[:self.kernel_size]
            band_kernels.append(torch.tensor(fir, dtype=torch.float32))

        # Stack to shape: (n_bands, kernel_size)
        stacked_kernels = torch.stack(band_kernels, dim=0)

        # Replicate for each input channel: (in_channels * n_bands, 1, kernel_size)
        # Groups = in_channels, so Conv1d weight shape is (out_channels, in_channels/groups, kernel_size) = (out_channels, 1, kernel_size)
        w_init = torch.zeros(self.in_channels * self.n_bands, 1, self.kernel_size, dtype=torch.float32)
        for c in range(self.in_channels):
            for b in range(self.n_bands):
                idx = c * self.n_bands + b
                w_init[idx, 0, :] = stacked_kernels[b]

        with torch.no_grad():
            self.depthwise_conv.weight.copy_(w_init)

    def forward(self, eeg: Tensor) -> Tensor:
        """Forward pass for causal filterbank.
        
        Args:
            eeg: Input EEG tensor of shape (B, T, in_channels)
            
        Returns:
            Projected latent representation of shape (B, T, d_model)
        """
        B, T, C = eeg.shape
        # Permute to (B, C, T) for Conv1d
        x = eeg.transpose(1, 2)

        # Strictly causal left-padding: add (kernel_size - 1) zeros on the left
        pad_left = self.kernel_size - 1
        x_padded = F.pad(x, (pad_left, 0), mode="constant", value=0.0)

        # Depthwise filtering: output length is exactly T
        x_filtered = self.depthwise_conv(x_padded)           # (B, total_band_channels, T)

        # Pointwise projection and activation
        x_proj = self.act(self.pointwise_conv(x_filtered))   # (B, d_model, T)

        # Permute back to (B, T, d_model) and normalize
        x_out = self.norm(x_proj.transpose(1, 2))
        return x_out


# ============================================================================
# 2. Causal State-Space Sequence Modeling (SimpleMambaBlock & MambaEncoder)
# ============================================================================

class SimpleMambaBlock(nn.Module):
    """Selective state-space model (SSM) block implemented in pure PyTorch.
    
    Provides O(T) complexity, strictly causal receptive field, and input-dependent
    selection mechanisms with TorchScript JIT acceleration.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 16,
        expand: int = 2,
        conv_kernel: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = d_model * expand
        self.conv_kernel = conv_kernel

        # Input projection: projects x to branch u and gating branch z
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise 1D conv on branch u
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=conv_kernel,
            padding=0,                              # Causal left-pad in forward
            groups=self.d_inner,
            bias=True,
        )

        # Projection to SSM parameters: dt, B, C
        self.dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Parameterize transition matrix A: A = -exp(A_log)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Feedthrough skip parameter D
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for selective state-space block.
        
        Args:
            x: Input tensor of shape (B, T, d_model)
            
        Returns:
            Output tensor of shape (B, T, d_model)
        """
        residual = x
        x_norm = self.norm(x)
        B, T, _ = x_norm.shape

        # Dual projection: (B, T, 2 * d_inner) -> u, z
        xz = self.in_proj(x_norm)
        u, z = xz.chunk(2, dim=-1)                           # each (B, T, d_inner)

        # Causal depthwise convolution on u
        pad_left = self.conv_kernel - 1
        u_conv_in = F.pad(u.transpose(1, 2), (pad_left, 0), mode="constant", value=0.0)
        u_conv = F.silu(self.conv1d(u_conv_in)).transpose(1, 2)  # (B, T, d_inner)

        # Input-dependent projections
        x_dbl = self.x_proj(u_conv)                          # (B, T, dt_rank + 2*d_state)
        dt_in, B_proj, C_proj = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = F.softplus(self.dt_proj(dt_in))                 # (B, T, d_inner)

        # Discretize continuous SSM parameters A and B:
        # A = -exp(A_log) -> (d_inner, d_state)
        A = -torch.exp(self.A_log)
        # dA = exp(dt * A) -> (B, T, d_inner, d_state)
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        # dB = dt * B -> (B, T, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_proj.unsqueeze(2)

        # TorchScript accelerated selective scan
        y = selective_scan_jit(u_conv, dA, dB, C_proj, self.D)

        # Multiplicative gating with branch z
        y_gated = y * F.silu(z)

        # Output projection and residual addition
        out = residual + self.out_proj(y_gated)
        return out


class MambaEncoder(nn.Module):
    """Cascaded 4-layer Mamba sequence encoder."""

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleMambaBlock(d_model=d_model, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ============================================================================
# 3. Differentiable Conduction Lag Alignment (Frequency Domain Linear Shift)
# ============================================================================

class LearnableLagAlignment(nn.Module):
    """Differentiable fractional time-shift via Fourier phase-ramp modulation.
    
    Models corticospinal motor conduction latency (~20–100 ms) organically.
    To prevent circular convolution wrap-around artifacts (which would reintroduce
    acausal lookahead from the end of the window), the sequence is zero-padded on
    the right by max_lag_samples before FFT, and sliced back to original length.
    
    tau = sigmoid(lag_logit) * max_lag_s
    Initialized at lag_logit = 0.0 -> tau = 0.5 * 100 ms = 50 ms.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_lag_ms: float = 100.0,
        fs: float = 500.0,
        per_channel: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_lag_ms = max_lag_ms
        self.max_lag_s = max_lag_ms / 1000.0
        self.fs = fs
        self.max_lag_samples = int(math.ceil(self.max_lag_s * fs))

        # Initial logit at 0.0 corresponds to sigmoid(0.0) = 0.5 -> 50 ms delay
        if per_channel:
            self.lag_logit = nn.Parameter(torch.zeros(d_model))
        else:
            self.lag_logit = nn.Parameter(torch.tensor([0.0]))

    @property
    def current_lag_ms(self) -> Tensor:
        """Current learned lag in milliseconds."""
        return torch.sigmoid(self.lag_logit) * self.max_lag_ms

    def forward(self, x: Tensor) -> Tensor:
        """Applies linear fractional delay to sequence x along dimension 1.
        
        Args:
            x: Tensor of shape (B, T, D)
            
        Returns:
            Delayed tensor of shape (B, T, D)
        """
        B, T, D = x.shape
        pad_len = self.max_lag_samples
        T_pad = T + pad_len

        # Zero-pad right by max_lag_samples to convert circular shift into linear shift
        x_padded = F.pad(x, (0, 0, 0, pad_len), mode="constant", value=0.0)

        # Real FFT along temporal dimension
        X_fft = torch.fft.rfft(x_padded, dim=1)              # (B, T_pad//2 + 1, D)
        freqs = torch.fft.rfftfreq(T_pad, d=1.0 / self.fs, device=x.device)  # (N_freq,)

        # Compute delay tau in seconds
        tau = torch.sigmoid(self.lag_logit) * self.max_lag_s # (1,) or (D,)

        # Phase modulation: H(f) = exp(-j * 2 * pi * f * tau)
        if tau.numel() == 1:
            phase = -2.0 * torch.pi * freqs.unsqueeze(-1) * tau   # (N_freq, 1)
        else:
            phase = -2.0 * torch.pi * freqs.unsqueeze(-1) * tau.unsqueeze(0)  # (N_freq, D)

        shift = torch.polar(torch.ones_like(phase), phase).unsqueeze(0)       # (1, N_freq, D or 1)
        X_shifted = X_fft * shift

        # Inverse real FFT and extract strictly the valid linear window [0, T)
        x_shifted = torch.fft.irfft(X_shifted, n=T_pad, dim=1).real.to(x.dtype)
        x_out = x_shifted[:, :T, :]
        return x_out


# ============================================================================
# 4. Muscle Synergy Bottleneck Decoder
# ============================================================================

class SynergyDecoder(nn.Module):
    """Muscle synergy decoder enforcing biological non-negativity constraints.
    
    1. Synergy Head: projects d_model -> 128 -> n_synergies via Softplus (s(t) >= 0).
    2. Mixing Matrix W: Softplus(synergy_mix) in R_{+}^{n_synergies x n_muscles}.
    3. Reconstruction: y_emg(t) = s(t) @ W.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_muscles: int = 5,
        n_synergies: int = 3,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_muscles = n_muscles
        self.n_synergies = n_synergies

        # Synergy activation head
        self.synergy_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, n_synergies),
            nn.Softplus(),                                   # Enforces s(t) >= 0
        )

        # Unconstrained parameter matrix for mixing matrix W: W = Softplus(synergy_mix)
        self.synergy_mix = nn.Parameter(torch.randn(n_synergies, n_muscles))

        # Optional linear scaling per channel
        self.out_scale = nn.Linear(n_muscles, n_muscles, bias=True)

    @property
    def W(self) -> Tensor:
        """Effective non-negative mixing matrix of shape (n_synergies, n_muscles)."""
        return F.softplus(self.synergy_mix)

    def init_synergy_mix_from_nmf(self, H: Tensor) -> None:
        """Warm-starts synergy mixing matrix from offline NMF components.
        
        Args:
            H: (n_synergies, n_muscles) non-negative component matrix.
        """
        with torch.no_grad():
            H_clamped = torch.clamp(H.float(), min=1e-4)
            # Invert softplus: W_param = log(exp(H) - 1)
            # Numerically stable inverse softplus:
            # for H > 20, softplus(H) ~ H, so W_param ~ H
            inv_sp = torch.where(
                H_clamped > 20.0,
                H_clamped,
                torch.log(torch.expm1(H_clamped) + 1e-7),
            )
            self.synergy_mix.copy_(inv_sp)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass for synergy decoder.
        
        Args:
            x: Latent tensor of shape (B, T, d_model)
            
        Returns:
            Tuple of:
            - emg_pred: Predicted EMG envelopes of shape (B, T, n_muscles)
            - synergies: Predicted synergy activations of shape (B, T, n_synergies)
        """
        # 1. Non-negative synergy activations s(t) >= 0
        synergies = self.synergy_head(x)                     # (B, T, n_synergies)

        # 2. Linear combination via non-negative mixing matrix W
        W_nonneg = self.W                                    # (n_synergies, n_muscles)
        emg_raw = torch.matmul(synergies, W_nonneg)          # (B, T, n_muscles)

        # 3. Channel scaling calibration
        emg_pred = self.out_scale(emg_raw)
        return emg_pred, synergies


# ============================================================================
# 5. Full CORAL-Net Model
# ============================================================================

class CORALNet(nn.Module):
    """CORAL-Net: Corticomuscular Ordered Regression with Aligned Latents.
    
    End-to-End Pipeline:
        Raw EEG -> Causal Filterbank -> Mamba SSM -> Conduction Lag -> [Kin Gate] -> Synergy Decoder
    """

    def __init__(
        self,
        n_eeg_channels: int = 32,
        n_muscles: int = 5,
        n_synergies: int = 3,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        fs: float = 500.0,
        max_lag_ms: float = 100.0,
        use_kinematics: bool = True,
        kin_dim: int = 36,
    ) -> None:
        super().__init__()
        self.n_eeg_channels = n_eeg_channels
        self.n_muscles = n_muscles
        self.n_synergies = n_synergies
        self.d_model = d_model
        self.use_kinematics = use_kinematics

        # [1] Learnable Causal Filterbank Frontend
        self.filterbank = LearnableFilterBank(
            in_channels=n_eeg_channels,
            kernel_size=125,
            fs=fs,
            d_model=d_model,
        )

        # [2] Causal State-Space Sequence Modeling
        self.encoder = MambaEncoder(
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
        )

        # [3] Differentiable Corticospinal Conduction Delay
        self.lag = LearnableLagAlignment(
            d_model=d_model,
            max_lag_ms=max_lag_ms,
            fs=fs,
            per_channel=False,
        )

        # [3b] Optional Kinematic Cross-Gating
        if use_kinematics:
            self.kin_proj = nn.Linear(kin_dim, d_model)
            self.kin_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid(),
            )

        # [4] Muscle Synergy Bottleneck Decoder
        self.synergy_decoder = SynergyDecoder(
            d_model=d_model,
            n_muscles=n_muscles,
            n_synergies=n_synergies,
        )

    def init_synergy_mix_from_nmf(self, H: Tensor) -> None:
        """Warm-starts synergy decoder with NMF components."""
        self.synergy_decoder.init_synergy_mix_from_nmf(H)

    @property
    def current_lag_ms(self) -> float:
        """Returns learned conduction delay in ms."""
        return float(self.lag.current_lag_ms.detach().item())

    def forward(
        self,
        eeg: Tensor,
        kin: Optional[Tensor] = None,
        return_synergies: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through CORAL-Net.
        
        Args:
            eeg: (B, T, n_eeg_channels) raw scalp EEG
            kin: (B, T, kin_dim) optional kinematics
            return_synergies: if True, returns (emg_pred, synergies)
            
        Returns:
            emg_pred (B, T, 5) or (emg_pred, synergies (B, T, 3))
        """
        # [1] Causal multi-band filtering
        x = self.filterbank(eeg)                             # (B, T, d_model)

        # [2] Sequence modeling via Mamba state-space layers
        x = self.encoder(x)                                  # (B, T, d_model)

        # [3] Differentiable conduction lag alignment
        x = self.lag(x)                                      # (B, T, d_model)

        # [3b] Kinematic cross-gating
        if self.use_kinematics and kin is not None:
            k = self.kin_proj(kin)                           # (B, T, d_model)
            gate = self.kin_gate(torch.cat([x, k], dim=-1))  # (B, T, d_model)
            x = x * gate + k * (1.0 - gate)

        # [4] Muscle synergy decoding
        emg_pred, synergies = self.synergy_decoder(x)        # (B, T, 5), (B, T, 3)

        if return_synergies:
            return emg_pred, synergies
        return emg_pred


# ============================================================================
# Utilities: NMF Extraction & Factory Builders
# ============================================================================

def extract_nmf_synergies(
    emg_arrays: List[np.ndarray],
    n_synergies: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts muscle synergies from continuous EMG envelopes via NMF.
    
    Args:
        emg_arrays: List of (T, 5) continuous EMG numpy arrays
        n_synergies: Number of synergy components (default: 3)
        random_state: Random seed for NMF convergence
        
    Returns:
        W_acts: Synergy activation profiles (N_total, n_synergies)
        H_comp: Muscle synergy matrix (n_synergies, 5)
    """
    from sklearn.decomposition import NMF

    emg_all = np.concatenate(emg_arrays, axis=0)
    # Ensure non-negativity
    emg_nonneg = np.maximum(emg_all, 0.0)

    nmf = NMF(
        n_components=n_synergies,
        init="nndsvda",
        max_iter=1000,
        random_state=random_state,
    )
    W_acts = nmf.fit_transform(emg_nonneg)
    H_comp = nmf.components_
    return W_acts, H_comp


def build_coral_net_from_config(
    cfg: Dict[str, Any],
    input_dim: int = 32,
    kin_dim: int = 36,
    nmf_H: Optional[Tensor] = None,
) -> CORALNet:
    """Instantiates CORALNet from configuration dictionary."""
    model_cfg = cfg.get("model", {})
    coral_cfg = model_cfg.get("coral", {})
    data_cfg = cfg.get("data", {})

    n_muscles = model_cfg.get("decoder", {}).get("out_channels", 5)
    n_synergies = coral_cfg.get("n_synergies", 3)
    d_model = coral_cfg.get("d_model", 256)
    n_layers = coral_cfg.get("n_layers", 4)
    d_state = coral_cfg.get("d_state", 16)
    fs = float(data_cfg.get("fs_eeg", 500.0))
    max_lag_ms = float(coral_cfg.get("max_lag_ms", 100.0))
    use_kinematics = coral_cfg.get("use_kinematics", True)

    model = CORALNet(
        n_eeg_channels=input_dim,
        n_muscles=n_muscles,
        n_synergies=n_synergies,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        fs=fs,
        max_lag_ms=max_lag_ms,
        use_kinematics=use_kinematics,
        kin_dim=kin_dim,
    )

    if nmf_H is not None:
        model.init_synergy_mix_from_nmf(nmf_H)

    return model
