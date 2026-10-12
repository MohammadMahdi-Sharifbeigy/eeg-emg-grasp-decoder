import copy
import json

import numpy as np
import torch

from v2 import data as D
from v2.config import CONFIG_V2
from v2.model import build_model, stream_of
from v2.train import fit, load_best


def _datasets():
    def raw(seed, T=12000):
        rng = np.random.default_rng(seed)
        kin = rng.standard_normal((T, 26)).astype(np.float32); kin[:, 10] = 0; kin[:, 11] = 0
        emg = np.abs(rng.standard_normal((T, 5))).astype(np.float32) * 0.01
        for a in range(1000, T - 2000, 3000):
            kin[a:a + 1500, 10] = 4.0; kin[a:a + 1500, 11] = 6.0; emg[a:a + 1500] += 0.05
        return {"series": seed, "eeg": rng.standard_normal((T, 32)).astype(np.float32), "kin": kin, "emg": emg}
    tr = [D.build_tokens(raw(s), CONFIG_V2) for s in (0, 1)]
    va = [D.build_tokens(raw(2), CONFIG_V2)]
    norm = D.fit_norm(tr)
    return D.TokenWindows(tr, norm, random_offset=True), D.TokenWindows(va, norm), norm


def _cfg(epochs=2, patience=15):
    c = copy.deepcopy(CONFIG_V2)
    c["training"].update(max_epochs=epochs, patience=patience, batch_size=4, use_amp=False, warmup_epochs=1)
    return c


def test_fit_runs_writes_checkpoint_and_learns_k(tmp_path):
    tr, va, _ = _datasets()
    torch.manual_seed(0)
    m = build_model(CONFIG_V2)
    h = fit(m, tr, va, _cfg(epochs=3), torch.device("cpu"), tmp_path, log=lambda *_: None)
    assert h["epochs_run"] == 3 and len(h["train"]) == 3 and len(h["val_kin"]) == 3 and len(h["components"]) == 3
    assert h["val_kin"][-1] < h["val_kin"][0]                         # the kinematic stream sees the planted hold
    assert (tmp_path / "best.pt").exists() and json.loads((tmp_path / "history.json").read_text())["epochs_run"] == 3
    assert 1 <= h["best_epoch_kin"] <= 3 and 1 <= h["best_epoch_eeg"] <= 3


def test_resume_skips_training_and_restores_weights(tmp_path):
    tr, va, _ = _datasets()
    m = build_model(CONFIG_V2)
    fit(m, tr, va, _cfg(epochs=1), torch.device("cpu"), tmp_path, log=lambda *_: None)
    saved = {k: v.clone() for k, v in m.state_dict().items()}
    m2 = build_model(CONFIG_V2)
    h2 = fit(m2, tr, va, _cfg(epochs=5), torch.device("cpu"), tmp_path, log=lambda *_: None)
    assert h2["epochs_run"] == 1
    assert all(torch.equal(saved[k], v) for k, v in m2.state_dict().items())
    m3 = build_model(CONFIG_V2)
    assert load_best(m3, tmp_path, torch.device("cpu"))["epochs_run"] == 1


def test_streams_stop_independently(tmp_path):
    tr, va, _ = _datasets()
    m = build_model(CONFIG_V2)
    h = fit(m, tr, va, _cfg(epochs=6, patience=1), torch.device("cpu"), tmp_path, log=lambda *_: None)
    assert h["epochs_run"] <= 6 and h["best_epoch_kin"] >= 1 and h["best_epoch_eeg"] >= 1
    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)["state"]
    assert {stream_of(k) for k in best} == {"kin", "eeg"}


def _stream_norm(model, stream):
    return torch.sqrt(sum(p.grad.pow(2).sum() for n, p in model.named_parameters() if stream_of(n) == stream))


def test_gradients_are_clipped_per_stream():
    from v2.train import clip_per_stream
    torch.manual_seed(0)
    m = build_model(CONFIG_V2)
    for big, small in (("eeg", "kin"), ("kin", "eeg")):
        for n, p in m.named_parameters():                             # big stream: norm far above 1; small: far below
            p.grad = torch.randn_like(p) * (10.0 if stream_of(n) == big else 1e-5)
        before = {n: p.grad.clone() for n, p in m.named_parameters()}
        assert _stream_norm(m, big) > 100.0 and _stream_norm(m, small) < 0.1
        clip_per_stream(m, 1.0)
        for n, p in m.named_parameters():                             # joint clipping would shrink these too
            if stream_of(n) == small:
                assert torch.equal(p.grad, before[n]), n
        assert _stream_norm(m, big) <= 1.0 + 1e-4
        assert _stream_norm(m, small) <= 1.0 + 1e-4
