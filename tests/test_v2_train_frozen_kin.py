import copy

import numpy as np
import pytest
import torch

from v2 import data as D
from v2.config import CONFIG_V2
from v2.model import build_model
from v2.train import fit, load_kin

CPU = torch.device("cpu")
QUIET = lambda *_: None
CENTRAL = ["C3", "Cz", "C4", "CP1", "CP2", "CP5", "CP6"]


def _datasets(cfg=CONFIG_V2):
    def raw(seed, T=12000):
        rng = np.random.default_rng(seed)
        kin = rng.standard_normal((T, 26)).astype(np.float32); kin[:, 10] = 0; kin[:, 11] = 0
        emg = np.abs(rng.standard_normal((T, 5))).astype(np.float32) * 0.01
        for a in range(1000, T - 2000, 3000):
            kin[a:a + 1500, 10] = 4.0; kin[a:a + 1500, 11] = 6.0; emg[a:a + 1500] += 0.05
        return {"series": seed, "eeg": rng.standard_normal((T, 32)).astype(np.float32), "kin": kin, "emg": emg}
    tr = [D.build_tokens(raw(s), cfg) for s in (0, 1)]
    va = [D.build_tokens(raw(2), cfg)]
    norm = D.fit_norm(tr)
    return D.TokenWindows(tr, norm, random_offset=True), D.TokenWindows(va, norm)


def _cfg(epochs=2, patience=15, base=CONFIG_V2, freeze_kin=None):
    c = copy.deepcopy(base)
    c["training"].update(max_epochs=epochs, patience=patience, batch_size=4, use_amp=False, warmup_epochs=1)
    if freeze_kin is not None:
        c["training"]["freeze_kin"] = freeze_kin
    return c


def _kin(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith("kin.")}


@pytest.fixture(scope="module")
def k_ckpt(tmp_path_factory):
    """best.pt of a normally trained 32-electrode model, and its K tensors."""
    folder = tmp_path_factory.mktemp("k")
    tr, va = _datasets()
    torch.manual_seed(0)
    fit(build_model(CONFIG_V2), tr, va, _cfg(epochs=2), CPU, folder, log=QUIET)
    st = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)["best_state"]
    return folder / "best.pt", {k: v.clone() for k, v in st.items() if k.startswith("kin.")}


def test_frozen_kin_stays_identical(tmp_path, k_ckpt):
    path, loaded = k_ckpt
    tr, va = _datasets()
    torch.manual_seed(1)
    m = build_model(CONFIG_V2)
    n = load_kin(m, path)
    assert n == len(loaded) and n > 0
    assert all(torch.equal(v, loaded[k]) for k, v in _kin(m).items())
    eeg_before = {k: v.detach().clone() for k, v in m.state_dict().items() if k.startswith("eeg.")}
    h = fit(m, tr, va, _cfg(epochs=2, freeze_kin=True), CPU, tmp_path / "e", log=QUIET)
    assert h["frozen_epoch"]["kin"] == -1 and h["best_epoch_kin"] == 0 and h["epochs_run"] == 2
    assert all(torch.equal(v, loaded[k]) for k, v in _kin(m).items())        # K never moved
    best = torch.load(tmp_path / "e" / "best.pt", map_location="cpu", weights_only=False)["best_state"]
    assert all(torch.equal(best[k], loaded[k]) for k in loaded)              # and is what was saved
    state = m.state_dict()
    assert any(not torch.equal(state[k], v) for k, v in eeg_before.items())  # E + fusion trained
    assert all(p.requires_grad for p in m.parameters())


def test_frozen_kin_keeps_k_in_eval_mode(tmp_path, monkeypatch, k_ckpt):
    import v2.train as T
    path, _ = k_ckpt
    tr, va = _datasets()
    m = build_model(CONFIG_V2)
    load_kin(m, path)
    modes = []
    real = T.v2_loss

    def spy(out, *a, **kw):
        if m.training:                                                    # a training batch
            modes.append((m.kin.training, m.eeg.training))
        return real(out, *a, **kw)

    monkeypatch.setattr(T, "v2_loss", spy)
    fit(m, tr, va, _cfg(epochs=1, freeze_kin=True), CPU, tmp_path / "e", log=QUIET)
    assert modes and all(k is False and e is True for k, e in modes)


def test_load_kin_reports_missing_file(tmp_path):
    missing = tmp_path / "nowhere" / "best.pt"
    with pytest.raises(FileNotFoundError, match="nowhere"):
        load_kin(build_model(CONFIG_V2), missing)


def test_load_kin_rejects_a_different_k_shape(k_ckpt):
    path, _ = k_ckpt
    other = copy.deepcopy(CONFIG_V2)
    other["model"]["d_model"] = CONFIG_V2["model"]["d_model"] * 2
    with pytest.raises(ValueError, match="kin\\."):
        load_kin(build_model(other), path)


def test_frozen_kin_resume_keeps_k(tmp_path, k_ckpt):
    path, loaded = k_ckpt
    tr, va = _datasets()
    m = build_model(CONFIG_V2)
    load_kin(m, path)
    h1 = fit(m, tr, va, _cfg(epochs=2, freeze_kin=True), CPU, tmp_path / "e", log=QUIET)
    said = []
    m2 = build_model(CONFIG_V2)                                           # a new process: K not loaded yet
    load_kin(m2, path)
    h2 = fit(m2, tr, va, _cfg(epochs=2, freeze_kin=True), CPU, tmp_path / "e", log=said.append)
    assert any("training skipped" in s for s in said)
    assert h2["epochs_run"] == h1["epochs_run"] and h2["frozen_epoch"]["kin"] == -1
    assert all(torch.equal(v, loaded[k]) for k, v in _kin(m2).items())
    assert all(torch.equal(m.state_dict()[k], v) for k, v in m2.state_dict().items())
    m3 = build_model(CONFIG_V2)                                           # even without load_kin, best.pt holds K
    fit(m3, tr, va, _cfg(epochs=2, freeze_kin=True), CPU, tmp_path / "e", log=QUIET)
    assert all(torch.equal(v, loaded[k]) for k, v in _kin(m3).items())


def test_freeze_kin_absent_changes_nothing(tmp_path):
    tr, va = _datasets()
    torch.manual_seed(0)
    m0 = build_model(CONFIG_V2)
    start = _kin(m0)
    torch.manual_seed(0)
    h0 = fit(m0, tr, va, _cfg(epochs=2), CPU, tmp_path / "a", log=QUIET)
    tr, va = _datasets()
    torch.manual_seed(0)
    m1 = build_model(CONFIG_V2)
    torch.manual_seed(0)
    h1 = fit(m1, tr, va, _cfg(epochs=2, freeze_kin=False), CPU, tmp_path / "b", log=QUIET)
    assert set(h0) == set(h1)
    assert h0["frozen_epoch"] == {"kin": 0, "eeg": 0} and h0["best_epoch_kin"] >= 1
    assert h0["val_kin"] == h1["val_kin"] and h0["val_eeg"] == h1["val_eeg"]
    assert any(not torch.equal(v, start[k]) for k, v in _kin(m0).items())   # K trained as before


def test_frozen_kin_from_a_model_with_more_electrodes(tmp_path, k_ckpt):
    path, loaded = k_ckpt                                   # K trained with all 32 electrodes
    cfg = _cfg(epochs=2, freeze_kin=True)
    cfg["eeg"]["channels"] = list(CENTRAL)
    tr, va = _datasets(cfg)
    m = build_model(cfg)
    full = build_model(CONFIG_V2).state_dict()
    assert m.state_dict()["eeg.identity"].shape != full["eeg.identity"].shape   # the EEG side differs in shape
    assert load_kin(m, path) == len(loaded)
    h = fit(m, tr, va, cfg, CPU, tmp_path / "e", log=QUIET)
    assert h["frozen_epoch"]["kin"] == -1 and h["epochs_run"] == 2
    assert all(torch.equal(v, loaded[k]) for k, v in _kin(m).items())
