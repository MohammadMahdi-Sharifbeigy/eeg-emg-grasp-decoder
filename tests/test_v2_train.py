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
    h2 = fit(m2, tr, va, _cfg(epochs=1), torch.device("cpu"), tmp_path, log=lambda *_: None)   # finished run: reused
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


def test_one_optimizer_per_stream_and_gate_without_weight_decay():
    from v2.train import build_optimizers
    m = build_model(CONFIG_V2)
    tc = CONFIG_V2["training"]
    opts = build_optimizers(m, tc)
    name_of = {id(p): n for n, p in m.named_parameters()}
    seen = []
    for stream, opt in opts.items():
        for g in opt.param_groups:
            names = [name_of[id(p)] for p in g["params"]]
            assert all(stream_of(n) == stream for n in names)          # no optimizer holds the other stream
            assert g["weight_decay"] == (0.0 if "fusion.gate" in names else tc["weight_decay"])
            assert names == ["fusion.gate"] or "fusion.gate" not in names
            seen += names
    assert sorted(seen) == sorted(name_of.values())                    # every parameter, exactly once
    assert set(opts) == {"kin", "eeg"} and "fusion.gate" in seen


def test_non_finite_eeg_gradients_do_not_stop_the_kinematic_step():
    from v2.train import build_optimizers, step_streams
    torch.manual_seed(0)
    m = build_model(CONFIG_V2)
    opts = build_optimizers(m, CONFIG_V2["training"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    for bad in (float("nan"), float("inf")):
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        for n, p in m.named_parameters():
            p.grad = torch.randn_like(p)
        m.eeg.up.weight.grad[0, 0] = bad                              # one bad entry spoils the EEG stream's step
        stepped = step_streams(m, opts, scaler, 1.0)
        assert stepped == {"kin": True, "eeg": False}
        for n, p in m.named_parameters():
            assert bool(torch.isfinite(p).all()), n
            if stream_of(n) == "eeg":
                assert torch.equal(p, before[n]), n                   # skipped: not even weight decay
        assert not torch.equal(m.kin.stem.weight, before["kin.stem.weight"])
        assert all(p.grad is None for p in m.parameters())


def test_combined_best_model_is_validated(tmp_path):
    tr, va, _ = _datasets()
    torch.manual_seed(0)
    m = build_model(CONFIG_V2)
    logged = []
    h = fit(m, tr, va, _cfg(epochs=2), torch.device("cpu"), tmp_path, log=logged.append)
    vc = h["val_combined"]
    assert set(vc) == {"kin", "eeg", "components"} and set(vc["components"]) == set(h["components"][0])
    # K never reads the EEG stream, so the combined model repeats K's best epoch exactly
    assert abs(vc["kin"] - h["val_kin"][h["best_epoch_kin"] - 1]) < 1e-6
    assert np.isfinite(vc["eeg"])
    assert json.loads((tmp_path / "history.json").read_text())["val_combined"]["kin"] == vc["kin"]
    assert sum("combined best model" in line for line in logged) == 1


def test_resume_reuses_continues_or_ignores_a_checkpoint(tmp_path):
    tr, va, _ = _datasets()
    m = build_model(CONFIG_V2)
    fit(m, tr, va, _cfg(epochs=1), torch.device("cpu"), tmp_path, log=lambda *_: None)
    stored = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)["config"]
    assert stored == {k: CONFIG_V2[k] for k in ("model", "tokens", "eeg", "history", "loss")}
    other = _cfg(epochs=1)
    other["loss"]["w_ccc"] = 0.0
    same = fit(build_model(CONFIG_V2), tr, va, _cfg(epochs=1), torch.device("cpu"), tmp_path, log=lambda *_: None)
    assert same["epochs_run"] == 1                                    # a finished run of the same config is reused
    longer, said = _cfg(epochs=3), []                                 # the training block is not part of the identity:
    h = fit(build_model(CONFIG_V2), tr, va, longer, torch.device("cpu"), tmp_path, log=said.append)
    assert h["epochs_run"] == 3 and any("resumed" in s for s in said)  # an unfinished run continues from last.pt
    said = []                                                         # another loss: the checkpoint is not used
    h = fit(build_model(CONFIG_V2), tr, va, other, torch.device("cpu"), tmp_path, log=said.append)
    assert any("config mismatch" in s and "'loss'" in s for s in said) and h["epochs_run"] == 1
    assert load_best(build_model(CONFIG_V2), tmp_path, torch.device("cpu"))["epochs_run"] == 1


def test_weighted_stream_loss_is_the_validation_loss_of_one_stream(tmp_path):
    from v2.train import weighted_stream_loss
    comps = {"E/main": 1.0, "E/ccc": 2.0, "E/lift": 4.0, "E/diff": 8.0, "K/main": 100.0, "F/main": 100.0}
    cfg = {"w_main": 1.0, "w_ccc": 0.3, "w_lift": 0.5, "w_diff": 0.1, "huber_delta": 0.5}
    assert abs(weighted_stream_loss(comps, "E", cfg) - (1.0 + 0.6 + 2.0 + 0.8)) < 1e-12
    assert isinstance(weighted_stream_loss(comps, "E", cfg), float)
    tr, va, _ = _datasets()
    h = fit(build_model(CONFIG_V2), tr, va, _cfg(epochs=1), torch.device("cpu"), tmp_path, log=lambda *_: None)
    k, e, f = (weighted_stream_loss(h["components"][0], s, CONFIG_V2["loss"]) for s in ("K", "E", "F"))
    assert abs(k - h["val_kin"][0]) < 1e-5 and abs(e + f - h["val_eeg"][0]) < 1e-5   # the parts of the logged losses


def test_skipped_steps_are_counted_per_epoch_and_stream(tmp_path, monkeypatch):
    import v2.train as T
    real, calls = T.clip_per_stream, []

    def clip(model, max_norm):                                        # the EEG stream's first two steps: no finite norm
        norms = real(model, max_norm)
        calls.append(1)
        return {**norms, "eeg": float("nan")} if len(calls) <= 2 else norms

    monkeypatch.setattr(T, "clip_per_stream", clip)
    tr, va, _ = _datasets()
    logged = []
    h = fit(build_model(CONFIG_V2), tr, va, _cfg(epochs=2), torch.device("cpu"), tmp_path, log=logged.append)
    assert h["skipped_eeg"] == [2, 0] and h["skipped_kin"] == [0, 0]
    told = [line for line in logged if "skipped steps" in line]
    assert len(told) == 1 and "epoch   1" in told[0] and "skipped steps K 0, E 2" in told[0]
    assert json.loads((tmp_path / "history.json").read_text())["skipped_eeg"] == [2, 0]


def test_a_stale_stream_is_frozen_at_its_best_while_the_other_keeps_training(tmp_path, monkeypatch):
    import v2.train as T
    tr, va, _ = _datasets()
    now, kin_at = {"epoch": 0}, {}
    set_epoch = tr.set_epoch
    monkeypatch.setattr(tr, "set_epoch", lambda e: (now.update(epoch=e + 1), set_epoch(e))[1])
    real = T.evaluate_loss
    k_script = {1: 1.0, 2: 0.9}                                         # K: best at epoch 2, worse from epoch 3 on

    def scripted(model, loader, crit, device, use_amp):                 # E + F: better every epoch
        _, _, comps = real(model, loader, crit, device, use_amp)
        ep = now["epoch"]
        kin_at.setdefault(ep, {k: v.clone() for k, v in model.kin.state_dict().items()})
        return k_script.get(ep, 0.95), 1.0 / ep, comps

    monkeypatch.setattr(T, "evaluate_loss", scripted)
    cfg = _cfg(epochs=8, patience=1)
    cfg["training"]["smooth_window"] = 1
    torch.manual_seed(0)
    m = build_model(CONFIG_V2)
    said = []
    h = fit(m, tr, va, cfg, torch.device("cpu"), tmp_path, log=said.append)
    assert h["frozen_epoch"] == {"kin": 3, "eeg": 0} and h["best_epoch_kin"] == 2 and h["epochs_run"] == 8
    assert any("K frozen at its best epoch 2" in s for s in said)
    same = lambda a, b: all(torch.equal(a[k], b[k]) for k in a)
    assert not same(kin_at[2], kin_at[3])                               # K was still training in epoch 3
    assert same(kin_at[2], kin_at[4]) and same(kin_at[2], kin_at[8])    # frozen: back at epoch 2 and never updated again
    assert same(kin_at[2], {k: v for k, v in m.kin.state_dict().items()})
    assert h["best_epoch_eeg"] == 8                                     # E + fusion kept training and improving
    assert all(p.requires_grad for p in m.parameters())                 # the returned model is not left frozen
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    assert last["history"]["frozen_epoch"]["kin"] == 3                  # a resumed run would keep K frozen
