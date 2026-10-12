import copy

import torch

from v2.attention import RECORDED, recording
from v2.blocks import AttentionBlock
from v2.config import CONFIG_V2
from v2.model import build_model, count_parameters, stream_of

B, T = 2, 60


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, 24, generator=g), torch.rand(B, T, 8, generator=g), torch.randn(B, T, 32, 4, generator=g)


def _listed(layers):
    return (layers["kin"]["temporal"] + layers["kin"]["muscle"] + layers["eeg"]["electrode"] + [layers["eeg"]["pool"]]
            + layers["eeg"]["temporal"] + layers["eeg"]["muscle"] + [layers["fusion"]["cross"]])


def _dropouts(module):
    return [d.p for d in module.modules() if isinstance(d, torch.nn.Dropout)]


def test_outputs_shapes_and_budget():
    m = build_model(CONFIG_V2).eval()
    out = m(*_inputs())
    for k in ("K", "E", "F", "K_base", "gain", "F_eeg"):
        assert out[k].shape == (B, T, 5) and torch.isfinite(out[k]).all(), k
    assert all((out[k] >= 0).all() for k in ("K", "E", "F")) and out["gate"].shape == (5,)
    assert count_parameters(m) <= 1_000_000


def test_every_attention_module_is_recorded_and_listed():
    m = build_model(CONFIG_V2).eval()
    with recording(m) as tape:
        m(*_inputs())
    layers = m.attention_layers()
    in_model = [n for n, mod in m.named_modules() if isinstance(mod, RECORDED)]
    assert sorted(_listed(layers)) == sorted(tape.attn) == sorted(in_model) and len(in_model) == 13
    assert _listed(layers) == list(tape.attn)                             # forward order: as the tape received them
    shape = lambda names: {tuple(tape.attn[n].shape) for n in names}
    assert shape(layers["kin"]["temporal"]) == shape(layers["eeg"]["temporal"]) == {(B, 4, T, T)}
    assert len(layers["kin"]["temporal"]) == 3 and len(layers["eeg"]["temporal"]) == 2
    assert shape(layers["eeg"]["electrode"]) == {(B, T, 4, 32, 32)} and shape([layers["eeg"]["pool"]]) == {(B, T, 1, 1, 32)}
    assert shape(layers["kin"]["muscle"]) == shape(layers["eeg"]["muscle"]) == {(B, T, 4, 5, 5)}
    assert shape([layers["fusion"]["cross"]]) == {(B, 4, T, T)}
    for a in tape.attn.values():
        torch.testing.assert_close(a.sum(-1), torch.ones(a.shape[:-1]), atol=1e-4, rtol=0)


def test_recording_leaves_the_outputs_unchanged():
    m = build_model(CONFIG_V2).eval()
    plain = m(*_inputs())
    with recording(m, grad=True):
        rec = m(*_inputs())
    assert all(torch.equal(plain[k], rec[k]) for k in plain)


def test_exact_parts_of_k_and_f():
    m = build_model(CONFIG_V2).eval()
    with torch.no_grad():
        m.fusion.gate.fill_(0.5)
        m.kin.gain[-1].bias.fill_(0.3)                # move the gain away from 1
    out = m(*_inputs())
    assert not torch.allclose(out["gain"], torch.ones(B, T, 5)) and float(out["F_eeg"].detach().abs().sum()) > 0
    torch.testing.assert_close(out["K"], out["K_base"] * out["gain"])
    torch.testing.assert_close(out["F"], torch.relu(out["K"] + out["F_eeg"]))


def test_untrained_gain_is_one_and_fused_equals_k():
    m = build_model(CONFIG_V2).eval()
    out = m(*_inputs())
    torch.testing.assert_close(out["gain"], torch.ones(B, T, 5))
    assert torch.equal(out["gate"], torch.zeros(5)) and torch.equal(out["F"], out["K"])
    assert float(out["F_eeg"].detach().abs().sum()) == 0.0


def test_time_is_mixed_only_by_recorded_attention_and_the_stem():
    m = build_model(CONFIG_V2).eval()
    layers = m.attention_layers()
    for name in layers["kin"]["temporal"] + layers["eeg"]["temporal"] + [layers["fusion"]["cross"]]:
        att = m.get_submodule(name)                   # force every time attention to look at its own token only
        att.forward = lambda q, context=None, bias=None, _f=att.forward: _f(q, context, 1e4 * torch.eye(q.shape[-2]))
    with torch.no_grad():
        m.fusion.gate.fill_(0.5)
    kin, hist, eeg = _inputs()
    base = m(kin, hist, eeg)
    kin2, eeg2 = kin.clone(), eeg.clone()
    kin2[:, 30] += 1.0; eeg2[:, 30] += 1.0
    alt = m(kin2, hist, eeg2)
    moved = lambda k: set((alt[k] - base[k]).abs().amax(dim=(0, 2)).gt(1e-6).nonzero().flatten().tolist())
    assert moved("E") == {30}
    assert 30 in moved("K") and moved("K") <= {28, 29, 30, 31, 32}         # the 5-tap stem
    assert 30 in moved("F") and moved("F") <= {28, 29, 30, 31, 32}


def test_stem_matrix_is_the_footprint_of_the_stem():
    m = build_model(CONFIG_V2).eval()
    s = m.kin.stem_matrix(40)
    assert s.shape == (40, 40) and s.device.type == "cpu"
    torch.testing.assert_close(s.sum(-1), torch.ones(40))
    assert (s[10, 8:13] > 0).all() and float(s[10, :8].sum()) == 0 and float(s[10, 13:].sum()) == 0
    assert (s[0, :3] > 0).all() and float(s[0, 3:].sum()) == 0             # the edge row keeps what exists


def test_k_ignores_the_eeg_input_and_the_eeg_weights():
    m = build_model(CONFIG_V2).eval()
    kin, hist, eeg = _inputs()
    k1 = m(kin, hist, eeg)["K"]
    k2 = m(kin, hist, torch.randn_like(eeg) * 5)["K"]
    assert torch.equal(k1, k2)
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        for n, p in m2.named_parameters():
            if stream_of(n) == "eeg":
                p.add_(torch.randn_like(p))
    assert torch.equal(k1, m2(kin, hist, eeg)["K"])


def test_e_ignores_kinematics():
    m = build_model(CONFIG_V2).eval()
    kin, hist, eeg = _inputs()
    assert torch.equal(m(kin, hist, eeg)["E"], m(torch.randn_like(kin), torch.rand_like(hist), eeg)["E"])


def test_gradient_isolation():
    m = build_model(CONFIG_V2).train()
    with torch.no_grad():
        m.fusion.gate.fill_(0.5)                      # open the gate so F really depends on the EEG path
    kin, hist, eeg = _inputs()
    out = m(kin, hist, eeg)
    (out["E"].sum() + out["F"].sum()).backward()
    for n, p in m.named_parameters():
        if stream_of(n) == "kin":
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, n
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in m.named_parameters() if n.startswith("fusion."))
    m.zero_grad()
    m(kin, hist, eeg)["K"].sum().backward()
    for n, p in m.named_parameters():
        if stream_of(n) == "eeg":
            assert p.grad is None or float(p.grad.abs().sum()) == 0.0, n


def test_electrode_dropout_only_in_training():
    m = build_model(CONFIG_V2)
    kin, hist, eeg = _inputs()
    m.eval(); a = m(kin, hist, eeg)["E"]; b = m(kin, hist, eeg)["E"]
    assert torch.equal(a, b)
    m.train(); torch.manual_seed(0); c = m(kin, hist, eeg)["E"]; torch.manual_seed(1); d = m(kin, hist, eeg)["E"]
    assert not torch.equal(c, d)


def test_finite_under_autocast_and_any_length():
    m = build_model(CONFIG_V2).eval()
    for t in (400, 37):
        kin, hist, eeg = torch.randn(1, t, 24), torch.rand(1, t, 8), torch.randn(1, t, 32, 4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = m(kin, hist, eeg)
        assert out["F"].shape == (1, t, 5) and all(torch.isfinite(out[k].float()).all() for k in ("K", "E", "F"))


def test_dropout_values_come_from_the_config():
    for change in ({}, {"dropout": 0.3, "muscle_dropout": 0.4}):          # the config as it is, then two other values
        cfg = copy.deepcopy(CONFIG_V2)
        cfg["model"].update(change)
        m = build_model(cfg)
        groups = {"muscle_dropout": [m.kin.head, m.eeg.head],              # the muscle graph of both streams
                  "dropout": [m.kin.enc, m.eeg.enc, m.eeg.graph, m.fusion]}  # time, electrode graph, cross-attention
        seen = 0
        for key, modules in groups.items():
            for mod in modules:
                ps = _dropouts(mod)
                assert ps and set(ps) == {cfg["model"][key]}, (key, ps)
                seen += len(ps)
        assert seen == len(_dropouts(m))                                   # no dropout layer outside these groups


def test_every_stacked_attention_layer_is_a_pre_norm_residual_block():
    m = build_model(CONFIG_V2)
    layers = m.attention_layers()
    stacked = (layers["kin"]["temporal"] + layers["kin"]["muscle"] + layers["eeg"]["electrode"]
               + layers["eeg"]["temporal"] + layers["eeg"]["muscle"])
    blocks = {n: mod for n, mod in m.named_modules() if isinstance(mod, AttentionBlock)}
    assert sorted(stacked) == sorted(f"{n}.attn" for n in blocks)          # these layers and no other block
    for name in stacked:
        block = blocks[name.rpartition(".")[0]]
        assert block.attn is m.get_submodule(name)
        with torch.no_grad():                                              # both sub-layers off: only the skip is left
            block.attn.out.weight.zero_(); block.attn.out.bias.zero_()
            if block.ffn is not None:
                block.ffn[-1].weight.zero_(); block.ffn[-1].bias.zero_()
        d = block.n1.normalized_shape[0]
        x = torch.randn(2, 9, d) if block.edge is None else torch.randn(2, 9, block.edge.shape[-1], d)
        assert torch.equal(block.eval()(x), x) and torch.equal(block.train()(x), x), name
        read = []                                                          # what each sub-layer gets: the normalised input
        hooks = [block.attn.register_forward_pre_hook(lambda mod, args: read.append(args[0]))]
        if block.ffn is not None:
            hooks.append(block.ffn.register_forward_pre_hook(lambda mod, args: read.append(args[0])))
        block.eval()(x)
        for h in hooks:
            h.remove()
        assert torch.equal(read[0], block.n1(x)) and not torch.equal(read[0], x), name
        assert block.ffn is None or torch.equal(read[1], block.n2(x)), name
    for name in (layers["eeg"]["pool"], layers["fusion"]["cross"]):       # the two layers without a skip
        assert not any(name.startswith(n + ".") for n in blocks), name


def _self_only(att):                                  # force an attention module to look at its own position only
    att.forward = lambda q, context=None, bias=None, _f=att.forward: _f(q, context, 1e4 * torch.eye(q.shape[-2]))


def test_fusion_has_no_skip():
    m = build_model(CONFIG_V2).eval()
    const = torch.randn(1, 1, CONFIG_V2["model"]["d_model"]).expand(B, T, -1)
    eeg_forward = m.eeg.forward
    m.eeg.forward = lambda x: (eeg_forward(x)[0], const)                   # the same EEG token at every time step
    with torch.no_grad():
        m.fusion.gate.fill_(0.5)
    kin, hist, eeg = _inputs()
    a = m(kin, hist, eeg)["F_eeg"]
    b = m(torch.randn_like(kin), torch.rand_like(hist), eeg)["F_eeg"]
    assert float(a.detach().abs().sum()) > 0                               # the queries only choose where to look,
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)                    # so other kinematics change nothing here


def test_history_gain_acts_on_one_time_step():
    torch.manual_seed(0)
    m = build_model(CONFIG_V2).eval()
    with torch.no_grad():
        m.kin.gain[-1].weight.normal_(0, 0.5)
    kin, hist, eeg = _inputs()
    hist2 = hist.clone(); hist2[:, 30] += 0.5
    change = (m(kin, hist2, eeg)["gain"] - m(kin, hist, eeg)["gain"]).abs().amax(dim=(0, 2))
    assert change.gt(1e-6).nonzero().flatten().tolist() == [30]


def test_history_gain_stays_between_a_half_and_two():
    torch.manual_seed(0)
    m = build_model(CONFIG_V2).eval()
    with torch.no_grad():
        m.kin.gain[-1].weight.normal_(0, 50.0)                             # far outside any trained range
    g = m(*_inputs())["gain"].detach()
    assert float(g.min()) >= 0.5 and float(g.max()) <= 2.0 and float(g.min()) < 0.51 and float(g.max()) > 1.99


def test_electrodes_are_mixed_only_by_recorded_attention():
    m = build_model(CONFIG_V2).eval()
    for name in m.attention_layers()["eeg"]["electrode"]:
        _self_only(m.get_submodule(name))
    nodes = []                                                             # what reaches the pooling: (B, T, 32, d)
    hook = m.eeg.pool.register_forward_pre_hook(lambda mod, args: nodes.append(args[0].detach()))
    kin, hist, eeg = _inputs()
    eeg2 = eeg.clone(); eeg2[:, :, 7] += 1.0
    m(kin, hist, eeg); m(kin, hist, eeg2)
    hook.remove()
    assert (nodes[1] - nodes[0]).abs().amax(dim=(0, 1, 3)).gt(1e-6).nonzero().flatten().tolist() == [7]


def test_electrode_dropout_zeroes_whole_electrodes_and_does_not_rescale():
    cfg = copy.deepcopy(CONFIG_V2)
    cfg["model"].update(dropout=0.0, muscle_dropout=0.0)                   # electrode dropout is the only random part
    m = build_model(cfg)
    fed = []                                                               # what the electrode embedding receives
    hook = m.eeg.embed.register_forward_pre_hook(lambda mod, args: fed.append(args[0].detach()))
    kin, hist, eeg = _inputs()
    m.train(); torch.manual_seed(0); c = m(kin, hist, eeg)["E"]; torch.manual_seed(1); d = m(kin, hist, eeg)["E"]
    m.eval(); m(kin, hist, eeg)
    hook.remove()
    assert not torch.equal(c, d) and not torch.equal(fed[0], fed[1]) and torch.equal(fed[2], eeg)
    for x in fed[:2]:
        kept = x.abs().amax(dim=(1, 3)) > 0                                # (B, electrodes): one draw per window
        assert kept.any() and not kept.all()
        assert torch.equal(x, eeg * kept[:, None, :, None])                # kept: unchanged; dropped: zero everywhere


def test_gate_output_is_a_copy():
    m = build_model(CONFIG_V2).eval()
    out = m(*_inputs())
    with torch.no_grad():
        m.fusion.gate.fill_(0.7)
    assert torch.equal(out["gate"], torch.zeros(5)) and not out["gate"].requires_grad
