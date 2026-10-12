import pytest
import torch

from v2.attention import Attention, AttentionPool, recording
from v2.blocks import GraphEncoder, MuscleHead, TemporalEncoder


def test_attention_records_what_it_used_per_head():
    att = Attention(64, 4).eval()
    x = torch.randn(2, 50, 64)
    with recording(att) as tape:
        y = att(x)
    a = tape.attn[""]
    assert y.shape == x.shape and a.shape == (2, 4, 50, 50) and not a.requires_grad
    torch.testing.assert_close(a.sum(-1), torch.ones(2, 4, 50), atol=1e-5, rtol=0)
    assert att.tape is None                                               # detached from the tape afterwards


def test_tape_is_removed_after_an_exception():
    att = Attention(64, 4)
    with pytest.raises(RuntimeError):
        with recording(att):
            att(torch.randn(2, 5, 63))                                    # wrong width
    assert att.tape is None


def test_cross_attention_depends_on_the_context_and_takes_a_bias():
    c = Attention(64, 4).eval()
    q, ctx = torch.randn(2, 20, 64), torch.randn(2, 30, 64)
    with recording(c) as tape:
        y = c(q, ctx)
    assert y.shape == q.shape and tape.attn[""].shape == (2, 4, 20, 30)
    assert not torch.allclose(y, c(q, torch.randn(2, 30, 64)))
    bias = torch.zeros(20, 30); bias[:, 7] = 50.0                          # every query looks at source 7
    with recording(c) as tape:
        c(q, ctx, bias=bias)
    assert (tape.attn[""][..., 7] > 0.999).all()


def test_attention_pool_is_a_recorded_weighted_mean():
    pool = AttentionPool(32).eval()
    x = torch.randn(2, 7, 9, 32)
    with recording(pool) as tape:
        y = pool(x)
    a = tape.attn[""]
    assert y.shape == (2, 7, 32) and a.shape == (2, 7, 1, 1, 9)
    torch.testing.assert_close(y, (a[:, :, 0, 0, :, None] * x).sum(-2), atol=1e-6, rtol=0)


def test_recording_does_not_change_the_output_and_can_keep_gradients():
    enc = TemporalEncoder(64, 3, 4, 256, 0.0).eval()
    x = torch.randn(2, 50, 64)
    plain = enc(x)
    with recording(enc, grad=True) as tape:
        y = enc(x)
    assert torch.equal(plain, y)
    names = [f"layers.{i}.attn" for i in range(3)]
    assert list(tape.attn) == names and all(tape.attn[n].shape == (2, 4, 50, 50) for n in names)
    target = y[0, 10, 0]                      # one component: the sum over a LayerNorm output is constant
    grads = torch.autograd.grad(target, [tape.attn[n] for n in names])
    last = grads[-1]
    assert float(last[0, :, 10].abs().sum()) > 0                          # the last layer feeds token 10 through row 10
    assert float(last[0, :, 11].abs().sum()) == 0 and float(last[1].abs().sum()) == 0
    assert float(grads[0][0].abs().sum()) > 0                             # earlier layers reach it through any row
    back = torch.flip(enc(torch.flip(x, [1])), [1])                       # reversed in, reversed back out
    assert not torch.allclose(plain, back, atol=1e-4)                     # so the order of the tokens matters


def test_graph_encoder_prior_and_shapes():
    prior = torch.zeros(5, 5); prior[0, 1] = 8.0                          # node 0 strongly prefers source 1
    g = GraphEncoder(5, 32, 2, 4, prior=prior).eval()
    x = torch.randn(2, 7, 5, 32)
    with recording(g) as tape:
        y = g(x)
    a = tape.attn["layers.0.attn"]
    assert y.shape == x.shape and a.shape == (2, 7, 4, 5, 5) and list(tape.attn) == ["layers.0.attn", "layers.1.attn"]
    torch.testing.assert_close(a.sum(-1), torch.ones(2, 7, 4, 5), atol=1e-5, rtol=0)
    assert (a[..., 0, 1] > 0.9).all()
    assert "layers.0.prior" in dict(g.named_buffers()) and "layers.0.edge" in dict(g.named_parameters())


def test_muscle_head_is_non_negative_and_starts_small():
    h = MuscleHead(64).eval()
    with recording(h) as tape:
        out = h(torch.randn(2, 30, 64))
    assert out.shape == (2, 30, 5) and (out >= 0).all() and out.mean() < 0.5
    assert [tuple(tape.attn[n].shape) for n in ("graph.layers.0.attn", "graph.layers.1.attn")] == [(2, 30, 4, 5, 5)] * 2


def test_blocks_are_finite_under_autocast():
    enc, h = TemporalEncoder(64, 1, 4, 256, 0.0), MuscleHead(64)
    with torch.autocast("cpu", dtype=torch.bfloat16), recording(enc) as tape:
        out = h(enc(torch.randn(2, 40, 64)))
    assert torch.isfinite(out.float()).all() and tape.attn["layers.0.attn"].dtype == torch.float32


def test_attention_is_recorded_before_dropout():
    att = Attention(64, 4, dropout=0.5).train()
    x = torch.randn(2, 10, 64)
    with recording(att) as first:
        y1 = att(x)
    with recording(att) as second:
        y2 = att(x)
    a = first.attn[""]
    torch.testing.assert_close(a.sum(-1), torch.ones(2, 4, 10), atol=1e-5, rtol=0)
    assert (a > 0).all()                                                  # dropout would leave zeros
    assert torch.equal(a, second.attn[""]) and not torch.equal(y1, y2)    # the same map, two different masks


def test_one_recording_covers_one_forward_pass():
    att = Attention(64, 4).eval()
    x = torch.randn(2, 5, 64)
    with pytest.raises(RuntimeError, match="ran twice inside one recording"):
        with recording(att, grad=True):
            att(x)
            att(x)                                                        # would replace the first entry
    assert att.tape is None
    with recording(att) as tape:                                          # a new recording starts empty
        att(x)
    assert list(tape.attn) == [""]


def test_recording_can_be_nested():
    enc = TemporalEncoder(64, 2, 4, 256, 0.0).eval()
    x = torch.randn(2, 6, 64)
    first = enc.layers[0].attn
    with recording(enc) as outer:
        with recording(enc.layers[0]) as inner:
            enc.layers[0](x)                                              # the innermost recording gets it
        assert list(inner.attn) == ["attn"] and not outer.attn
        assert first.tape is outer and first.tape_name == "layers.0.attn"
        with pytest.raises(RuntimeError):
            with recording(enc):
                enc(torch.randn(2, 6, 63))                                # wrong width
        assert first.tape is outer and first.tape_name == "layers.0.attn"   # restored after an exception too
        y = enc(x)
    assert list(outer.attn) == ["layers.0.attn", "layers.1.attn"] and torch.equal(y, enc(x))
    assert all(layer.attn.tape is None for layer in enc.layers)


def test_every_graph_layer_owns_its_prior_and_checks_its_shape():
    prior = torch.zeros(5, 5); prior[0, 1] = 8.0
    g = GraphEncoder(5, 32, 2, 4, prior=prior)
    p0, p1 = g.layers[0].prior, g.layers[1].prior
    assert len({p0.data_ptr(), p1.data_ptr(), prior.data_ptr()}) == 3
    with torch.no_grad():
        p0.zero_()                                                        # changing one layer's prior ...
    assert float(p1[0, 1]) == 8.0 and float(prior[0, 1]) == 8.0           # ... leaves the others alone
    assert [n for n, _ in g.named_buffers()] == ["layers.0.prior", "layers.1.prior"]
    from_numpy = GraphEncoder(5, 32, 1, 4, prior=prior.numpy()).layers[0].prior
    assert from_numpy.data_ptr() != prior.data_ptr() and torch.equal(from_numpy, prior)
    for wrong in (torch.zeros(5), torch.zeros(4, 4), torch.zeros(1, 5, 5)):
        with pytest.raises(ValueError, match="prior has shape"):
            GraphEncoder(5, 32, 2, 4, prior=wrong)
