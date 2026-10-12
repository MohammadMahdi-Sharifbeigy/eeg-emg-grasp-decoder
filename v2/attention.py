"""Attention with a recorder.

Every place where KG-GT v2 moves information between positions (time steps, electrodes, muscles,
streams) is one of the two modules in this file, so one Tape sees all of it. Convention everywhere:
row = receiving position (query), column = source position.
"""
import math
from contextlib import contextmanager

import torch
import torch.nn as nn


class Tape:
    """What each attention module used in one forward pass, keyed by the module's name in the model.

    attn[name] is (*lead, heads, n_query, n_source), after the softmax and before dropout.
    With grad=True the stored tensors are the ones used in the forward pass (still attached to the
    graph), so d(output) / d(attention) can be taken; otherwise they are detached float32 copies.
    Under CPU bfloat16 autocast the recorded rows sum to 1 only to about 1e-3, and with grad=True the
    entry keeps the dtype of the forward pass.
    One tape covers one forward pass: a module that runs a second time raises instead of replacing
    its entry.
    """

    def __init__(self, grad: bool = False):
        self.grad = bool(grad)
        self.attn: dict[str, torch.Tensor] = {}

    def put(self, name: str, a: torch.Tensor) -> None:
        if name in self.attn:
            raise RuntimeError(f"attention module {name!r} ran twice inside one recording; "
                               "one recording covers one forward pass")
        self.attn[name] = a if self.grad else a.detach().float()


def _heads(t: torch.Tensor, h: int) -> torch.Tensor:
    return t.unflatten(-1, (h, -1)).transpose(-3, -2)                 # (..., n, d) -> (..., h, n, d / h)


class Attention(nn.Module):
    """Multi-head attention over the node axis of (..., n, d); self-attention when context is None.

    A bias must broadcast to (heads, n_query, n_source); a (batch, n_query, n_source) bias is not supported.
    """

    def __init__(self, d: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d % n_heads:
            raise ValueError(f"width {d} is not a multiple of {n_heads} heads")
        self.h = n_heads
        self.q, self.k, self.v, self.out = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.tape, self.tape_name = None, ""

    def forward(self, query, context=None, bias=None):
        context = query if context is None else context
        q, k, v = _heads(self.q(query), self.h), _heads(self.k(context), self.h), _heads(self.v(context), self.h)
        s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if bias is not None:
            s = s + bias.to(s.dtype)
        a = s.softmax(dim=-1)                                          # (..., h, n_query, n_source)
        if self.tape is not None:
            self.tape.put(self.tape_name, a)
        return self.out((self.drop(a) @ v).transpose(-3, -2).flatten(-2))


class AttentionPool(nn.Module):
    """Weighted mean over the node axis, one learned score per node: (..., n, d) -> (..., d)."""

    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Linear(d, 1)
        self.tape, self.tape_name = None, ""

    def forward(self, x):
        a = self.score(x).transpose(-1, -2).softmax(dim=-1).unsqueeze(-3)      # (..., 1, 1, n)
        if self.tape is not None:
            self.tape.put(self.tape_name, a)
        return (a.squeeze(-3) @ x).squeeze(-2)


RECORDED = (Attention, AttentionPool)


@contextmanager
def recording(model: nn.Module, grad: bool = False):
    """`with recording(model) as tape:` — every attention module inside `model` writes to `tape`.

    Recordings can be nested: a module writes to the innermost open recording that covers it, and on
    exit it goes back to the tape and the name it had before (none outside every recording).
    """
    tape = Tape(grad)
    mods = [(name, m) for name, m in model.named_modules() if isinstance(m, RECORDED)]
    before = [(m.tape, m.tape_name) for _, m in mods]
    try:
        for name, m in mods:
            m.tape, m.tape_name = tape, name
        yield tape
    finally:
        for (_, m), (old_tape, old_name) in zip(mods, before):
            m.tape, m.tape_name = old_tape, old_name
