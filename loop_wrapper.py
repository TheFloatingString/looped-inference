"""Training-free looped transformer wrapper (arXiv 2605.23872), no Modal deps.

Patches the `forward` of individual decoder layers on a HF causal LM so a
contiguous window [a, b] is re-applied K times at inference. Weights are untouched;
`unpatch` restores the original behaviour. Because we wrap per-layer forwards, each
layer is always called with its own kwargs (masks, position embeddings), which keeps
Cohere2's alternating sliding/global attention correct.

Only valid with use_cache=False: re-applying a layer would otherwise write its KV
entries more than once per position.
"""
from dataclasses import dataclass

import torch

STRATEGIES = ("none", "naive", "euler", "rk")
MODES = ("block", "layer")


@dataclass(frozen=True)
class LoopConfig:
    a: int = 0
    b: int = 0
    K: int = 1
    mode: str = "block"      # block: (L_b..L_a)^K ; layer: L_b^K ∘ ... ∘ L_a^K
    strategy: str = "none"   # none | naive | euler | rk
    beta: float = 0.5        # anchor weight for rk (paper's Algorithm 3); 0 == euler

    def name(self):
        if self.strategy == "none":
            return "baseline"
        s = f"{self.strategy}-{self.mode}-K{self.K}-w{self.a}_{self.b}"
        return s + (f"-beta{self.beta}" if self.strategy == "rk" else "")


def integrate(g, x0, K, strategy, beta=0.5, y0=None):
    """Advance x0 through window operator g with K applications.

    y0 = g(x0) if already computed (saves one evaluation).
    naive: x <- g(x), K times (t=K endpoint; paper's negative control).
    euler: x <- x + (1/K)(g(x)-x), K times (damped sub-steps to the t=1 endpoint).
    rk:    beta*g(x0) + (1-beta)*euler  (paper's Algorithm 3 / Eq. 16).
    """
    if strategy == "none" or K <= 1:
        return y0 if y0 is not None else g(x0)
    if strategy == "naive":
        x = y0 if y0 is not None else g(x0)
        for _ in range(K - 1):
            x = g(x)
        return x
    x = x0
    anchor = None
    for k in range(K):
        y = y0 if (k == 0 and y0 is not None) else g(x)
        if k == 0:
            anchor = y
        x = x + (y - x) / K
    if strategy == "euler":
        return x
    if strategy == "rk":
        return beta * anchor + (1 - beta) * x
    raise ValueError(strategy)


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _rewrap(out, h):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


def get_layers(model):
    return model.model.layers


def patch(model, cfg: LoopConfig):
    """Install looping on model's decoder layers. Idempotent: unpatches first."""
    unpatch(model)
    if cfg.strategy == "none" or cfg.K <= 1:
        return model
    assert cfg.strategy in STRATEGIES and cfg.mode in MODES, cfg
    layers = get_layers(model)
    assert 0 <= cfg.a <= cfg.b < len(layers), (cfg, len(layers))
    orig = {i: layers[i].forward for i in range(cfg.a, cfg.b + 1)}
    state = {}

    def guard(kw):
        assert kw.get("past_key_values") is None and not kw.get("use_cache", False), \
            "looping requires use_cache=False"

    if cfg.mode == "layer":
        def make(i):
            def fwd(x, **kw):
                guard(kw)
                out0 = orig[i](x, **kw)
                g = lambda z: _hidden(orig[i](z, **kw))
                h = integrate(g, x, cfg.K, cfg.strategy, cfg.beta, y0=_hidden(out0))
                return _rewrap(out0, h)
            return fwd
    else:
        def make(i):
            def fwd(x, **kw):
                guard(kw)
                if i == cfg.a:
                    state.clear()
                    state["x0"] = x
                state[i] = kw
                out = orig[i](x, **kw)
                if i != cfg.b:
                    return out

                def g(z):
                    for j in range(cfg.a, cfg.b + 1):
                        z = _hidden(orig[j](z, **state[j]))
                    return z
                h = integrate(g, state["x0"], cfg.K, cfg.strategy, cfg.beta, y0=_hidden(out))
                return _rewrap(out, h)
            return fwd

    for i in range(cfg.a, cfg.b + 1):
        layers[i].forward = make(i)
    return model


def unpatch(model):
    for layer in get_layers(model):
        layer.__dict__.pop("forward", None)
    return model
