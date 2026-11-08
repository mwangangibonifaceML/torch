# `minitorch.nn`

Base `Module`/`Parameter` abstractions and the standard layer zoo:
`Linear`, `Dropout`, `Sequential`, `Residual`, `LayerNormalization`,
`BatchNormalization`, `Flatten`.

## `Parameter`
A `Tensor` subclass that always has `requires_grad=True`. `detach()` is
overridden to raise — a parameter is never meant to leave the graph, since
doing so silently would make an optimizer update a copy instead of the live
weight (see the `minitorch.optim` README for exactly this class of bug).

## `Module`
Base class every layer inherits from.

- `forward(...)` — must be overridden; `__call__` dispatches to it.
- `modules()` — depth-first walk of `self` + every descendant `Module`,
  found by introspecting `__dict__` (including values nested in lists,
  tuples and dicts, via the `_walk` helper).
- `parameters()` — collects every `Parameter` reachable from `modules()`,
  de-duplicated by `id()`. **This returns the live objects**, not copies —
  important, see below.
- `train(mode=True)` / `eval(mode=False)` — recursively flips `.training`
  on every submodule (used by `Dropout`, `BatchNormalization`).
- `zero_grad()` — zeroes every parameter's gradient.

## Layers

| Layer | Notes |
|---|---|
| `Linear(in, out, bias=False)` | Xavier-uniform init, `y = xW (+ b)` |
| `Dropout(p)` | inverted dropout (scales kept units by `1/(1-p)`); no-op in eval mode |
| `Sequential(*layers)` | chains modules, forwards `*args/**kwargs` through (used here to thread `mask` into `TransformerBlock`s) |
| `Residual(fn)` | `y = x + fn(x)` |
| `LayerNormalization(dim)` | normalizes the last axis, learnable `gamma`/`beta` |
| `BatchNormalization(dim)` | normalizes axis 0, tracks running mean/var for eval |
| `Flatten()` | `(N, d1, d2, …) → (N, d1*d2*…)` |

## Usage

```python
from minitorch.nn import Sequential, Linear, LayerNormalization, Dropout

net = Sequential(
    Linear(64, 256, bias=True),
    LayerNormalization(256),
    Dropout(0.1),
    Linear(256, 64, bias=True),
)
params = net.parameters()   # live references, safe to hand to an optimizer
```
