# `minitorch.activations`

Elementwise activation layers, each implemented as a `Module` with a
hand-written analytic backward pass (no reliance on `Tensor`'s own
autograd for these — they compute forward/backward directly against
`.data` for speed and numerical control).

| Class | Formula | Notes |
|---|---|---|
| `Sigmoid` | `σ(x) = 1/(1+e^-x)` | numerically stable (branches on sign of `x`) |
| `ReLU` | `max(0, x)` | |
| `Tanh` | `tanh(x)` | |
| `GELU(approximate="none"\|"tanh"\|"sigmoid")` | see below | three interchangeable approximations |
| `Softmax(dim=-1)` | `eˣⁱ / Σeˣʲ` | max-subtraction for stability |

## `GELU`
Supports three forms via `_gelu_np`:
- `"none"` — exact GELU using an `erf` polynomial approximation (`_erf`,
  Abramowitz & Stegun 7.1.26, accurate to ~1.5e-7)
- `"tanh"` — the standard tanh approximation used in GPT-2/BERT
- `"sigmoid"` — `x · σ(1.702x)`, the cheapest approximation; **this is the
  variant `minitorch.transformer.MLP` actually uses**

All three return `(value, local_gradient)` from `_gelu_np` so the backward
pass is a single multiply against the upstream gradient — no graph
recursion needed for this op.

## Usage
```python
from minitorch.activations import GELU, Softmax

gelu = GELU(approximate="sigmoid")
h = gelu(x)

probs = Softmax(dim=-1)(logits)
```

### Classes that subclass `Module` but have no parameters
`Sigmoid`, `ReLU`, `Tanh`, `GELU`, `Softmax` all inherit from `Module`
purely to get the `__call__`/`training` plumbing, even though none of them
have learnable weights. This is consistent with the rest of the framework
and not a bug — just note that `.parameters()` on any of these will
correctly return `[]`, and `Module.modules()` will still walk into them
when they're nested inside a container like `MLP`.

### No numerical guard against extreme inputs in `Tanh`/`ReLU`
`np.tanh` and `np.maximum` are already numerically stable for all finite
float32 inputs, so this is fine — flagged only because `Sigmoid` and the
GELU path *do* go out of their way to be stable (max-subtraction,
sign-branching), so it's worth knowing which activations need that
treatment and which don't if you add new ones later (e.g. a custom
Swish/SiLU should follow the `Sigmoid` stability pattern).