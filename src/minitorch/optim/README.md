# `minitorch.optim`

Optimizers: `SGD` (with momentum + weight decay), `Adam`, `AdamW`. All share
the `Optimizer` base class, which stores `self.params` (a list of live
`Parameter` objects from `Module.parameters()`), `self.lr`, and a
`step_count`.

```python
Optimizer(params, lr)
  .step()        # abstract, each subclass implements its own update rule
  .zero_grad()   # calls param.zero_grad() on every tracked parameter
  ._extract_grad_data(gradient)  # unwraps a Tensor to its .data, or passes a raw ndarray through
```

## `SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)`
Classic (optionally momentum) SGD with L2-style weight decay added directly
into the gradient before the update. Also exposes
`get_momentum_state()`/`set_momentum_state()` for checkpointing.

## `Adam` / `AdamW`
Standard bias-corrected first/second moment estimates:
```
m_t = β1 m_{t-1} + (1-β1) g_t         v_t = β2 v_{t-1} + (1-β2) g_t²
m̂_t = m_t / (1-β1^t)                   v̂_t = v_t / (1-β2^t)
θ_t = θ_{t-1} − lr · m̂_t / (√v̂_t + ε)
```
`AdamW` differs from `Adam` only in *where* weight decay is applied:
decoupled, as a multiplicative shrink of the parameter itself
(`param.data *= (1 - lr * weight_decay)`) rather than folded into the
gradient — this is the correct AdamW formulation (Loshchilov & Hutter 2019).


## Usage
```python
from minitorch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```