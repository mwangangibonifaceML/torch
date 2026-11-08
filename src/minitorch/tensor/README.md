# `minitorch.tensor`

The autograd core of minitorch. `Tensor` wraps a NumPy array and builds a
dynamic computational graph (define-by-run, like PyTorch) by recording each
op's parents and a `_backward` closure. Calling `.backward()` on a scalar
output walks the graph in reverse topological order and accumulates
gradients into `.grad` on every tensor that requires one.

## Core class

```python
Tensor(data, requires_grad=False, dtype=np.float32, device=None, _parents=())
```

- `.data` — the underlying `np.ndarray`
- `.grad` — accumulated gradient (`None` until the tensor is first touched
  by a backward pass, or zero-initialized if `requires_grad=True`)
- `.shape`, `.size`, `.dtype` — thin wrappers over the NumPy array
- `._parents` — tuple of tensors this one was produced from (the graph edge)
- `._backward` — a no-op by default; each op overwrites it on the node it
  returns

## Supported operations

Elementwise: `__add__`, `__sub__`, `__mul__`, `__truediv__`, `__neg__`,
`__pow__`, `abs()`, plus the reflected forms (`__radd__`, `__rsub__`, …).
Reductions: `sum`, `mean`, `var`, `max`. Shape ops: `reshape`, `transpose`,
`__getitem__` (basic/advanced indexing with scatter-add in the backward
pass), `masked_fill`. Linear algebra: `matmul` / `@`, which handles the
0-D, 1-D and batched-2-D+ cases PyTorch handles, including gradient shape
promotion for 1-D operands. Construction helpers: `zeros`, `ones`, `randn`,
`rand`, `arange`, `constant`, `tril`, `stack`, `randint`, `from_numpy`.

Broadcasting is undone on the way back by `unbroadcast()`, which sums a
gradient down to a target shape by first collapsing extra leading dims and
then summing any dims that were originally size-1.

## Usage

```python
import numpy as np
from minitorch.tensor import Tensor

x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
y = (x * 2 + 1).sum()
y.backward()
print(x.grad)  # [2. 2. 2.]
```

## ⚠️ Known bugs

### 1. `__pow__` only supports a scalar exponent, and assumes `self.data > 0` implicitly for non-integer powers
Fine for the current usage (`** 0.5`, `** 2`), but `self.data ** other` with
a negative base and fractional exponent will silently produce `nan` with no
guard or message. Worth a `ValueError` or doc note if `var`/`std`-style
calls are ever applied to values that can go negative pre-sqrt (they can't
here because `var + eps` is always ≥ 0, so this is low-priority).



## Notes for future work
- No in-place ops (`+=` etc. on `Tensor` itself) — all ops return new nodes,
  which is correct for autograd safety but means long chains build large
  graphs; fine at this project's scale (character-level, small context).
- `detach()` returns a `Tensor` with `requires_grad=False`; `Parameter`
  (defined in `minitorch.nn`) overrides `detach()` to raise, which is a
  deliberate design choice documented there.