from collections.abc import Callable

import numpy as np

from minitorch.tensor import Tensor


def gradcheck(fn: Callable, shape: tuple, seed: int = 42, eps: float = 1e-2):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(shape).astype(np.float32)
    x = Tensor(x0.copy(), requires_grad=True)
    out = fn(x)
    wei = np.array(rng.standard_normal(out.shape), dtype=np.float32)
    (out * Tensor(wei)).sum().backward()
    numeric = np.zeros(shape)
    for idx in np.ndindex(*shape):
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        fp = (fn(Tensor(xp).data.astype(np.float64)) * wei).sum()
        fm = (fn(Tensor(xm).data.astype(np.float64)) * wei).sum()
        numeric[idx] = (fp - fm) / (2 * eps)
    return bool(np.allclose(x.grad, numeric, rtol=2e-2, atol=2e-3))


def check_max_ties():
    x = Tensor(np.array([[1.0, 3.0, 3.0]], dtype=np.float32), requires_grad=True)
    x.max(axis=1).sum().backward()
    return np.allclose(x.grad, [[0.0, 0.5, 0.5]])  # split evenly between ties


def check_masked_fill():
    rng = np.random.default_rng(0)
    x = Tensor(rng.standard_normal((2, 3, 3)).astype(np.float32), requires_grad=True)
    w = rng.standard_normal((2, 3, 3)).astype(np.float32)
    out = Tensor.masked_fill(x, filler=-1e9)
    (out * Tensor(w)).sum().backward()
    lower = np.tril(np.ones((3, 3), dtype=bool))  # kept positions
    expected = np.where(lower, w, 0.0)  # zero grad where filled
    forward_ok = np.all(out.data[:, ~lower] == -1e9) and np.allclose(
        out.data[:, lower], x.data[:, lower]
    )
    return bool(forward_ok and np.allclose(x.grad, expected))


def run(name, f):
    try:
        ok = f()
    except Exception as e:  # a crash is also a failure
        ok = False
        name += f"  ({type(e).__name__})"
    print(f"{'PASS' if ok else 'FAIL'}  {name}")


CASES = [  # (name, fn, input shape)
    ("var()", lambda x: x.var(), (3, 4)),
    ("var(axis=0)", lambda x: x.var(axis=0), (3, 4)),
    ("var(axis=-1, keepdims)", lambda x: x.var(axis=-1, keepdims=True), (3, 4)),
    ("var(ddof=0)", lambda x: x.var(axis=1, ddof=0), (3, 4)),
    ("max()", lambda x: x.max(), (3, 4)),
    ("max(axis=0)", lambda x: x.max(axis=0), (3, 4)),
    ("max(axis=-1, keepdims)", lambda x: x.max(axis=-1, keepdims=True), (3, 4)),
    ("max(axis=(0,2))", lambda x: x.max(axis=(0, 2)), (2, 3, 4)),
    ("sum(axis=(-1,-2))", lambda x: x.sum(axis=(-1, -2)), (2, 3, 4)),
    ("sum(axis=1)", lambda x: x.sum(axis=1), (2, 3, 4)),
    ("x[:, 1:3]", lambda x: x[:, 1:3], (3, 4)),
    ("x[1, 2]", lambda x: x[1, 2], (3, 4)),
    ("x[[0, 0, 2]] (repeats)", lambda x: x[[0, 0, 2]], (3, 4)),
]


if __name__ == "__main__":
    for name, fn, shape in CASES:
        run(name, lambda fn=fn, shape=shape: gradcheck(fn, shape))
    run("max() ties split evenly", check_max_ties)
    run("masked_fill forward + grad", check_masked_fill)
