
import numpy as np
import torch

from minitorch.tensor import Tensor
from minitorch.conftest import assert_close, assert_same_grad


def test_masked_fill_forward_backward():
    rng = np.random.default_rng(42)

    data = rng.standard_normal((3, 4)).astype(np.float32)
    mask = rng.random((3, 4)) > 0.5

    x_custom = Tensor(data.copy(), requires_grad=True)
    x_torch = torch.tensor(data.copy(), requires_grad=True)

    y_custom = x_custom.masked_fill(mask, -1e9)
    y_torch = x_torch.masked_fill(
        torch.tensor(mask), -1e9
    )

    assert_close(y_custom, y_torch)

    upstream = rng.standard_normal((3, 4)).astype(np.float32)

    (y_custom * Tensor(upstream)).sum().backward()
    (y_torch * torch.tensor(upstream)).sum().backward()

    assert_same_grad(x_custom, x_torch)


def test_masked_fill_broadcast_mask():
    rng = np.random.default_rng(42)

    data = rng.standard_normal((2, 3, 4)).astype(np.float32)
    mask = np.array(
        [[False, True, False, True]],
        dtype=bool,
    )

    x_custom = Tensor(data.copy(), requires_grad=True)
    x_torch = torch.tensor(data.copy(), requires_grad=True)

    y_custom = x_custom.masked_fill(mask, -1e9)
    y_torch = x_torch.masked_fill(torch.tensor(mask), -1e9)

    assert_close(y_custom, y_torch)


if __name__ == "__main__":
    test_masked_fill_forward_backward()