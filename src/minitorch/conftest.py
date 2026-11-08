
import numpy as np
import pytest
import torch

from minitorch.tensor import Tensor


DTYPE = np.float32

# Tight enough for float32 CPU operations, but allow small differences
# caused by floating-point operation ordering.
ATOL = 1e-5
RTOL = 1e-4


def as_numpy(value):
    """Extract NumPy data from either a custom Tensor or an array."""
    if isinstance(value, Tensor):
        value = value.data

    return np.asarray(value)


def assert_close(actual, expected, atol=ATOL, rtol=RTOL):
    """Compare custom-engine output with PyTorch output."""
    np.testing.assert_allclose(
        as_numpy(actual),
        expected.detach().cpu().numpy()
        if isinstance(expected, torch.Tensor)
        else np.asarray(expected),
        atol=atol,
        rtol=rtol,
        err_msg="Numerical mismatch between babytorch and PyTorch",
    )


def assert_same_shape(actual, expected):
    assert tuple(actual.shape) == tuple(expected.shape)


def assert_same_grad(custom_tensor, torch_tensor):
    """Compare gradients accumulated by both engines."""
    assert custom_tensor.grad is not None
    assert torch_tensor.grad is not None

    assert_close(custom_tensor.grad, torch_tensor.grad)


def random_array(shape, seed=42, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) * scale).astype(DTYPE)


def compare_unary_operation(
    custom_operation,
    torch_operation,
    shape,
    seed=42,
):
    """
    Compare forward and backward behavior for a unary operation.

    Uses a random upstream gradient instead of simply calling
    backward on the output sum.
    """
    x_data = random_array(shape, seed=seed)

    x_custom = Tensor(x_data.copy(), requires_grad=True)
    x_torch = torch.tensor(
        x_data.copy(),
        dtype=torch.float32,
        requires_grad=True,
    )

    y_custom = custom_operation(x_custom)
    y_torch = torch_operation(x_torch)

    assert_same_shape(y_custom, y_torch)
    assert_close(y_custom, y_torch)

    # Same upstream gradient for both engines.
    upstream = random_array(y_torch.shape, seed=seed + 1)

    upstream_custom = Tensor(upstream)
    upstream_torch = torch.tensor(upstream, dtype=torch.float32)

    # Scalarize the outputs using the same upstream gradient.
    (y_custom * upstream_custom).sum().backward()
    (y_torch * upstream_torch).sum().backward()

    assert_same_grad(x_custom, x_torch)


def compare_binary_operation(
    custom_operation,
    torch_operation,
    shape_a,
    shape_b,
    seed=42,
):
    """Compare forward and backward behavior for binary operations."""
    a_data = random_array(shape_a, seed=seed)
    b_data = random_array(shape_b, seed=seed + 1)

    a_custom = Tensor(a_data.copy(), requires_grad=True)
    b_custom = Tensor(b_data.copy(), requires_grad=True)

    a_torch = torch.tensor(a_data.copy(), requires_grad=True)
    b_torch = torch.tensor(b_data.copy(), requires_grad=True)

    y_custom = custom_operation(a_custom, b_custom)
    y_torch = torch_operation(a_torch, b_torch)

    assert_same_shape(y_custom, y_torch)
    assert_close(y_custom, y_torch)

    upstream = random_array(y_torch.shape, seed=seed + 2)

    (y_custom * Tensor(upstream)).sum().backward()
    (y_torch * torch.tensor(upstream)).sum().backward()

    assert_same_grad(a_custom, a_torch)
    assert_same_grad(b_custom, b_torch)