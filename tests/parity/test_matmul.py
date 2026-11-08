
import numpy as np
import torch

from minitorch.tensor import Tensor
from minitorch.conftest import assert_close, assert_same_grad


def compare_matmul(shape_a, shape_b, seed=42):
    rng = np.random.default_rng(seed)

    a_data = rng.standard_normal(shape_a).astype(np.float32)
    b_data = rng.standard_normal(shape_b).astype(np.float32)

    a_custom = Tensor(a_data.copy(), requires_grad=True)
    b_custom = Tensor(b_data.copy(), requires_grad=True)

    a_torch = torch.tensor(a_data.copy(), requires_grad=True)
    b_torch = torch.tensor(b_data.copy(), requires_grad=True)

    y_custom = a_custom @ b_custom
    y_torch = a_torch @ b_torch

    assert_close(y_custom, y_torch)

    upstream = rng.standard_normal(y_torch.shape).astype(np.float32)

    (y_custom * Tensor(upstream)).sum().backward()
    (y_torch * torch.tensor(upstream)).sum().backward()

    assert_same_grad(a_custom, a_torch)
    assert_same_grad(b_custom, b_torch)


def test_matmul_2d():
    compare_matmul((3, 4), (4, 5))


def test_matmul_batched():
    compare_matmul((2, 3, 4), (2, 4, 5))


def test_matmul_broadcasted_batches():
    compare_matmul((2, 3, 4), (1, 4, 5))


def test_matmul_broadcast_multiple_dimensions():
    compare_matmul((2, 1, 3, 4), (1, 5, 4, 6))


def test_matmul_vector_matrix():
    compare_matmul((4,), (4, 5))


def test_matmul_matrix_vector():
    compare_matmul((3, 4), (4,))


def test_matmul_vector_vector():
    compare_matmul((4,), (4,))
