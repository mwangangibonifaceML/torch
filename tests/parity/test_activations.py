
import numpy as np
import torch

from minitorch.tensor import Tensor
from minitorch.activations import (
    ReLU, Sigmoid, Tanh, GELU, Softmax
)
from minitorch.conftest import compare_unary_operation, assert_close


def test_relu():
    compare_unary_operation(
        ReLU(),
        torch.relu,
        (3, 4),
    )


def test_Sigmoid():
    compare_unary_operation(
        Sigmoid(),
        torch.sigmoid,
        (3, 4),
    )


def test_tanh():
    compare_unary_operation(
        Tanh(),
        torch.tanh,
        (3, 4),
    )


def test_gelu():
    compare_unary_operation(
        GELU(),
        lambda x: torch.nn.functional.gelu(x),
        (3, 4),
    )


def test_softmax():
    compare_unary_operation(
        lambda x: Softmax(dim=-1)(x),
        lambda x: torch.softmax(x, dim=-1),
        (2, 3, 4),
    )


def test_softmax_axis_1():
    compare_unary_operation(
        lambda x: Softmax(dim=1)(x),
        lambda x: torch.softmax(x, dim=1),
        (2, 3, 4),
    )


def test_softmax_sums_to_one():
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
    probs = Softmax(dim=-1)(x)

    np.testing.assert_allclose(
        probs.data.sum(axis=-1),
        np.ones((2, 3)),
        atol=1e-6,
    )


if __name__ == "__main__":
    test_relu()