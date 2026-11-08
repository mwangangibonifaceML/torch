
from minitorch.conftest import compare_unary_operation
from minitorch.tensor import Tensor
import numpy as np
import torch


def test_sum_all():
    compare_unary_operation(
        lambda x: x.sum(),
        lambda x: x.sum(),
        (3, 4),
    )


def test_sum_axis():
    compare_unary_operation(
        lambda x: x.sum(axis=1),
        lambda x: x.sum(dim=1),
        (3, 4),
    )


def test_sum_keepdims():
    compare_unary_operation(
        lambda x: x.sum(axis=1, keepdims=True),
        lambda x: x.sum(dim=1, keepdim=True),
        (3, 4),
    )


def test_mean_all():
    compare_unary_operation(
        lambda x: x.mean(),
        lambda x: x.mean(),
        (3, 4),
    )


def test_mean_axis():
    compare_unary_operation(
        lambda x: x.mean(axis=1),
        lambda x: x.mean(dim=1),
        (3, 4),
    )


def test_mean_keepdims():
    compare_unary_operation(
        lambda x: x.mean(axis=1, keepdims=True),
        lambda x: x.mean(dim=1, keepdim=True),
        (3, 4),
    )


def test_mean_negative_axis():
    compare_unary_operation(
        lambda x: x.mean(axis=-1),
        lambda x: x.mean(dim=-1),
        (2, 3, 4),
    )


# def test_exp():
#     compare_unary_operation(
#         lambda x: x.exp(),
#         lambda x: x.exp(),
#         (3, 4),
#     )


# def test_log():
#     x = np.exp(
#         np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
#     )

#     from minitorch.conftest import assert_close

#     custom = Tensor(x.copy())
#     torch_tensor = torch.tensor(x)

#     assert_close(custom.log(), torch_tensor.log())


def test_sqrt():
    x = np.abs(
        np.random.default_rng(42).standard_normal((3, 4)).astype(np.float32)
    ) + 0.1

    from minitorch.conftest import assert_close

    custom = Tensor(x.copy())
    torch_tensor = torch.tensor(x)

    assert_close(custom ** 0.5, torch_tensor.sqrt())
