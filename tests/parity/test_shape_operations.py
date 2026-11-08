
from minitorch.conftest import compare_unary_operation


def test_reshape():
    compare_unary_operation(
        lambda x: x.reshape(2, 6),
        lambda x: x.reshape(2, 6),
        (3, 4),
    )


def test_transpose():
    compare_unary_operation(
        lambda x: x.transpose(0, 1),
        lambda x: x.transpose(0, 1),
        (3, 4),
    )


def test_transpose_batched():
    compare_unary_operation(
        lambda x: x.transpose(-1, -2),
        lambda x: x.transpose(-1, -2),
        (2, 3, 4),
    )


def test_transpose_negative_axes():
    compare_unary_operation(
        lambda x: x.transpose(-1, -2),
        lambda x: x.transpose(-1, -2),
        (2, 3, 4),
    )


def test_reshape_after_transpose():
    compare_unary_operation(
        lambda x: x.transpose(0, 1).reshape(4, 3),
        lambda x: x.transpose(0, 1).reshape(4, 3),
        (3, 4),
    )


def test_transpose_after_reshape():
    compare_unary_operation(
        lambda x: x.reshape(2, 6).transpose(0, 1),
        lambda x: x.reshape(2, 6).transpose(0, 1),
        (3, 4),
    )
