
from minitorch.conftest import (
    compare_unary_operation,
    compare_binary_operation,
)


def test_add():
    compare_binary_operation(
        lambda a, b: a + b,
        lambda a, b: a + b,
        (3, 4),
        (3, 4),
    )


def test_add_broadcasting():
    compare_binary_operation(
        lambda a, b: a + b,
        lambda a, b: a + b,
        (2, 3, 4),
        (4,),
    )


def test_subtract():
    compare_binary_operation(
        lambda a, b: a - b,
        lambda a, b: a - b,
        (3, 4),
        (3, 4),
    )


def test_subtract_broadcasting():
    compare_binary_operation(
        lambda a, b: a - b,
        lambda a, b: a - b,
        (2, 3, 4),
        (1, 3, 1),
    )


def test_multiply():
    compare_binary_operation(
        lambda a, b: a * b,
        lambda a, b: a * b,
        (3, 4),
        (3, 4),
    )


def test_multiply_broadcasting():
    compare_binary_operation(
        lambda a, b: a * b,
        lambda a, b: a * b,
        (2, 3, 4),
        (1, 3, 1),
    )


def test_division():
    compare_binary_operation(
        lambda a, b: a / b,
        lambda a, b: a / b,
        (3, 4),
        (3, 4),
    )


def test_division_broadcasting():
    compare_binary_operation(
        lambda a, b: a / b,
        lambda a, b: a / b,
        (2, 3, 4),
        (4,),
    )


def test_negation():
    compare_unary_operation(
        lambda x: -x,
        lambda x: -x,
        (3, 4),
    )


def test_power():
    compare_unary_operation(
        lambda x: x ** 2,
        lambda x: x ** 2,
        (3, 4),
    )


def test_scalar_power():
    compare_unary_operation(
        lambda x: x ** 3,
        lambda x: x ** 3,
        (3, 4),
    )


def test_scalar_multiplication():
    compare_unary_operation(
        lambda x: x * 3.5,
        lambda x: x * 3.5,
        (3, 4),
    )


def test_scalar_division():
    compare_unary_operation(
        lambda x: x / 2.5,
        lambda x: x / 2.5,
        (3, 4),
    )



if __name__ == "__main__":
    test_add()
    test_add_broadcasting()
    test_subtract()
    test_subtract_broadcasting()
    test_multiply()
    test_multiply_broadcasting()
    test_division()
    test_division_broadcasting()
    test_negation()
    test_power()
    test_scalar_power()
    test_scalar_multiplication()
    test_scalar_division()