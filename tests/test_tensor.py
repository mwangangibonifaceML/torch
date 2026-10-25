from __future__ import annotations

import numpy as np

from minitorch.tensor import Tensor


class TestTensorCreation:
    "Test Tensor Creation and Initialization"

    def test_tensor_creation_from_integers(self):
        t1 = Tensor(10)

        assert t1.shape == (), (
            f"Input tensor is wrong "
            f"Input is an integer "
            f"expected shape () "
            f"received {t1.shape}"
        )

        assert t1.data == np.array(10)

    def test_tensor_creation_from_floats(self):
        t1 = Tensor(10.8)

        assert t1.shape == (), (
            f"Input tensor is wrong "
            f"Input is an integer "
            f"expected shape () "
            f"received {t1.shape}"
        )

        assert t1.data == np.array(10.8, dtype=np.float32)

    def test_tensor_creation_from_list(self):
        "Test tensor creation from python lists."

        t1 = Tensor([1, 2, 3, 4])
        assert t1.shape == (4,), (
            f"1D input tensor is wrong "
            f"Input has elements [1,2,3,4] "
            f"expected shape is (4,) "
            f"received shape {t1.shape}"
        )

        t2 = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert t2.shape == (2, 4), (
            f"2D input tensor has wrong shape "
            f"Input has elements [[1,2,3,4], [5,6,7,8]] "
            f"expected shape is (2,4) "
            f"received shape {t2.shape}"
        )

    def test_tensor_creation_from_tuple(self):
        "Test tensor creation from python tuples."

        t1 = Tensor((1, 2, 3, 4))
        assert t1.shape == (4,), (
            f"1D input tensor has wrong shape "
            f"Input has elements (1,2,3,4) "
            f"expected shape is (4,) "
            f"received shape {t1.shape}"
        )

        t2 = Tensor(((1, 2, 3, 4), (5, 6, 7, 8)))
        assert t2.shape == (2, 4), (
            f"1D input tensor has wrong shape "
            f"Input has elements ((1,2,3,4), (5,6,7,8)) "
            f"expected shape is (2,4) "
            f"received shape {t2.shape}"
        )

    def test_tensor_creation_from_numpy_arrays(self):
        "Test tensor creation from numpy arrays."

        t1 = Tensor(np.array((1, 2, 3, 4)))
        assert t1.shape == (4,), (
            f"1D input tensor is wrong "
            f"Input has elements (1,2,3,4) "
            f"expected shape is (4,) "
            f"received shape {t1.shape}"
        )
        assert t1.dtype == np.float32, (
            "Wrong dtype from the input tensor Expected np.float32."
        )

        t2 = Tensor(np.array(((1, 2, 3, 4), (5, 6, 7, 8))))
        assert t2.shape == (2, 4), (
            f"2D input tensor is wrong "
            f"Input has elements (1,2,3,4), (5,6,7,8) "
            f"expected shape is (2,4) "
            f"received shape {t2.shape}"
        )

        assert t2.dtype == np.float32, (
            "Wrong dtype from the input tensor Expected np.float32."
        )


class TestTensorOperations:
    """
    Test Tensor Operations like additions, multiplication, addition
    and others."""

    def test_element_wise_addition(self):
        "Test tensor addition."

        t1 = Tensor([1, 2, 3, 4])
        t2 = Tensor([5, 6, 7, 8])

        expected = Tensor([6, 8, 10, 12])
        result = t1 + t2

        assert isinstance(result, Tensor), (
            f"Addition should return a tensor "
            f"This maintains the computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(expected.data, result.data), (
            f"Element-wise addition failed "
            f"{t1.data} + {t2.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_element_wise_multiplication(self):
        "Test element wise multiplication."

        t1 = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        t2 = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        expected = Tensor([[1, 4, 9, 16], [25, 36, 49, 64]])
        result = t1 * t2

        assert isinstance(result, Tensor), (
            f"Multiplication should return a tensor "
            f"This maintains the computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(result.data, expected.data), (
            f"Element-wise mutliplication failed "
            f"{t1.data} + {t2.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_scalar_vector_multiplication(self):
        scalar = Tensor(5.0)
        vector = Tensor([1, 2, 3])

        # Scalar * vector should raise ValueError
        expected = Tensor([5.0, 10.0, 15.0])
        result = scalar * vector

        assert isinstance(result, Tensor), (
            f"Scalar and tensor should result a tensor "
            f"helps maintain the computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(result.data, expected.data), (
            f"Scalar and tensor multiplication failed "
            f"{scalar.data} @ {vector.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_vector_scalar_multiplication(self):
        scalar = Tensor(5.0)
        vector = Tensor([1, 2, 3])

        # Scalar * vector should raise ValueError
        expected = Tensor([5.0, 10.0, 15.0])
        result = vector * scalar

        assert isinstance(result, Tensor), (
            f"Scalar and tensor should result a tensor "
            f"helps maintain the computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(result.data, expected.data), (
            f"Scalar and tensor multiplication failed "
            f"{vector.data} * {scalar.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_matrix_multiplication(self):
        "Test matrix multiplication. The core operation in ML."
        t1 = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        t2 = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])

        expected = Tensor([[50, 60], [114, 140]])
        result = t1 @ t2

        assert isinstance(result, Tensor), (
            f"Matrix multiplication should return a tensor "
            f"This helps maintain computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(result.data, expected.data), (
            f"Matrix multiplication failed "
            f"{t1.data} @ {t2.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_scalar_vector_multiplication(self):

        scalar = Tensor(5.0)
        vector = Tensor([1, 2, 3])

        expected = Tensor([5.0, 10.0, 15.0])
        result = scalar @ vector

        assert isinstance(result, Tensor), (
            f"Scalar and vector multiplication should produce a tensor "
            f"Helps maintain the computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(result.data, expected.data), (
            f"scalar and vector multiplication failed "
            f"{scalar.data}  - {vector.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_scalar_matrix_multiplication(self):
        "Matrix multiplication rejects 0D tensors (scalars)."
        scalar = Tensor(5.0)
        t1 = Tensor([1, 2, 3], [5, 6, 7, 8])

        expected = Tensor([5.0, 10.0, 15.0], [25.0, 30.0, 35.0, 40.0])
        result = scalar @ t1

        assert isinstance(result, Tensor), (
            f"Scalar and vector multiplication should produce a tensor "
            f"Helps maintain the computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(result.data, expected.data), (
            f"scalar and vector multiplication failed "
            f"{scalar.data}  - {t1.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_matrix_scalar_multiplication(self):

        scalar = Tensor(5.0)
        t1 = Tensor([1, 2, 3], [5, 6, 7, 8])

        expected = Tensor([5.0, 10.0, 15.0], [25.0, 30.0, 35.0, 40.0])
        result = t1 @ scalar

        assert isinstance(result, Tensor), (
            f"Scalar and vector multiplication should produce a tensor "
            f"Helps maintain the computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(result.data, expected.data), (
            f"scalar and vector multiplication failed "
            f"{scalar.data}  - {t1.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_element_wise_subtraction(self):
        "Test element wise tensor subtraction."

        t1 = Tensor([1, 2, 3, 4])
        t2 = Tensor([10, 11, 15, 25])

        expected = Tensor([9, 9, 12, 21])
        result = t2 - t1

        assert isinstance(result, Tensor), (
            f"Addition should return a tensor "
            f"This maintains the computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(expected.data, result.data), (
            f"Element-wise addition failed "
            f"{t1.data}  - {t2.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_element_wise_division(self):
        "Test element wise tensor division."

        t1 = Tensor([2, 3, 5, 5])
        t2 = Tensor([10, 12, 15, 25])

        expected = Tensor([5, 4, 3, 5])
        result = t2 / t1

        assert isinstance(result, Tensor), (
            f"Addition should return a tensor "
            f"This maintains the computational graph for autodiff "
            f"received {type(result)}"
        )

        assert np.array_equal(expected.data, result.data), (
            f"Element-wise addition failed "
            f"{t1.data}  / {t2.data} "
            f"Expected: {expected} "
            f"Got: {result.data}"
        )

    def test_tensor_negation(self):
        tensor = Tensor([1, 2, 3, 4, 5])
        result = tensor.__neg__()
        expected = Tensor([-1, -2, -3, -4, -5])

        assert np.array_equal(result.data, expected.data)

    def test_tensor_power(self):
        tensor = Tensor([1, 2, 3, 4])
        power = 2

        result = tensor**power
        expected = Tensor([1, 4, 9, 16])

        assert isinstance(result, Tensor)
        assert np.array_equal(result.data, expected.data)


class TestTensorManipulation:
    def test_tensor_sum_along_axis(self):
        tensor = Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

        # * sum along the rows
        cols_sum = tensor.sum(axis=0, keepdims=False)
        rows_sum = tensor.sum(axis=1, keepdims=False)

        cols_expected = Tensor([18, 21, 24, 27, 30])
        rows_expected = Tensor([15, 40, 65])

        assert isinstance(cols_sum, Tensor)
        assert isinstance(rows_sum, Tensor)
        assert np.array_equal(cols_expected.data, cols_sum.data)
        assert np.array_equal(rows_sum.data, rows_expected.data)

    def test_tensor_mean_along_axis(self):
        tensor = Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

        # * sum along the rows
        cols_mean = tensor.mean(axis=0, keepdims=False)
        rows_mean = tensor.mean(axis=1, keepdims=False)

        cols_expected = Tensor([6, 7, 8, 9, 10])
        rows_expected = Tensor([3, 8, 13])

        assert isinstance(cols_mean, Tensor)
        assert isinstance(rows_mean, Tensor)
        assert np.array_equal(cols_expected.data, cols_mean.data)
        assert np.array_equal(rows_mean.data, rows_expected.data)

    def test_tensor_max_along_axis(self):
        tensor = Tensor([[6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [1, 2, 3, 4, 5]])

        # * sum along the rows
        cols_mean = tensor.max(axis=0, keepdims=False)
        rows_mean = tensor.max(axis=1, keepdims=False)

        cols_expected = Tensor([11, 12, 13, 14, 15])
        rows_expected = Tensor([10, 15, 5])

        assert isinstance(cols_mean, Tensor)
        assert isinstance(rows_mean, Tensor)
        assert np.array_equal(cols_expected.data, cols_mean.data)
        assert np.array_equal(rows_mean.data, rows_expected.data)

    def test_tensor_transpose(self):
        tensor1 = Tensor([[6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [1, 2, 3, 4, 5]])
        transposed1 = tensor1.transpose(dim0=1, dim1=0)
        expected1 = Tensor(
            [[6, 11, 1], [7, 12, 2], [8, 13, 3], [9, 14, 4], [10, 15, 5]]
        )

        assert isinstance(transposed1, Tensor)
        assert np.array_equal(transposed1.data, expected1.data)

        tensor2 = Tensor([[6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [1, 2, 3, 4, 5]])
        transposed2 = tensor2.transpose()
        expected2 = Tensor(
            [[6, 11, 1], [7, 12, 2], [8, 13, 3], [9, 14, 4], [10, 15, 5]]
        )

        assert isinstance(transposed2, Tensor)
        assert np.array_equal(transposed2.data, expected2.data)

        tensor3 = Tensor([6, 7, 8, 9, 10])
        transposed3 = tensor3.transpose()
        expected3 = Tensor([6, 7, 8, 9, 10])

        assert isinstance(transposed3, Tensor)
        assert np.array_equal(transposed3.data, expected3.data)

    def test_tensor_indexing(self):
        tensor = Tensor([[6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [1, 2, 3, 4, 5]])

        # * tensor[j]
        key = 1
        result1 = tensor[key]
        expected = Tensor([11, 12, 13, 14, 15])

        assert isinstance(result1, Tensor)
        assert np.array_equal(expected.data, result1.data)

        # * tensor[i:j]
        result2 = tensor[key - 1 : key + 1]
        expected = Tensor([[6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])

        assert isinstance(result2, Tensor)
        assert np.array_equal(result2.data, expected.data)

        # * tensor[:, :key]
        result3 = tensor[:, : key + 1]
        expected = Tensor([[6, 7], [11, 12], [1, 2]])

        assert isinstance(result3, Tensor)
        assert np.array_equal(expected.data, result3.data)

        # * tensor[:, 1:-1]
        result4 = tensor[:, key:-1]
        expected = Tensor([[7, 8, 9], [12, 13, 14], [2, 3, 4]])

        assert isinstance(result4, Tensor)
        assert np.array_equal(expected.data, result4.data)

        # * tensor[:, :-1]
        result5 = tensor[:, :-1]
        expected = Tensor([[6, 7, 8, 9], [11, 12, 13, 14], [1, 2, 3, 4]])

        assert isinstance(result5, Tensor)
        assert np.array_equal(expected.data, result5.data)

        # * tensor[i,j]
        result6 = tensor[key, key]
        expected = Tensor(12)

        assert isinstance(result6, Tensor)
        assert np.array_equal(expected.data, result6.data)
