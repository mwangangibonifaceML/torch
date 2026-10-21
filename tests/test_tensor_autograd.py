"""
Unit tests for the autograd engine of `Tensor`.

Run with:  pytest test_tensor_autograd.py -v

Adjust the import below to wherever your Tensor class lives.

Strategy
--------
1. Exact tests: gradients with known closed-form answers.
2. Numerical tests: every op is compared against central finite differences
   (`check_grad`). Each output is weighted by a fixed random tensor so that
   transposition / axis mix-ups can't cancel out the way they do with a plain sum.
3. Graph tests: shared nodes, diamonds, chains, requires_grad propagation.

Tests marked `known_bug` fail against the class as written (I found these by
reading the code). They are xfail so the suite stays green; remove the marker
once you fix the corresponding bug (use `pytest --runxfail` to see them fail).
"""

import numpy as np
import pytest

from minitorch.tensor import Tensor

known_bug = lambda reason: pytest.mark.xfail(reason=reason, strict=False)

RNG = np.random.RandomState(42)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def pos(*shape, lo=0.5, hi=2.0):
    """Strictly positive random array (safe for div / pow / log-like ops)."""
    return RNG.uniform(lo, hi, size=shape)


def rnd(*shape):
    return RNG.randn(*shape)


def check_grad(fn, *arrays, eps=1e-2, rtol=2e-2, atol=2e-2):
    """Compare autograd gradients of `fn` against central finite differences.

    fn: callable taking Tensors, returning a Tensor (any shape).
    Tolerances are loose because the Tensor class forces float32.
    """
    arrays = [np.asarray(a, dtype=np.float64) for a in arrays]
    tensors = [Tensor(a.copy(), requires_grad=True) for a in arrays]
    out = fn(*tensors)
    W = np.asarray(RNG.uniform(0.5, 1.5, size=out.shape), dtype=np.float32)
    loss = (out * Tensor(W)).sum()
    loss.backward()

    def scalar_f(arrs):
        o = fn(*[Tensor(a) for a in arrs])
        return float((o.data.astype(np.float64) * W).sum())

    for i, arr in enumerate(arrays):
        num = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            plus = [a.copy() for a in arrays]
            minus = [a.copy() for a in arrays]
            plus[i][idx] += eps
            minus[i][idx] -= eps
            num[idx] = (scalar_f(plus) - scalar_f(minus)) / (2 * eps)
        g = tensors[i].grad
        assert g is not None, f"input {i} has no grad"
        assert g.shape == arr.shape, f"input {i}: grad shape {g.shape} != {arr.shape}"
        np.testing.assert_allclose(
            g, num, rtol=rtol, atol=atol, err_msg=f"gradient mismatch for input {i}"
        )


def T(data, rg=True):
    return Tensor(np.array(data, dtype=np.float32), requires_grad=rg)


# --------------------------------------------------------------------------- #
# 1. Leaf / bookkeeping behaviour
# --------------------------------------------------------------------------- #
class TestLeafBasics:
    def test_grad_none_without_requires_grad(self):
        assert Tensor([1.0, 2.0]).grad is None

    def test_grad_initialised_to_zeros(self):
        t = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        np.testing.assert_array_equal(t.grad, np.zeros(3))
        assert t.grad.dtype == np.float32

    def test_backward_seeds_ones(self):
        x = T([1.0, 2.0, 3.0])
        x.backward()
        np.testing.assert_array_equal(x.grad, np.ones(3))

    def test_backward_on_scalar_leaf(self):
        x = T(5.0)
        x.backward()
        assert x.grad == 1.0

    def test_add_grad_accumulates(self):
        x = T([1.0, 2.0])
        x._add_grad(np.array([1.0, 1.0], dtype=np.float32))
        x._add_grad(np.array([2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(x.grad, [3.0, 4.0])

    def test_add_grad_ignored_when_not_requires_grad(self):
        x = T([1.0, 2.0], rg=False)
        x._add_grad(np.array([1.0, 1.0], dtype=np.float32))
        assert x.grad is None

    def test_add_grad_accepts_tensor(self):
        x = T([1.0, 2.0])
        x._add_grad(Tensor([5.0, 6.0]))
        np.testing.assert_allclose(x.grad, [5.0, 6.0])

    def test_add_grad_unbroadcasts_scalar_to_shape(self):
        x = T([[1.0, 2.0], [3.0, 4.0]])
        x._add_grad(np.ones((2, 2), dtype=np.float32))
        assert x.grad.shape == (2, 2)

    def test_detach_cuts_graph(self):
        x = T([1.0, 2.0])
        y = (x * 3).detach()
        assert y.requires_grad is False
        assert y._parents == ()
        z = y * 2
        assert z.requires_grad is False

    def test_detach_copies_data(self):
        x = T([1.0, 2.0])
        d = x.detach()
        d.data[0] = 99
        assert x.data[0] == 1.0

    def test_parents_recorded(self):
        a, b = T([1.0]), T([2.0])
        c = a + b
        assert set(map(id, c._parents)) == {id(a), id(b)}

    def test_invalid_data_type_raises(self):
        with pytest.raises(ValueError):
            Tensor("not a tensor")

    def test_grads_are_float32(self):
        x = T([1.0, 2.0])
        (x * x).sum().backward()
        assert x.grad.dtype == np.float32


# --------------------------------------------------------------------------- #
# 2. Elementwise ops: exact gradients
# --------------------------------------------------------------------------- #
class TestElementwiseExact:
    def test_add(self):
        a, b = T([1.0, 2.0]), T([3.0, 4.0])
        (a + b).sum().backward()
        np.testing.assert_allclose(a.grad, [1, 1])
        np.testing.assert_allclose(b.grad, [1, 1])

    def test_sub(self):
        a, b = T([1.0, 2.0]), T([3.0, 4.0])
        (a - b).sum().backward()
        np.testing.assert_allclose(a.grad, [1, 1])
        np.testing.assert_allclose(b.grad, [-1, -1])

    def test_mul(self):
        a, b = T([1.0, 2.0]), T([3.0, 4.0])
        (a * b).sum().backward()
        np.testing.assert_allclose(a.grad, [3, 4])
        np.testing.assert_allclose(b.grad, [1, 2])

    def test_div(self):
        a, b = T([2.0, 6.0]), T([4.0, 3.0])
        (a / b).sum().backward()
        np.testing.assert_allclose(a.grad, [0.25, 1 / 3], rtol=1e-5)
        np.testing.assert_allclose(b.grad, [-2 / 16, -6 / 9], rtol=1e-5)

    def test_neg(self):
        a = T([1.0, -2.0])
        (-a).sum().backward()
        np.testing.assert_allclose(a.grad, [-1, -1])

    def test_pow_int(self):
        a = T([1.0, 2.0, 3.0])
        (a**3).sum().backward()
        np.testing.assert_allclose(a.grad, [3, 12, 27], rtol=1e-5)

    def test_pow_float(self):
        a = T([4.0, 9.0])
        (a**0.5).sum().backward()
        np.testing.assert_allclose(a.grad, [0.25, 1 / 6], rtol=1e-5)

    def test_pow_zero_exponent(self):
        a = T([1.0, 2.0])
        (a**0).sum().backward()
        np.testing.assert_allclose(a.grad, [0, 0])

    def test_pow_negative_exponent(self):
        a = T([2.0, 4.0])
        (a**-1).sum().backward()
        np.testing.assert_allclose(a.grad, [-0.25, -1 / 16], rtol=1e-5)

    def test_pow_rejects_non_scalar_exponent(self):
        with pytest.raises(AssertionError):
            T([1.0]) ** T([2.0])

    # ---- reflected / scalar operators -------------------------------------
    def test_radd_scalar(self):
        a = T([1.0, 2.0])
        (5.0 + a).sum().backward()
        np.testing.assert_allclose(a.grad, [1, 1])

    def test_add_scalar(self):
        a = T([1.0, 2.0])
        (a + 5.0).sum().backward()
        np.testing.assert_allclose(a.grad, [1, 1])

    def test_rmul_scalar(self):
        a = T([1.0, 2.0])
        (3.0 * a).sum().backward()
        np.testing.assert_allclose(a.grad, [3, 3])

    def test_mul_scalar(self):
        a = T([1.0, 2.0])
        (a * 3.0).sum().backward()
        np.testing.assert_allclose(a.grad, [3, 3])

    def test_sub_scalar(self):
        a = T([1.0, 2.0])
        (a - 3.0).sum().backward()
        np.testing.assert_allclose(a.grad, [1, 1])

    def test_truediv_scalar(self):
        a = T([2.0, 4.0])
        (a / 2.0).sum().backward()
        np.testing.assert_allclose(a.grad, [0.5, 0.5])

    @known_bug(
        "__rsub__ builds Tensor(other - self.data) -> Tensor(Tensor) raises ValueError"
    )
    def test_rsub_scalar(self):
        a = T([1.0, 2.0])
        out = 10.0 - a
        np.testing.assert_allclose(out.data, [9, 8])
        out.sum().backward()
        np.testing.assert_allclose(a.grad, [-1, -1])

    def test_rtruediv_scalar(self):
        a = T([2.0, 4.0])
        (1.0 / a).sum().backward()
        np.testing.assert_allclose(a.grad, [-0.25, -1 / 16], rtol=1e-5)

    def test_rtruediv_forward_value(self):
        a = T([2.0, 4.0])
        np.testing.assert_allclose((8.0 / a).data, [4, 2])


# --------------------------------------------------------------------------- #
# 3. Elementwise ops: numerical gradient checks (incl. broadcasting)
# --------------------------------------------------------------------------- #
BROADCAST_SHAPES = [
    ((3, 4), (3, 4)),
    ((3, 4), (4,)),
    ((3, 4), (3, 1)),
    ((3, 1), (1, 4)),
    ((2, 3, 4), (4,)),
    ((2, 3, 4), (3, 1)),
    ((2, 1, 4), (3, 1)),
    ((3, 4), ()),
    ((), (3, 4)),
]


@pytest.mark.parametrize("sa,sb", BROADCAST_SHAPES)
class TestBroadcastNumerical:
    def test_add(self, sa, sb):
        check_grad(lambda a, b: a + b, rnd(*sa), rnd(*sb))

    def test_sub(self, sa, sb):
        check_grad(lambda a, b: a - b, rnd(*sa), rnd(*sb))

    def test_mul(self, sa, sb):
        check_grad(lambda a, b: a * b, rnd(*sa), rnd(*sb))

    def test_div(self, sa, sb):
        check_grad(lambda a, b: a / b, rnd(*sa), pos(*sb))


class TestElementwiseNumerical:
    def test_pow_chain(self):
        check_grad(lambda a: (a**2) ** 1.5, pos(3, 3))

    def test_neg_chain(self):
        check_grad(lambda a, b: -(a * b) - a, rnd(2, 3), rnd(2, 3))

    def test_rtruediv(self):
        check_grad(lambda a: 2.0 / a, pos(2, 3))

    def test_rational_expression(self):
        check_grad(lambda a, b: (a * b + a) / (b**2 + 1.0), rnd(3, 2), rnd(3, 2))

    def test_scalar_tensor_broadcast_gradient_is_summed(self):
        a = T([[1.0, 2.0], [3.0, 4.0]])
        s = T(2.0)
        (a * s).sum().backward()
        assert s.grad.shape == ()
        np.testing.assert_allclose(s.grad, 10.0)  # sum of a

    def test_bias_gradient_is_summed_over_batch(self):
        x = T(np.ones((5, 3)))
        b = T(np.zeros(3))
        (x + b).sum().backward()
        assert b.grad.shape == (3,)
        np.testing.assert_allclose(b.grad, [5, 5, 5])


# --------------------------------------------------------------------------- #
# 4. Graph structure
# --------------------------------------------------------------------------- #
class TestGraph:
    def test_x_times_x(self):
        x = T([1.0, 2.0, 3.0])
        (x * x).sum().backward()
        np.testing.assert_allclose(x.grad, [2, 4, 6])

    def test_x_plus_x(self):
        x = T([1.0, 2.0])
        (x + x).sum().backward()
        np.testing.assert_allclose(x.grad, [2, 2])

    def test_x_minus_x_is_zero(self):
        x = T([1.0, 2.0])
        (x - x).sum().backward()
        np.testing.assert_allclose(x.grad, [0, 0])

    def test_x_div_x_is_zero(self):
        x = T([1.0, 2.0])
        (x / x).sum().backward()
        np.testing.assert_allclose(x.grad, [0, 0], atol=1e-6)

    def test_diamond(self):
        # y = x*2 ; z = x*3 ; out = y*z = 6x^2  -> d/dx = 12x
        x = T([1.0, 2.0])
        y, z = x * 2, x * 3
        (y * z).sum().backward()
        np.testing.assert_allclose(x.grad, [12, 24])

    def test_reused_intermediate(self):
        # h = x*x ; out = h + h*h = x^2 + x^4 -> 2x + 4x^3
        x = T([1.0, 2.0])
        h = x * x
        (h + h * h).sum().backward()
        np.testing.assert_allclose(x.grad, [6, 36], rtol=1e-5)

    def test_deep_chain(self):
        x = T([1.0, 2.0])
        y = x
        for _ in range(50):
            y = y * 1.0 + 0.0
        y.sum().backward()
        np.testing.assert_allclose(x.grad, [1, 1])

    def test_deep_chain_multiplicative(self):
        x = T([1.0, 1.0])
        y = x
        for _ in range(10):
            y = y * 2.0
        y.sum().backward()
        np.testing.assert_allclose(x.grad, [1024, 1024])

    def test_two_independent_leaves(self):
        a, b = T([2.0]), T([3.0])
        (a * b + a**2).sum().backward()
        np.testing.assert_allclose(a.grad, [3 + 4])
        np.testing.assert_allclose(b.grad, [2])

    def test_requires_grad_propagation(self):
        a, b = T([1.0]), T([2.0], rg=False)
        assert (a + b).requires_grad
        assert (b + a).requires_grad
        assert not (b + b).requires_grad

    def test_constant_operand_gets_no_grad(self):
        a, c = T([1.0, 2.0]), T([3.0, 4.0], rg=False)
        (a * c).sum().backward()
        assert c.grad is None
        np.testing.assert_allclose(a.grad, [3, 4])

    def test_no_grad_graph_backward_does_not_crash(self):
        a, b = T([1.0], rg=False), T([2.0], rg=False)
        (a * b).sum().backward()  # nothing requires grad

    def test_topological_order_with_late_use(self):
        # out = a*b + c where c depends on a too; ordering must be respected
        a, b = T([2.0]), T([3.0])
        c = a * a
        out = a * b + c
        out.sum().backward()
        np.testing.assert_allclose(a.grad, [3 + 4])
        np.testing.assert_allclose(b.grad, [2])

    def test_backward_on_non_scalar_seeds_ones(self):
        x = T([1.0, 2.0])
        y = x * 3
        y.backward()
        np.testing.assert_allclose(x.grad, [3, 3])

    @known_bug(
        "intermediate nodes' grads are never reset, so a 2nd backward() "
        "over the same graph double-counts upstream gradients"
    )
    def test_second_backward_accumulates_on_leaf_only(self):
        x = T([1.0, 2.0])
        loss = (x * 3 * 2).sum()
        loss.backward()
        loss.backward()
        np.testing.assert_allclose(
            x.grad, [12, 12]
        )  # 2 x 6, like PyTorch w/ retain_graph


# --------------------------------------------------------------------------- #
# 5. Reductions
# --------------------------------------------------------------------------- #
SUM_AXES = [None, 0, 1, 2, -1, -2, (0, 1), (0, 2), (1, 2), (0, 1, 2)]


class TestSum:
    @pytest.mark.parametrize("axis", SUM_AXES)
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_sum_numerical(self, axis, keepdims):
        check_grad(lambda a: a.sum(axis=axis, keepdims=keepdims), rnd(2, 3, 4))

    def test_sum_all_exact(self):
        x = T(np.ones((2, 3)))
        x.sum().backward()
        np.testing.assert_allclose(x.grad, np.ones((2, 3)))

    def test_sum_scalar_input(self):
        x = T(3.0)
        x.sum().backward()
        np.testing.assert_allclose(x.grad, 1.0)

    def test_sum_then_scale(self):
        x = T(np.ones((2, 3)))
        (x.sum() * 5).backward()
        np.testing.assert_allclose(x.grad, np.full((2, 3), 5.0))

    def test_sum_axis_then_broadcast_back(self):
        # softmax-style: x / x.sum(axis=-1, keepdims=True)
        check_grad(lambda a: a / a.sum(axis=-1, keepdims=True), pos(3, 4))


class TestMean:
    @pytest.mark.parametrize("axis", [None, 0, 1, 2, -1, (0, 1), (1, 2), (0, 1, 2)])
    @pytest.mark.parametrize("keepdims", [False, True])
    def test_mean_numerical(self, axis, keepdims):
        check_grad(lambda a: a.mean(axis=axis, keepdims=keepdims), rnd(2, 3, 4))

    def test_mean_exact(self):
        x = T(np.ones((2, 5)))
        x.mean().backward()
        np.testing.assert_allclose(x.grad, np.full((2, 5), 0.1), rtol=1e-6)

    def test_mean_axis_exact(self):
        x = T(np.ones((4, 3)))
        x.mean(axis=0).sum().backward()
        np.testing.assert_allclose(x.grad, np.full((4, 3), 0.25), rtol=1e-6)

    def test_manual_variance_via_mean(self):
        # layernorm-style statistics built from differentiable primitives
        def f(a):
            mu = a.mean(axis=-1, keepdims=True)
            return ((a - mu) ** 2).mean(axis=-1, keepdims=True)

        check_grad(f, rnd(3, 5))

    def test_layernorm_like(self):
        def f(a):
            mu = a.mean(axis=-1, keepdims=True)
            var = ((a - mu) ** 2).mean(axis=-1, keepdims=True)
            return (a - mu) / ((var + 1e-5) ** 0.5)

        check_grad(f, rnd(3, 6), eps=1e-2, atol=5e-2, rtol=5e-2)


class TestVar:
    @known_bug(
        "var backward is result.grad / size; the true gradient is "
        "2*(x-mean)/(N-1) (ddof=1) and ignores axis/keepdims"
    )
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_var_numerical(self, axis):
        check_grad(lambda a: a.var(axis=axis), rnd(4, 5))

    @known_bug("see above")
    def test_var_exact(self):
        x = T([1.0, 2.0, 3.0, 4.0])
        x.var().backward()
        mean = 2.5
        expected = 2 * (np.array([1, 2, 3, 4]) - mean) / 3
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_var_forward_uses_ddof_1(self):
        x = T([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            x.var().data, np.var([1, 2, 3, 4], ddof=1), rtol=1e-6
        )


class TestMax:
    @known_bug("max() defines no _backward, so no gradient reaches the input")
    @pytest.mark.parametrize("axis", [None, 0, 1])
    def test_max_numerical(self, axis):
        # well-separated values avoid ties/non-differentiable points
        a = np.arange(12, dtype=np.float64).reshape(3, 4)
        RNG.shuffle(a.ravel())
        check_grad(lambda t: t.max(axis=axis), a)

    @known_bug("see above")
    def test_max_exact(self):
        x = T([1.0, 5.0, 3.0])
        x.max().backward()
        np.testing.assert_allclose(x.grad, [0, 1, 0])


# --------------------------------------------------------------------------- #
# 6. Shape ops
# --------------------------------------------------------------------------- #
class TestReshape:
    def test_reshape_numerical(self):
        check_grad(lambda a: a.reshape(6, 2), rnd(3, 4))

    def test_reshape_tuple_arg(self):
        check_grad(lambda a: a.reshape((4, 3)), rnd(3, 4))

    def test_reshape_infer_dim(self):
        check_grad(lambda a: a.reshape(-1, 2), rnd(3, 4))

    def test_reshape_to_flat(self):
        check_grad(lambda a: a.reshape(-1), rnd(2, 3, 2))

    def test_reshape_grad_shape_matches_input(self):
        x = T(np.ones((2, 6)))
        (x.reshape(3, 4) * 2).sum().backward()
        assert x.grad.shape == (2, 6)

    def test_reshape_with_downstream_nonuniform_weights(self):
        check_grad(
            lambda a: a.reshape(4, 3) * Tensor(np.arange(12).reshape(4, 3)), rnd(3, 4)
        )

    def test_reshape_bad_size_raises(self):
        with pytest.raises(ValueError):
            T(np.ones((2, 3))).reshape(4, 2)

    def test_reshape_two_unknowns_raises(self):
        with pytest.raises(ValueError):
            T(np.ones((2, 3))).reshape(-1, -1)


class TestTranspose:
    def test_default_2d(self):
        check_grad(lambda a: a.transpose(), rnd(3, 4))

    def test_default_3d_swaps_last_two(self):
        check_grad(lambda a: a.transpose(), rnd(2, 3, 4))

    def test_explicit_dims(self):
        check_grad(lambda a: a.transpose(0, 2), rnd(2, 3, 4))

    def test_explicit_dims_middle(self):
        check_grad(lambda a: a.transpose(1, 2), rnd(2, 3, 4))

    def test_negative_dims(self):
        check_grad(lambda a: a.transpose(-1, -2), rnd(2, 3, 4))

    def test_1d_noop(self):
        check_grad(lambda a: a.transpose(), rnd(5))

    def test_double_transpose_is_identity(self):
        x = T(rnd(3, 4))
        (x.transpose().transpose() * 2).sum().backward()
        np.testing.assert_allclose(x.grad, np.full((3, 4), 2.0))

    def test_transpose_grad_shape(self):
        x = T(np.ones((3, 5)))
        (x.transpose() * 2).sum().backward()
        assert x.grad.shape == (3, 5)

    def test_same_dims_raises(self):
        with pytest.raises(ValueError):
            T(np.ones((2, 3))).transpose(0, 0)

    def test_out_of_range_dims_raises(self):
        with pytest.raises(ValueError):
            T(np.ones((2, 3))).transpose(0, 5)


class TestGetItem:
    def test_int_index(self):
        check_grad(lambda a: a[1], rnd(3, 4))

    def test_negative_int_index(self):
        check_grad(lambda a: a[-1], rnd(3, 4))

    def test_slice(self):
        check_grad(lambda a: a[1:3], rnd(4, 3))

    def test_slice_with_step(self):
        check_grad(lambda a: a[::2], rnd(5, 3))

    def test_column_slice(self):
        check_grad(lambda a: a[:, 1:-1], rnd(3, 5))

    def test_last_column(self):
        check_grad(lambda a: a[:, -1], rnd(3, 5))

    def test_two_int_indices_scalar_result(self):
        check_grad(lambda a: a[1, 2], rnd(3, 4))

    def test_3d_mixed(self):
        check_grad(lambda a: a[:, 1, ::2], rnd(2, 3, 4))

    def test_ellipsis(self):
        check_grad(lambda a: a[..., 0], rnd(2, 3, 4))

    def test_boolean_mask(self):
        mask = np.array([True, False, True, False])
        check_grad(lambda a: a[mask], rnd(4, 3))

    def test_unselected_positions_get_zero(self):
        x = T(np.ones((3, 3)))
        x[0].sum().backward()
        expected = np.zeros((3, 3))
        expected[0] = 1
        np.testing.assert_allclose(x.grad, expected)

    def test_multiple_slices_of_same_tensor_accumulate(self):
        x = T([1.0, 2.0, 3.0, 4.0])
        (x[:3].sum() + x[1:].sum()).backward()
        np.testing.assert_allclose(x.grad, [1, 2, 2, 1])

    def test_overlapping_slices_numerical(self):
        check_grad(lambda a: a[:3] * a[1:], rnd(4))

    def test_getitem_forward_values(self):
        x = Tensor(np.arange(6).reshape(2, 3))
        np.testing.assert_array_equal(x[1].data, [3, 4, 5])

    @known_bug("grad_input[key] += g drops repeated indices; needs np.add.at")
    def test_fancy_index_with_duplicates(self):
        x = T([1.0, 2.0, 3.0])
        x[[0, 0, 2]].sum().backward()
        np.testing.assert_allclose(x.grad, [2, 0, 1])


# --------------------------------------------------------------------------- #
# 7. Matmul
# --------------------------------------------------------------------------- #
class TestMatmul:
    def test_2d_exact(self):
        A = T([[1.0, 2.0], [3.0, 4.0]])
        B = T([[5.0, 6.0], [7.0, 8.0]])
        (A @ B).sum().backward()
        np.testing.assert_allclose(A.grad, np.ones((2, 2)) @ B.data.T)
        np.testing.assert_allclose(B.grad, A.data.T @ np.ones((2, 2)))

    def test_2d_numerical(self):
        check_grad(lambda a, b: a @ b, rnd(3, 4), rnd(4, 5))

    def test_non_square_numerical(self):
        check_grad(lambda a, b: a @ b, rnd(2, 5), rnd(5, 1))

    def test_method_and_operator_agree(self):
        a, b = rnd(2, 3), rnd(3, 2)
        np.testing.assert_allclose(T(a).matmul(T(b)).data, (T(a) @ T(b)).data)

    def test_only_left_requires_grad(self):
        A, B = T(rnd(2, 3)), T(rnd(3, 4), rg=False)
        (A @ B).sum().backward()
        assert B.grad is None
        np.testing.assert_allclose(A.grad, np.ones((2, 4)) @ B.data.T, rtol=1e-5)

    def test_only_right_requires_grad(self):
        A, B = T(rnd(2, 3), rg=False), T(rnd(3, 4))
        (A @ B).sum().backward()
        assert A.grad is None
        np.testing.assert_allclose(B.grad, A.data.T @ np.ones((2, 4)), rtol=1e-5)

    def test_batched_3d(self):
        check_grad(lambda a, b: a @ b, rnd(2, 3, 4), rnd(2, 4, 5))

    def test_batched_4d(self):
        check_grad(lambda a, b: a @ b, rnd(2, 2, 3, 4), rnd(2, 2, 4, 2))

    def test_batched_times_shared_2d_weight(self):
        # (B,T,D) @ (D,H): weight grad must be summed over batch
        check_grad(lambda x, w: x @ w, rnd(3, 4, 5), rnd(5, 2))

    def test_batch_broadcast_left_2d(self):
        check_grad(lambda a, b: a @ b, rnd(3, 4), rnd(2, 4, 5))

    def test_matrix_vector_grad_wrt_matrix(self):
        A, v = T(rnd(3, 4)), T(rnd(4), rg=False)
        (A @ v).sum().backward()
        np.testing.assert_allclose(A.grad, np.tile(v.data, (3, 1)), rtol=1e-5)

    @known_bug(
        "matmul backward for 1-D right operand: uses np.outer(self.data @ grad) "
        "(wrong arity) and writes to self instead of other"
    )
    def test_matrix_vector_grad_wrt_vector(self):
        check_grad(lambda a, v: a @ v, rnd(3, 4), rnd(4))

    @known_bug("vector @ matrix: backward calls swapaxes on a 1-D tensor")
    def test_vector_matrix(self):
        check_grad(lambda v, B: v @ B, rnd(4), rnd(4, 3))

    @known_bug("1-D @ 1-D dot product backward hits the broken 1-D branch")
    def test_dot_product(self):
        a, b = T(rnd(4)), T(rnd(4))
        (a @ b).backward()
        np.testing.assert_allclose(a.grad, b.data, rtol=1e-5)
        np.testing.assert_allclose(b.grad, a.data, rtol=1e-5)

    @known_bug("scalar branch of matmul reuses the 2-D backward formulas")
    def test_scalar_matmul(self):
        a, b = T(3.0), T(4.0)
        (a @ b).backward()
        np.testing.assert_allclose(a.grad, 4.0)
        np.testing.assert_allclose(b.grad, 3.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            T(np.ones((2, 3))) @ T(np.ones((4, 5)))

    def test_matmul_chain(self):
        check_grad(lambda a, b, c: (a @ b) @ c, rnd(2, 3), rnd(3, 4), rnd(4, 2))

    def test_matmul_with_transpose(self):
        # attention-style scores: Q @ K^T
        check_grad(lambda q, k: q @ k.transpose(), rnd(3, 4), rnd(5, 4))

    def test_batched_attention_scores(self):
        check_grad(lambda q, k: q @ k.transpose(-1, -2), rnd(2, 3, 4), rnd(2, 5, 4))


# --------------------------------------------------------------------------- #
# 8. Composite / "mini network" tests
# --------------------------------------------------------------------------- #
class TestComposite:
    def test_linear_layer_mse(self):
        X = rnd(6, 3)
        y = rnd(6, 2)

        def f(W, b):
            pred = Tensor(X) @ W + b
            return ((pred - Tensor(y)) ** 2).mean()

        check_grad(f, rnd(3, 2), rnd(2))

    def test_two_layer_net_with_input_grad(self):
        def f(x, W1, W2):
            return ((x @ W1) @ W2 * 0.5 + 1.0).mean()

        check_grad(f, rnd(4, 3), rnd(3, 5), rnd(5, 2))

    def test_reshape_matmul_sum_chain(self):
        def f(a, w):
            return (a.reshape(6, 2) @ w).sum(axis=0)

        check_grad(f, rnd(3, 4), rnd(2, 3))

    def test_slice_then_matmul(self):
        check_grad(lambda a, w: a[:, :3] @ w, rnd(4, 5), rnd(3, 2))

    def test_attention_like_block(self):
        def f(x, Wq, Wk, Wv):
            q, k, v = x @ Wq, x @ Wk, x @ Wv
            scores = (q @ k.transpose()) * (1.0 / 2.0)
            w = scores / (scores * scores + 1.0)  # stand-in for softmax (no exp op)
            return w @ v

        check_grad(f, rnd(3, 4), rnd(4, 4), rnd(4, 4), rnd(4, 4), atol=5e-2, rtol=5e-2)

    def test_gradient_descent_step_reduces_loss(self):
        X, y = rnd(10, 3), rnd(10, 1)
        W = T(rnd(3, 1))

        def loss_of(W_):
            return ((Tensor(X) @ W_ - Tensor(y)) ** 2).mean()

        l0 = loss_of(W)
        l0.backward()
        W2 = Tensor(W.data - 0.05 * W.grad, requires_grad=True)
        assert float(loss_of(W2).data) < float(l0.data)


# --------------------------------------------------------------------------- #
# 9. Classmethods that should participate in autograd
# --------------------------------------------------------------------------- #
class TestMaskedFill:
    @known_bug(
        "masked_fill never assigns result._backward, and reads input.grad "
        "instead of result.grad"
    )
    def test_causal_mask_grad(self):
        x = T(np.ones((1, 3, 3)))
        out = Tensor.masked_fill(x, filler=-1e9)
        # only lower-triangle (incl. diagonal) survives; use a finite filler
        (out * Tensor(np.tril(np.ones((3, 3))))).sum().backward()
        np.testing.assert_allclose(x.grad[0], np.tril(np.ones((3, 3))))

    def test_masked_fill_forward(self):
        x = Tensor(np.ones((1, 3, 3)))
        out = Tensor.masked_fill(x, filler=-5.0)
        assert out.data[0, 0, 1] == -5.0 and out.data[0, 1, 0] == 1.0

    def test_masked_fill_rejects_2d(self):
        with pytest.raises(ValueError):
            Tensor.masked_fill(Tensor(np.ones((3, 3))))


class TestStackAndTril:
    @known_bug("Tensor.stack drops the graph (no _parents / _backward)")
    def test_stack_grad(self):
        a, b = T([1.0, 2.0]), T([3.0, 4.0])
        out = Tensor.stack([a, b], axis=0)
        (out * Tensor([[1.0, 1.0], [2.0, 2.0]])).sum().backward()
        np.testing.assert_allclose(a.grad, [1, 1])
        np.testing.assert_allclose(b.grad, [2, 2])

    @known_bug("Tensor.tril drops the graph (no _parents / _backward)")
    def test_tril_grad(self):
        x = T(np.ones((3, 3)))
        Tensor.tril(x, requires_grad=True).sum().backward()
        np.testing.assert_allclose(x.grad, np.tril(np.ones((3, 3))))

    def test_stack_forward(self):
        out = Tensor.stack([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])], axis=1)
        assert out.shape == (2, 2)

    def test_tril_forward(self):
        out = Tensor.tril(Tensor(np.ones((3, 3))))
        np.testing.assert_array_equal(out.data, np.tril(np.ones((3, 3))))


# --------------------------------------------------------------------------- #
# 10. Invariants over many ops
# --------------------------------------------------------------------------- #
OPS = {
    "add": lambda a: a + a,
    "mul": lambda a: a * a,
    "sub": lambda a: a - 1.0,
    "div": lambda a: a / 2.0,
    "pow": lambda a: a**2,
    "neg": lambda a: -a,
    "sum0": lambda a: a.sum(axis=0),
    "mean1": lambda a: a.mean(axis=1),
    "T": lambda a: a.transpose(),
    "reshape": lambda a: a.reshape(-1),
    "slice": lambda a: a[1:],
    "matmul": lambda a: a @ a.transpose(),
}


@pytest.mark.parametrize("name", OPS)
def test_grad_shape_and_dtype_invariant(name):
    x = T(pos(3, 3))
    out = OPS[name](x)
    out.sum().backward()
    assert x.grad.shape == x.shape
    assert x.grad.dtype == np.float32
    assert np.all(np.isfinite(x.grad))


@pytest.mark.parametrize("name", OPS)
def test_numerical_invariant(name):
    check_grad(OPS[name], pos(3, 3))
