"""
test_activation_layers.py -- unit tests for Sigmoid, ReLU, Tanh, GELU, Softmax.

What is covered
  * forward values against independent float64 reference implementations
  * backward: closed-form derivatives AND finite-difference gradient checks
  * graph behaviour: requires_grad propagation, no graph kept at inference,
    fan-out accumulation, layer reuse, chain rule through neighbouring ops
  * input handling: Tensor / ndarray / list / scalar / float64 / int / non-contiguous
  * numerics: extreme values, NaN / inf, no RuntimeWarnings, no NaN gradients
  * per-activation properties (symmetry, bounds, kinks, Jacobian, masks, ...)
  * integration: small networks must actually learn (regression + classification)
"""

import contextlib
import math
import unittest
import warnings

import numpy as np

from minitorch.activations import GELU, ReLU, Sigmoid, Softmax, Tanh
from minitorch.tensor import Tensor

# =============================================================================
# Reference math: float64, written independently of the code under test
# =============================================================================
_erf = np.vectorize(math.erf)
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)
_C = math.sqrt(2.0 / math.pi)


def sigmoid_ref(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_dref(x):
    s = sigmoid_ref(x)
    return s * (1.0 - s)


def tanh_ref(x):
    return np.tanh(x)


def tanh_dref(x):
    return 1.0 - np.tanh(x) ** 2


def relu_ref(x):
    return np.maximum(x, 0.0)


def relu_dref(x):
    return (x > 0).astype(np.float64)  # convention: 0 at x == 0


def gelu_ref(x):
    return x * 0.5 * (1.0 + _erf(x / _SQRT2))


def gelu_dref(x):
    return 0.5 * (1.0 + _erf(x / _SQRT2)) + x * np.exp(-0.5 * x * x) / _SQRT2PI


def gelu_tanh_ref(x):
    return 0.5 * x * (1.0 + np.tanh(_C * (x + 0.044715 * x**3)))


def gelu_tanh_dref(x):
    t = np.tanh(_C * (x + 0.044715 * x**3))
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * _C * (1.0 + 3 * 0.044715 * x * x)


def gelu_sig_ref(x):
    return x * sigmoid_ref(1.702 * x)


def gelu_sig_dref(x):
    s = sigmoid_ref(1.702 * x)
    return s + 1.702 * x * s * (1.0 - s)


def softmax_ref(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# =============================================================================
# Helpers
# =============================================================================
def T(a, requires_grad=False, **kw):
    """float32 Tensor from anything array-like."""
    return Tensor(np.asarray(a, dtype=np.float32), requires_grad=requires_grad, **kw)


@contextlib.contextmanager
def no_numeric_warnings():
    """Turn overflow / invalid / divide RuntimeWarnings into errors."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def analytic_grad(fn, x0, w):
    """d/dx of sum(fn(x) * w), through the engine's backward()."""
    x = T(x0, True)
    (fn(x) * T(w)).sum().backward()
    return x.grad


def numeric_grad(fn, x0, w, eps):
    """Central finite differences of sum(fn(x) * w) (accumulated in float64)."""
    g = np.zeros(x0.shape, dtype=np.float64)
    for idx in np.ndindex(*x0.shape):
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        step = float(xp[idx]) - float(xm[idx])  # the step float32 really took
        fp = (fn(T(xp)).data.astype(np.float64) * w).sum()
        fm = (fn(T(xm)).data.astype(np.float64) * w).sum()
        g[idx] = (fp - fm) / step
    return g


def assert_gradcheck(fn, x0, seed=1, eps=1e-3, rtol=1e-2, atol=1e-3, msg=""):
    rng = np.random.default_rng(seed)
    out_shape = fn(T(x0)).shape
    w = rng.standard_normal(out_shape).astype(np.float32)
    np.testing.assert_allclose(
        analytic_grad(fn, x0, w),
        numeric_grad(fn, x0, w, eps),
        rtol=rtol,
        atol=atol,
        err_msg=msg or "backward() disagrees with finite differences",
    )


def sgd(params, lr):
    for p in params:
        p.data -= lr * p.grad
        p.grad[...] = 0.0


EXTREME = np.array([-1e4, -100, -30, -1, 0, 1, 30, 100, 1e4], dtype=np.float32)
GRID = np.linspace(-8, 8, 161).astype(np.float32)


# =============================================================================
# Contract shared by EVERY activation layer (mixin; collected via subclasses)
# =============================================================================
class LayerContract:
    make = None  # staticmethod -> fresh layer
    shapes = [(), (1,), (5,), (3, 4), (2, 3, 4)]

    def setUp(self):
        self.layer = self.make()
        self.rng = np.random.default_rng(0)

    def sample(self, shape=(3, 4), scale=1.5):
        x = np.asarray(self.rng.standard_normal(shape) * scale, dtype=np.float32)
        return np.where(np.abs(x) < 0.05, x + 0.3, x).astype(
            np.float32
        )  # stay off ReLU's kink

    # ---- shapes / dtypes ----------------------------------------------------
    def test_output_shape_dtype_and_gradient_dtype(self):
        for shape in self.shapes:
            x = T(self.sample(shape), True)
            out = self.layer(x)
            self.assertEqual(out.shape, shape, f"output shape for input {shape}")
            self.assertEqual(out.dtype, np.float32)
            out.sum().backward()
            self.assertEqual(x.grad.shape, shape, f"grad shape for input {shape}")
            self.assertEqual(x.grad.dtype, np.float32)

    # ---- graph bookkeeping --------------------------------------------------
    def test_no_graph_when_input_does_not_require_grad(self):
        out = self.layer(T(self.sample()))
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, (), "inference should not keep the graph alive")
        self.assertIsNone(out.grad)

    def test_graph_is_linked_when_input_requires_grad(self):
        x = T(self.sample(), True)
        out = self.layer(x)
        self.assertTrue(out.requires_grad)
        self.assertTrue(
            any(p is x for p in out._parents), "input must be a parent of the output"
        )

    # ---- input handling -----------------------------------------------------
    def test_accepts_non_tensor_inputs(self):
        base = self.sample()
        expected = self.layer(T(base)).data
        candidates = {
            "float32 ndarray": base,
            "float64 ndarray": base.astype(np.float64),
            "nested list": base.tolist(),
            "fortran-ordered": np.asfortranarray(base),
        }
        for name, raw in candidates.items():
            np.testing.assert_allclose(
                self.layer(raw).data, expected, rtol=1e-6, atol=1e-6, err_msg=name
            )
        view = base.T  # non-contiguous view
        np.testing.assert_allclose(
            self.layer(view).data,
            self.layer(T(view)).data,
            rtol=1e-6,
            atol=1e-6,
            err_msg="non-contiguous view",
        )
        ints = np.array([[1, -2, 3], [0, 4, -5]])
        np.testing.assert_allclose(
            self.layer(ints).data,
            self.layer(T(ints)).data,
            rtol=1e-6,
            atol=1e-6,
            err_msg="integer array",
        )

    def test_preserves_device(self):
        x = Tensor(
            np.ones((2, 3), dtype=np.float32), requires_grad=True, device="cuda:0"
        )
        self.assertEqual(self.layer(x).device, "cuda:0")

    def test_does_not_mutate_input(self):
        x0 = self.sample()
        w = self.rng.standard_normal(x0.shape).astype(np.float32)
        x = T(x0.copy(), True)
        (self.layer(x) * T(w)).sum().backward()
        np.testing.assert_array_equal(x.data, x0)

    # ---- gradients ----------------------------------------------------------
    def test_gradcheck_matches_finite_differences(self):
        for shape in [s for s in self.shapes if len(s) >= 2]:
            assert_gradcheck(self.layer, self.sample(shape), msg=f"shape {shape}")

    def test_gradcheck_through_surrounding_ops(self):
        fn = lambda x: self.layer(x * 2.0) * 3.0 + 1.0
        assert_gradcheck(fn, self.sample((3, 4)))

    def test_gradients_accumulate_across_fan_out(self):
        x0 = self.sample()
        w = self.rng.standard_normal(x0.shape).astype(np.float32)
        single = analytic_grad(self.layer, x0, w)
        double = analytic_grad(lambda x: self.layer(x) + self.layer(x), x0, w)
        np.testing.assert_allclose(double, 2 * single, rtol=1e-5, atol=1e-6)

    def test_zero_upstream_gradient_gives_zero_input_gradient(self):
        x0 = self.sample()
        g = analytic_grad(self.layer, x0, np.zeros_like(x0))
        self.assertTrue(np.isfinite(g).all())
        np.testing.assert_array_equal(g, np.zeros_like(x0))

    # ---- numerics -----------------------------------------------------------
    def test_extreme_inputs_stay_finite_without_warnings(self):
        with no_numeric_warnings():
            x = T(EXTREME, True)
            out = self.layer(x)
            out.sum().backward()
        self.assertTrue(np.isfinite(out.data).all(), f"non-finite output: {out.data}")
        self.assertTrue(np.isfinite(x.grad).all(), f"non-finite gradient: {x.grad}")

    # ---- module behaviour ---------------------------------------------------
    def test_call_matches_forward_and_has_no_parameters(self):
        x = T(self.sample())
        np.testing.assert_array_equal(self.layer(x).data, self.layer.forward(x).data)
        self.assertEqual(list(self.layer.parameters()), [])

    def test_is_deterministic(self):
        x = T(self.sample())
        np.testing.assert_array_equal(self.layer(x).data, self.layer(x).data)

    def test_layer_instance_is_reusable_and_stateless(self):
        # same layer object applied to two inputs (weight sharing / RNN style)
        x1, x2 = self.sample(), self.sample()
        w1 = self.rng.standard_normal(x1.shape).astype(np.float32)
        w2 = self.rng.standard_normal(x2.shape).astype(np.float32)
        a, b = T(x1, True), T(x2, True)
        out_a, out_b = self.layer(a), self.layer(b)
        ((out_a * T(w1)).sum() + (out_b * T(w2)).sum()).backward()
        np.testing.assert_allclose(
            a.grad, analytic_grad(self.make(), x1, w1), rtol=1e-5, atol=1e-6
        )
        np.testing.assert_allclose(
            b.grad, analytic_grad(self.make(), x2, w2), rtol=1e-5, atol=1e-6
        )


# =============================================================================
# Extra contract for element-wise activations
# =============================================================================
class ElementwiseContract(LayerContract):
    ref = None  # float64 forward reference
    dref = None  # float64 derivative reference

    def test_forward_matches_reference(self):
        out = self.layer(T(GRID)).data
        np.testing.assert_allclose(
            out, self.ref(GRID.astype(np.float64)), rtol=1e-5, atol=1e-5
        )

    def test_backward_matches_closed_form_derivative(self):
        x = T(GRID, True)
        self.layer(x).sum().backward()
        np.testing.assert_allclose(
            x.grad, self.dref(GRID.astype(np.float64)), rtol=1e-4, atol=1e-5
        )

    def test_accepts_python_scalars(self):
        for v in (0.5, -2.0, 3):
            np.testing.assert_allclose(
                self.layer(v).data, self.ref(np.float64(v)), rtol=1e-5, atol=1e-5
            )

    def test_nan_propagates_only_where_it_occurs(self):
        x = np.array([np.nan, -1.0, 2.0], dtype=np.float32)
        out = self.layer(T(x)).data
        self.assertTrue(
            np.isnan(out[0]), "NaN input must not be silently turned into a number"
        )
        np.testing.assert_allclose(
            out[1:], self.ref(x[1:].astype(np.float64)), rtol=1e-5, atol=1e-5
        )

    def test_each_output_depends_only_on_its_own_input(self):
        x0 = self.sample((3, 4))
        for idx in [(0, 0), (1, 2), (2, 3)]:
            w = np.zeros_like(x0)
            w[idx] = 1.0
            g = analytic_grad(self.layer, x0, w)
            mask = np.ones_like(g, dtype=bool)
            mask[idx] = False
            np.testing.assert_array_equal(
                g[mask], 0.0, err_msg=f"gradient leaked away from {idx}"
            )


# =============================================================================
# Sigmoid
# =============================================================================
class TestSigmoid(ElementwiseContract, unittest.TestCase):
    make = staticmethod(Sigmoid)
    ref, dref = staticmethod(sigmoid_ref), staticmethod(sigmoid_dref)

    def test_known_values(self):
        out = self.layer(T([-1.0, 0.0, 1.0])).data
        np.testing.assert_allclose(out, [0.26894143, 0.5, 0.7310586], rtol=1e-6)

    def test_symmetry(self):
        x = T(GRID)
        np.testing.assert_allclose(
            self.layer(x).data + self.layer(T(-GRID)).data, 1.0, atol=1e-6
        )

    def test_bounded_and_monotonic(self):
        out = self.layer(T(np.linspace(-50, 50, 1001))).data
        self.assertTrue(((out >= 0) & (out <= 1)).all())
        self.assertTrue((np.diff(out) >= 0).all())

    def test_infinite_inputs_saturate_exactly(self):
        with no_numeric_warnings():
            out = self.layer(T([np.inf, -np.inf])).data
        np.testing.assert_array_equal(out, [1.0, 0.0])

    def test_gradient_peaks_at_one_quarter(self):
        x = T(GRID, True)
        self.layer(x).sum().backward()
        self.assertLessEqual(float(x.grad.max()), 0.25 + 1e-7)
        self.assertAlmostEqual(float(self.layer_grad_at(0.0)), 0.25, places=6)

    def test_saturated_gradient_vanishes_without_nan(self):
        for v in (100.0, -100.0):
            g = float(self.layer_grad_at(v))
            self.assertTrue(
                np.isfinite(g) and 0.0 <= g < 1e-30, f"gradient at {v} was {g}"
            )

    def layer_grad_at(self, v):
        x = T([v], True)
        self.layer(x).sum().backward()
        return x.grad[0]


# =============================================================================
# ReLU
# =============================================================================
class TestReLU(ElementwiseContract, unittest.TestCase):
    make = staticmethod(ReLU)
    ref, dref = staticmethod(relu_ref), staticmethod(relu_dref)

    def test_known_values(self):
        np.testing.assert_array_equal(
            self.layer(T([-1.0, 0.0, 1.0])).data, [0.0, 0.0, 1.0]
        )

    def test_gradient_is_zero_at_zero_and_below_one_above(self):
        x = T([-2.0, -0.5, 0.0, 0.5, 2.0], True)
        self.layer(x).sum().backward()
        np.testing.assert_array_equal(x.grad, [0.0, 0.0, 0.0, 1.0, 1.0])

    def test_positive_inputs_pass_through_exactly(self):
        x = np.abs(self.sample()) + 0.1
        np.testing.assert_array_equal(self.layer(T(x)).data, x)

    def test_is_idempotent(self):
        x = T(self.sample())
        np.testing.assert_array_equal(
            self.layer(self.layer(x)).data, self.layer(x).data
        )

    def test_upstream_gradient_is_masked_not_scaled(self):
        x0 = self.sample()
        w = self.rng.standard_normal(x0.shape).astype(np.float32)
        np.testing.assert_allclose(
            analytic_grad(self.layer, x0, w), w * (x0 > 0), rtol=1e-6
        )

    def test_all_negative_input_gives_dead_gradient(self):
        x0 = -np.abs(self.sample()) - 0.1
        w = np.ones_like(x0)
        np.testing.assert_array_equal(analytic_grad(self.layer, x0, w), 0.0)

    def test_infinite_upstream_gradient_does_not_create_nan(self):
        x = T([-1.0, 2.0], True)
        out = self.layer(x)
        out.grad = np.array([np.inf, 1.0], dtype=np.float32)  # inf hits a dead unit
        out._backward()
        np.testing.assert_array_equal(x.grad, [0.0, 1.0])  # inf * 0 must not become NaN


# =============================================================================
# Tanh
# =============================================================================
class TestTanh(ElementwiseContract, unittest.TestCase):
    make = staticmethod(Tanh)
    ref, dref = staticmethod(tanh_ref), staticmethod(tanh_dref)

    def test_known_values(self):
        out = self.layer(T([-1.0, 0.0, 1.0])).data
        np.testing.assert_allclose(
            out, [-0.7615942, 0.0, 0.7615942], rtol=1e-6, atol=1e-7
        )

    def test_is_odd(self):
        np.testing.assert_array_equal(
            self.layer(T(-GRID)).data, -self.layer(T(GRID)).data
        )

    def test_bounded_by_one(self):
        self.assertTrue((np.abs(self.layer(T(EXTREME)).data) <= 1.0).all())

    def test_infinite_inputs_saturate(self):
        with no_numeric_warnings():
            np.testing.assert_array_equal(
                self.layer(T([np.inf, -np.inf])).data, [1.0, -1.0]
            )

    def test_gradient_is_one_at_zero_and_one_minus_y_squared(self):
        x = T([0.0, 0.7, -1.3], True)
        out = self.layer(x)
        out.sum().backward()
        self.assertEqual(float(x.grad[0]), 1.0)
        np.testing.assert_allclose(x.grad, 1.0 - out.data**2, rtol=1e-6)

    def test_deeply_saturated_gradient_is_tiny_not_negative(self):
        x = T([20.0, -20.0], True)
        self.layer(x).sum().backward()
        self.assertTrue((x.grad >= 0).all() and (x.grad < 1e-6).all())


# =============================================================================
# GELU (all three modes get the full element-wise contract)
# =============================================================================
class TestGELUExact(ElementwiseContract, unittest.TestCase):
    make = staticmethod(GELU)
    ref, dref = staticmethod(gelu_ref), staticmethod(gelu_dref)

    def test_default_mode_is_exact(self):
        self.assertEqual(GELU().approximate, "none")

    def test_known_values(self):
        out = self.layer(T([-1.0, 0.0, 1.0])).data
        np.testing.assert_allclose(
            out, [-0.15865526, 0.0, 0.8413447], rtol=1e-5, atol=1e-7
        )

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            GELU("relu")

    def test_gradient_at_zero_is_one_half_in_every_mode(self):
        for mode in ("none", "tanh", "sigmoid"):
            x = T([0.0], True)
            GELU(mode)(x).sum().backward()
            self.assertAlmostEqual(float(x.grad[0]), 0.5, places=6, msg=mode)

    def test_gradient_is_negative_left_of_the_minimum_in_every_mode(self):
        for mode in ("none", "tanh", "sigmoid"):
            x = T([-1.0], True)
            GELU(mode)(x).sum().backward()
            self.assertLess(float(x.grad[0]), 0.0, msg=mode)

    def test_is_non_monotonic_with_minimum_near_minus_0_75(self):
        xs = np.linspace(-3, 0, 301).astype(np.float32)
        out = self.layer(T(xs)).data
        self.assertTrue(-0.171 < float(out.min()) < -0.169)
        self.assertTrue(-0.80 < float(xs[out.argmin()]) < -0.70)

    def test_asymptotics(self):
        out = self.layer(T([10.0, -10.0])).data
        self.assertAlmostEqual(float(out[0]), 10.0, places=4)
        self.assertAlmostEqual(float(out[1]), 0.0, places=4)

    def test_approximation_quality_ordering(self):
        xs = T(np.linspace(-6, 6, 2001))
        exact = GELU("none")(xs).data
        err_tanh = np.abs(GELU("tanh")(xs).data - exact).max()
        err_sig = np.abs(GELU("sigmoid")(xs).data - exact).max()
        self.assertLess(err_tanh, 1e-3)
        self.assertLess(err_sig, 0.03)
        self.assertLess(err_tanh, err_sig)


class TestGELUTanh(ElementwiseContract, unittest.TestCase):
    make = staticmethod(lambda: GELU("tanh"))
    ref, dref = staticmethod(gelu_tanh_ref), staticmethod(gelu_tanh_dref)


class TestGELUSigmoid(ElementwiseContract, unittest.TestCase):
    make = staticmethod(lambda: GELU("sigmoid"))
    ref, dref = staticmethod(gelu_sig_ref), staticmethod(gelu_sig_dref)


# =============================================================================
# Softmax
# =============================================================================
class TestSoftmax(LayerContract, unittest.TestCase):
    make = staticmethod(lambda: Softmax(dim=-1))
    shapes = [(1,), (5,), (3, 4), (2, 3, 4)]  # needs at least one axis

    def test_matches_reference_and_sums_to_one_for_every_dim(self):
        x0 = self.sample((2, 3, 4))
        for dim in (-3, -2, -1, 0, 1, 2):
            out = Softmax(dim=dim)(T(x0)).data
            np.testing.assert_allclose(
                out,
                softmax_ref(x0.astype(np.float64), dim),
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"dim={dim}",
            )
            np.testing.assert_allclose(
                out.sum(axis=dim), 1.0, atol=1e-6, err_msg=f"dim={dim}"
            )
            self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_negative_and_positive_dims_agree(self):
        x = T(self.sample((2, 3, 4)))
        np.testing.assert_array_equal(Softmax(dim=-1)(x).data, Softmax(dim=2)(x).data)
        np.testing.assert_array_equal(Softmax(dim=-2)(x).data, Softmax(dim=1)(x).data)

    def test_gradcheck_for_every_dim(self):
        for dim in (0, 1, -1):
            assert_gradcheck(Softmax(dim=dim), self.sample((3, 4)), msg=f"dim={dim}")
        assert_gradcheck(Softmax(dim=1), self.sample((2, 3, 4)), msg="3-D, dim=1")

    def test_default_dim_is_the_last_axis(self):
        x1 = T(self.sample((5,)))
        np.testing.assert_allclose(
            Softmax()(x1).data.sum(), 1.0, atol=1e-6
        )  # 1-D must work
        x3 = T(self.sample((2, 3, 4)))
        np.testing.assert_array_equal(Softmax()(x3).data, Softmax(dim=-1)(x3).data)

    def test_invalid_dim_raises(self):
        with self.assertRaises((ValueError, IndexError)):
            Softmax(dim=5)(T(np.ones((2, 3))))

    def test_shift_invariance(self):
        x0 = self.sample((3, 4))
        for c in (-100.0, 7.5, 100.0):
            np.testing.assert_allclose(
                self.layer(T(x0 + c)).data, self.layer(T(x0)).data, atol=1e-4
            )

    def test_huge_logits_do_not_overflow(self):
        with no_numeric_warnings():
            out = self.layer(T([[1000.0, 1001.0, 1002.0]])).data
        np.testing.assert_allclose(
            out, softmax_ref(np.array([[0.0, 1.0, 2.0]])), rtol=1e-6
        )

    def test_uniform_logits_give_uniform_probabilities(self):
        np.testing.assert_allclose(
            self.layer(T(np.full((2, 5), 3.0))).data, 0.2, rtol=1e-6
        )

    def test_permutation_equivariance(self):
        x0 = self.sample((6,))
        perm = self.rng.permutation(6)
        np.testing.assert_allclose(
            self.layer(T(x0[perm])).data, self.layer(T(x0)).data[perm], rtol=1e-6
        )

    def test_constant_upstream_gradient_gives_zero_input_gradient(self):
        g = analytic_grad(
            self.layer, self.sample((3, 4)), np.ones((3, 4), dtype=np.float32)
        )
        np.testing.assert_allclose(g, 0.0, atol=1e-6)  # outputs always sum to 1

    def test_jacobian_is_diag_s_minus_s_outer_s(self):
        x0 = self.sample((5,))
        s = softmax_ref(x0.astype(np.float64))
        J = np.zeros((5, 5))
        for i in range(5):
            e = np.zeros(5, dtype=np.float32)
            e[i] = 1.0
            J[i] = analytic_grad(self.layer, x0, e)  # row i = d out_i / d x
        np.testing.assert_allclose(J, np.diag(s) - np.outer(s, s), atol=1e-6)

    def test_cross_entropy_gradient_identity(self):
        # with loss = -log p[y], d loss / d logits must equal p - onehot(y)
        x0 = self.sample((4, 5))
        y = np.array([0, 3, 1, 4])
        s = softmax_ref(x0.astype(np.float64))
        w = np.zeros_like(x0)
        w[np.arange(4), y] = -1.0 / s[np.arange(4), y]  # d(-log p_y)/d p_y
        expected = s - np.eye(5)[y]
        np.testing.assert_allclose(
            analytic_grad(self.layer, x0, w), expected, atol=1e-5
        )

    def test_rows_are_independent(self):
        x0 = self.sample((4, 5))
        w = np.zeros_like(x0)
        w[1] = self.rng.standard_normal(5)
        g = analytic_grad(self.layer, x0, w)
        np.testing.assert_array_equal(g[[0, 2, 3]], 0.0)
        self.assertGreater(np.abs(g[1]).max(), 0.0)

    def test_causal_mask_with_minus_inf(self):
        x0 = np.array([[0.5, -np.inf, -np.inf], [0.1, 0.9, -np.inf]], dtype=np.float32)
        w = self.rng.standard_normal(x0.shape).astype(np.float32)
        with no_numeric_warnings():
            x = T(x0, True)
            out = self.layer(x)
            (out * T(w)).sum().backward()
        masked = np.isinf(x0)
        np.testing.assert_array_equal(out.data[masked], 0.0)
        np.testing.assert_allclose(out.data.sum(axis=-1), 1.0, atol=1e-6)
        self.assertTrue(np.isfinite(x.grad).all())
        np.testing.assert_array_equal(x.grad[masked], 0.0)
        s = out.data.astype(np.float64)
        np.testing.assert_allclose(
            x.grad, s * (w - (w * s).sum(-1, keepdims=True)), atol=1e-6
        )

    def test_fully_masked_row_is_nan_but_other_rows_are_fine(self):
        # same behaviour as PyTorch: an all -inf row has no valid distribution
        x0 = np.array([[-np.inf, -np.inf], [1.0, 2.0]], dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self.layer(T(x0)).data
        self.assertTrue(np.isnan(out[0]).all())
        np.testing.assert_allclose(out[1].sum(), 1.0, atol=1e-6)

    def test_nan_only_poisons_its_own_row(self):
        x0 = np.array([[np.nan, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32)
        out = self.layer(T(x0)).data
        self.assertTrue(np.isnan(out[0]).all())
        np.testing.assert_allclose(out[1].sum(), 1.0, atol=1e-6)


# =============================================================================
# Integration: the layers must work inside a real training loop
# =============================================================================
def tlog(x):
    """log as a tiny autograd op, so a cross-entropy loss can be built in the test."""
    eps = 1e-12
    out = Tensor(
        np.log(np.maximum(x.data, eps)),
        requires_grad=x.requires_grad,
        _parents=(x,) if x.requires_grad else (),
        dtype=np.float32,
        device=x.device,
    )

    def _backward():
        if x.requires_grad and out.grad is not None:
            x._add_grad(out.grad / np.maximum(x.data, eps))

    out._backward = _backward
    return out


def check_network_gradients(test, loss_fn, params, eps=3e-3, rtol=2e-2, atol=1e-3):
    """Compare backward() on a full network loss with finite differences, parameter by parameter."""
    for p in params:
        p.grad[...] = 0.0
    loss_fn().backward()
    analytic = [p.grad.copy() for p in params]
    for p, a in zip(params, analytic):
        num = np.zeros(p.data.shape, dtype=np.float64)
        for idx in np.ndindex(*p.data.shape):
            orig = float(p.data[idx])
            p.data[idx] = orig + eps
            hi = float(loss_fn().data)
            p.data[idx] = orig - eps
            lo = float(loss_fn().data)
            p.data[idx] = orig
            num[idx] = (hi - lo) / (
                float(np.float32(orig + eps)) - float(np.float32(orig - eps))
            )
        np.testing.assert_allclose(
            a, num, rtol=rtol, atol=atol, err_msg="network parameter gradient"
        )


class MLPRegressionContract:
    make_act = None

    def test_mlp_fits_a_nonlinear_target(self):
        rng = np.random.default_rng(0)
        X = T(rng.uniform(-1, 1, (64, 2)))
        y = T(
            (X.data[:, 0] * X.data[:, 1]).reshape(-1, 1)
        )  # XOR-like, needs a nonlinearity
        W1, b1 = T(rng.standard_normal((2, 16)) * 0.5, True), T(np.zeros(16), True)
        W2, b2 = T(rng.standard_normal((16, 1)) * 0.5, True), T(np.zeros(1), True)
        params, act, losses = [W1, b1, W2, b2], self.make_act(), []
        for step in range(300):
            pred = act(X @ W1 + b1) @ W2 + b2
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            if step == 0:
                for name, p in zip(("W1", "b1", "W2", "b2"), params):
                    self.assertGreater(
                        float(np.abs(p.grad).sum()), 0.0, f"no gradient reached {name}"
                    )
            losses.append(float(loss.data))
            sgd(params, 0.1)
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(
            losses[-1], 0.5 * losses[0], f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_network_gradients_match_finite_differences(self):
        rng = np.random.default_rng(3)
        X, y = T(rng.standard_normal((5, 2))), T(rng.standard_normal((5, 1)))
        W1, b1 = (
            T(rng.standard_normal((2, 4)) * 0.7, True),
            T(rng.standard_normal(4) * 0.3, True),
        )
        W2, b2 = T(rng.standard_normal((4, 1)) * 0.7, True), T(np.zeros(1), True)
        act = self.make_act()
        loss_fn = lambda: ((act(X @ W1 + b1) @ W2 + b2 - y) ** 2).mean()
        check_network_gradients(self, loss_fn, [W1, b1, W2, b2])


class TestMLPSigmoid(MLPRegressionContract, unittest.TestCase):
    make_act = staticmethod(Sigmoid)


class TestMLPTanh(MLPRegressionContract, unittest.TestCase):
    make_act = staticmethod(Tanh)


class TestMLPReLU(MLPRegressionContract, unittest.TestCase):
    make_act = staticmethod(ReLU)


class TestMLPGELU(MLPRegressionContract, unittest.TestCase):
    make_act = staticmethod(GELU)


class TestMLPGELUTanh(MLPRegressionContract, unittest.TestCase):
    make_act = staticmethod(lambda: GELU("tanh"))


class SoftmaxClassifierContract:
    make_hidden = None

    def test_classifier_with_softmax_cross_entropy_learns(self):
        rng = np.random.default_rng(1)
        labels = np.repeat([0, 1, 2], 40)
        centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])
        X = T(centers[labels] + rng.standard_normal((120, 2)) * 0.6)
        Y = T(np.eye(3)[labels])
        W1, b1 = T(rng.standard_normal((2, 16)) * 0.5, True), T(np.zeros(16), True)
        W2, b2 = T(rng.standard_normal((16, 3)) * 0.5, True), T(np.zeros(3), True)
        params, hidden, softmax, losses = (
            [W1, b1, W2, b2],
            self.make_hidden(),
            Softmax(dim=-1),
            [],
        )
        for step in range(150):
            probs = softmax(hidden(X @ W1 + b1) @ W2 + b2)
            loss = -(Y * tlog(probs)).sum() * (1.0 / 120)
            loss.backward()
            if step == 0:
                for name, p in zip(("W1", "b1", "W2", "b2"), params):
                    self.assertGreater(
                        float(np.abs(p.grad).sum()), 0.0, f"no gradient reached {name}"
                    )
            losses.append(float(loss.data))
            sgd(params, 0.3)
        accuracy = float((probs.data.argmax(axis=1) == labels).mean())
        self.assertTrue(np.isfinite(losses).all())
        self.assertLess(
            losses[-1], 0.5 * losses[0], f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
        self.assertGreater(accuracy, 0.95)

    def test_network_gradients_match_finite_differences(self):
        rng = np.random.default_rng(4)
        X = T(rng.standard_normal((6, 2)))
        Y = T(np.eye(3)[[0, 1, 2, 0, 1, 2]])
        W1, b1 = (
            T(rng.standard_normal((2, 4)) * 0.7, True),
            T(rng.standard_normal(4) * 0.3, True),
        )
        W2, b2 = T(rng.standard_normal((4, 3)) * 0.7, True), T(np.zeros(3), True)
        hidden, softmax = self.make_hidden(), Softmax(dim=-1)
        loss_fn = lambda: (
            -(Y * tlog(softmax(hidden(X @ W1 + b1) @ W2 + b2))).sum() * (1.0 / 6)
        )
        check_network_gradients(self, loss_fn, [W1, b1, W2, b2])


class TestClassifierReLU(SoftmaxClassifierContract, unittest.TestCase):
    make_hidden = staticmethod(ReLU)


class TestClassifierTanh(SoftmaxClassifierContract, unittest.TestCase):
    make_hidden = staticmethod(Tanh)


class TestClassifierGELU(SoftmaxClassifierContract, unittest.TestCase):
    make_hidden = staticmethod(GELU)


class TestClassifierSigmoid(SoftmaxClassifierContract, unittest.TestCase):
    make_hidden = staticmethod(Sigmoid)


class TestStackedActivations(unittest.TestCase):
    def test_gradcheck_through_a_stack_of_every_activation(self):
        relu, gelu, tanh, sig, soft = ReLU(), GELU(), Tanh(), Sigmoid(), Softmax(dim=-1)
        fn = lambda x: soft(sig(tanh(gelu(relu(x) * 1.5 - 0.5))))
        rng = np.random.default_rng(0)
        x0 = (rng.standard_normal((3, 4)) * 1.5).astype(np.float32)
        x0 = np.where(np.abs(x0) < 0.05, x0 + 0.3, x0).astype(np.float32)
        assert_gradcheck(fn, x0, eps=3e-3, atol=5e-4, msg="stacked chain rule")


if __name__ == "__main__":
    unittest.main(verbosity=2)
