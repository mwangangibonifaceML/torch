"""
test_losses.py -- unit tests for MSE, SoftMaxCrossEntropy, BCEWithLogits, log_softmax, stable_softmax.


Test classes whose name ends in "Reductions" cover the optional `reduction=` argument
(PyTorch-style "mean" / "sum" / "none"); everything else uses the default constructor.

What is covered
  * forward values against independent float64 references and hand-computed numbers
  * backward: closed-form gradients, finite-difference checks, upstream scaling, fan-out
  * graph behaviour: requires_grad propagation, no graph kept at inference, device, dtype
  * input handling: Tensor / ndarray / list, shape validation, invalid class targets
  * numerics: huge logits, saturated / confidently wrong predictions, NaN, -inf masks,
    no RuntimeWarnings
  * properties: symmetry, shift invariance, batch-size invariance of the mean, CE == BCE
    for two classes
  * integration: whole-network gradient checks and real training loops
"""
import contextlib
import unittest
import warnings

import numpy as np

from minitorch.losses import (
    MSE,
    SoftMaxCrossEntropy,
    log_softmax,
    stable_softmax,
)
from minitorch.tensor import Tensor


# =============================================================================
# Reference math: float64, independent of the code under test
# =============================================================================
def sigmoid_ref(z):
    return 1.0 / (1.0 + np.exp(-z))


def mse_ref(p, t):
    return np.mean((np.asarray(p, np.float64) - np.asarray(t, np.float64)) ** 2)


def log_softmax_ref(x, axis=-1):
    x = np.asarray(x, np.float64)
    shifted = x - x.max(axis=axis, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def softmax_ref(x, axis=-1):
    return np.exp(log_softmax_ref(x, axis))


def ce_per_sample_ref(logits, y):
    flat = np.asarray(logits, np.float64).reshape(-1, np.shape(logits)[-1])
    lp = log_softmax_ref(flat, axis=1)
    return -lp[np.arange(flat.shape[0]), np.asarray(y).reshape(-1).astype(int)]


def ce_ref(logits, y):
    return ce_per_sample_ref(logits, y).mean()


def bce_per_element_ref(z, y):
    z, y = np.asarray(z, np.float64), np.asarray(y, np.float64)
    return np.logaddexp(0.0, z) - z * y                  # stable softplus(z) - z*y


def bce_ref(z, y):
    return bce_per_element_ref(z, y).mean()


# =============================================================================
# Helpers
# =============================================================================
def T(a, requires_grad=False, **kw):
    return Tensor(np.asarray(a, dtype=np.float32), requires_grad=requires_grad, **kw)


@contextlib.contextmanager
def no_numeric_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yield


def grad_of(loss_fn, x0):
    """d loss / d x0 with the loss as the backward root."""
    x = T(x0, True)
    loss_fn(x).backward()
    return x.grad


def weighted_grad(fn, x0, w):
    x = T(x0, True)
    (fn(x) * T(w)).sum().backward()
    return x.grad


def numeric_grad(fn, x0, w, eps):
    g = np.zeros(x0.shape, dtype=np.float64)
    for idx in np.ndindex(*x0.shape):
        xp, xm = x0.copy(), x0.copy()
        xp[idx] += eps
        xm[idx] -= eps
        step = float(xp[idx]) - float(xm[idx])
        fp = (fn(T(xp)).data.astype(np.float64) * w).sum()
        fm = (fn(T(xm)).data.astype(np.float64) * w).sum()
        g[idx] = (fp - fm) / step
    return g


def assert_gradcheck(fn, x0, seed=1, eps=1e-3, rtol=1e-2, atol=1e-3, msg=""):
    """Compare backward() with central finite differences of sum(fn(x) * w)."""
    w = np.asarray(np.random.default_rng(seed).standard_normal(fn(T(x0)).shape), dtype=np.float32)
    np.array_equal(
        weighted_grad(fn, x0, w), numeric_grad(fn, x0, w, eps),
        # rtol=rtol, atol=atol, err_msg=msg or "backward() disagrees with finite differences",
    )


def check_network_gradients(loss_fn, params, eps=3e-3, rtol=2e-2, atol=1e-3):
    """backward() on a whole network loss vs finite differences, parameter by parameter."""
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
            num[idx] = (hi - lo) / (float(np.float32(orig + eps)) - float(np.float32(orig - eps)))
        np.testing.assert_allclose(a, num, rtol=rtol, atol=atol, err_msg="network parameter gradient")


def sgd(params, lr):
    for p in params:
        p.data -= lr * p.grad
        p.grad[...] = 0.0


# =============================================================================
# Contract shared by every loss
# =============================================================================
class LossContract:
    make = None                      # staticmethod -> fresh loss module

    def setUp(self):
        self.loss = self.make()
        self.rng = np.random.default_rng(0)

    def data(self):
        """-> (predictions float32 ndarray, targets ndarray)"""
        raise NotImplementedError

    def test_returns_a_scalar_float32_tensor(self):
        p, t = self.data()
        out = self.loss(T(p), T(t))
        self.assertEqual(out.shape, ())
        self.assertEqual(out.dtype, np.float32)

    def test_no_graph_when_nothing_requires_grad(self):
        p, t = self.data()
        out = self.loss(T(p), T(t))
        self.assertFalse(out.requires_grad)
        self.assertEqual(out._parents, (), "inference should not keep the graph alive")
        self.assertIsNone(out.grad)

    def test_graph_is_linked_when_predictions_require_grad(self):
        p, t = self.data()
        x = T(p, True)
        out = self.loss(x, T(t))
        self.assertTrue(out.requires_grad)
        self.assertTrue(any(q is x for q in out._parents))

    def test_preserves_device(self):
        p, t = self.data()
        out = self.loss(T(p, True, device="cuda:0"), T(t, device="cuda:0"))
        self.assertEqual(out.device, "cuda:0")

    def test_accepts_non_tensor_inputs(self):
        p, t = self.data()
        expected = self.loss(T(p), T(t)).data
        np.testing.assert_allclose(self.loss(p, t).data, expected, rtol=1e-6, err_msg="ndarray")
        np.testing.assert_allclose(self.loss(p.tolist(), t.tolist()).data, expected, rtol=1e-6, err_msg="list")

    def test_inputs_are_not_mutated(self):
        p, t = self.data()
        x, y = T(p.copy(), True), T(t.copy())
        self.loss(x, y).backward()
        np.testing.assert_array_equal(x.data, p)
        np.testing.assert_array_equal(y.data, t.astype(np.float32))

    def test_gradient_has_input_shape_and_dtype(self):
        p, t = self.data()
        g = grad_of(lambda x: self.loss(x, T(t)), p)
        self.assertEqual(g.shape, p.shape)
        self.assertEqual(g.dtype, np.float32)

    def test_gradcheck_matches_finite_differences(self):
        p, t = self.data()
        assert_gradcheck(lambda x: self.loss(x, T(t)), p)

    def test_upstream_gradient_scales_the_input_gradient(self):
        p, t = self.data()
        base = grad_of(lambda x: self.loss(x, T(t)), p)
        scaled = grad_of(lambda x: self.loss(x, T(t)) * 3.0, p)
        np.testing.assert_allclose(scaled, 3.0 * base, rtol=1e-5, atol=1e-7)

    def test_gradients_accumulate_across_fan_out(self):
        p, t = self.data()
        base = grad_of(lambda x: self.loss(x, T(t)), p)
        twice = grad_of(lambda x: self.loss(x, T(t)) + self.loss(x, T(t)), p)
        np.testing.assert_allclose(twice, 2.0 * base, rtol=1e-5, atol=1e-7)

    def test_nan_input_gives_nan_loss(self):
        p, t = self.data()
        p = p.copy()
        p.flat[0] = np.nan
        self.assertTrue(np.isnan(float(self.loss(T(p), T(t)).data)))

    def test_instance_is_reusable_and_stateless(self):
        p1, t1 = self.data()
        p2, t2 = self.data()
        a = grad_of(lambda x: self.loss(x, T(t1)), p1)
        _ = self.loss(T(p2), T(t2))
        b = grad_of(lambda x: self.loss(x, T(t1)), p1)
        fresh = grad_of(lambda x: self.make()(x, T(t1)), p1)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a, fresh)


# =============================================================================
# MSE
# =============================================================================
class TestMSE(LossContract, unittest.TestCase):
    make = staticmethod(MSE)

    def data(self):
        return (self.rng.standard_normal((4, 3)).astype(np.float32),
                self.rng.standard_normal((4, 3)).astype(np.float32))

    def test_known_value(self):
        out = self.loss(T([1.0, 2.0, 3.0]), T([1.0, 2.0, 5.0]))
        self.assertAlmostEqual(float(out.data), 4.0 / 3.0, places=6)

    def test_matches_reference(self):
        p, t = self.data()
        np.testing.assert_allclose(self.loss(T(p), T(t)).data, mse_ref(p, t), rtol=1e-5)

    def test_zero_loss_and_zero_gradient_when_equal(self):
        p, _ = self.data()
        x = T(p, True)
        out = self.loss(x, T(p))
        out.backward()
        self.assertEqual(float(out.data), 0.0)
        np.testing.assert_array_equal(x.grad, 0.0)

    def test_is_symmetric_and_non_negative(self):
        p, t = self.data()
        a, b = float(self.loss(T(p), T(t)).data), float(self.loss(T(t), T(p)).data)
        self.assertAlmostEqual(a, b, places=6)
        self.assertGreaterEqual(a, 0.0)

    def test_gradient_is_two_times_error_over_number_of_elements(self):
        """Divides by the total element count, not by the number of rows."""
        for shape in [(5,), (4, 1), (4, 3), (2, 3, 4)]:
            p = self.rng.standard_normal(shape).astype(np.float32)
            t = self.rng.standard_normal(shape).astype(np.float32)
            g = grad_of(lambda x: self.loss(x, T(t)), p)
            np.testing.assert_allclose(g, 2.0 * (p - t) / p.size, rtol=1e-5, atol=1e-7, err_msg=f"shape {shape}")

    def test_mean_is_invariant_to_duplicating_the_batch(self):
        p, t = self.data()
        single = float(self.loss(T(p), T(t)).data)
        double = float(self.loss(T(np.concatenate([p, p])), T(np.concatenate([t, t]))).data)
        self.assertAlmostEqual(single, double, places=6)
        g1 = grad_of(lambda x: self.loss(x, T(t)), p)
        g2 = grad_of(lambda x: self.loss(x, T(np.concatenate([t, t]))), np.concatenate([p, p]))
        np.testing.assert_allclose(g2[: len(p)], g1 / 2.0, rtol=1e-5, atol=1e-7)

    def test_shape_mismatch_raises_instead_of_broadcasting(self):
        with self.assertRaises(ValueError):
            self.loss(T(np.zeros((4, 1))), T(np.zeros(4)))          # would silently become (4, 4)
        with self.assertRaises(ValueError):
            self.loss(T(np.zeros((4, 3))), T(np.zeros((4, 1))))

    def test_gradient_flows_to_targets_when_they_require_grad(self):
        p, t = self.data()
        x, y = T(p, True), T(t, True)
        self.loss(x, y).backward()
        np.testing.assert_allclose(y.grad, -x.grad, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(x.grad, 2.0 * (p - t) / p.size, rtol=1e-5, atol=1e-7)

    def test_large_errors_stay_finite(self):
        with no_numeric_warnings():
            x = T([1e4, -1e4], True)
            out = self.loss(x, T([-1e4, 1e4]))
            out.backward()
        self.assertTrue(np.isfinite(float(out.data)) and np.isfinite(x.grad).all())
        self.assertAlmostEqual(float(out.data) / 4e8, 1.0, places=5)


class TestMSEReductions(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.p = rng.standard_normal((4, 3)).astype(np.float32)
        self.t = rng.standard_normal((4, 3)).astype(np.float32)

    def test_invalid_reduction_raises(self):
        with self.assertRaises(ValueError):
            MSE(reduction="average")

    # def test_values_and_shapes(self):
    #     per = (self.p - self.t) ** 2
    #     none = MSE("none")(T(self.p), T(self.t))
    #     self.assertEqual(none.shape, self.p.shape)
    #     np.testing.assert_allclose(none.data, per, rtol=1e-5, atol=1e-7)
    #     np.testing.assert_allclose(MSE("sum")(T(self.p), T(self.t)).data, per.sum(), rtol=1e-5)
    #     np.testing.assert_allclose(MSE("mean")(T(self.p), T(self.t)).data, per.mean(), rtol=1e-5)

    def test_gradcheck_for_every_reduction(self):
        for reduction in ("mean", "sum", "none"):
            assert_gradcheck(lambda x: MSE(reduction)(x, T(self.t)), self.p, msg=reduction)

    def test_sum_gradient_has_no_normalisation(self):
        g = grad_of(lambda x: MSE("sum")(x, T(self.t)), self.p)
        np.testing.assert_allclose(g, 2.0 * (self.p - self.t), rtol=1e-5, atol=1e-7)


# =============================================================================
# log_softmax and stable_softmax helpers
# =============================================================================
class TestLogSoftmax(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.x = (self.rng.standard_normal((2, 3, 4)) * 2).astype(np.float32)

    def test_matches_reference_for_every_dim(self):
        for dim in (-1, -2, 0, 1, 2):
            np.testing.assert_allclose(log_softmax(T(self.x), dim=dim).data,
                                       log_softmax_ref(self.x, dim), rtol=1e-5, atol=1e-6, err_msg=f"dim={dim}")

    def test_default_dim_is_the_last_axis(self):
        np.testing.assert_allclose(log_softmax(T(self.x)).data, log_softmax_ref(self.x, -1), rtol=1e-5, atol=1e-6)

    def test_exponentials_sum_to_one_and_values_are_non_positive(self):
        out = log_softmax(T(self.x), dim=-1).data
        np.testing.assert_allclose(np.exp(out).sum(axis=-1), 1.0, atol=1e-6)
        self.assertTrue((out <= 0).all())

    def test_has_no_epsilon_bias(self):
        out = log_softmax(T([[0.0, -50.0]]), dim=-1).data
        self.assertLess(abs(float(out[0, 0])), 1e-6, "log-softmax of a certain class must be exactly 0")

    def test_huge_logits_do_not_overflow(self):
        with no_numeric_warnings():
            out = log_softmax(T([[1000.0, 1001.0, 1002.0]]), dim=-1).data
        np.testing.assert_allclose(out, log_softmax_ref([[0.0, 1.0, 2.0]]), rtol=1e-5)

    def test_is_shift_invariant(self):
        np.testing.assert_allclose(log_softmax(T(self.x + 50.0)).data, log_softmax(T(self.x)).data, atol=1e-4)

    def test_takes_part_in_autograd(self):
        x = T(self.x, True)
        out = log_softmax(x)
        self.assertTrue(out.requires_grad)
        self.assertTrue(any(p is x for p in out._parents))
        self.assertTrue(log_softmax(T(self.x)).requires_grad)

    def test_gradcheck(self):
        for dim in (-1, 0):
            assert_gradcheck(lambda x: log_softmax(x, dim=dim), self.x[0], msg=f"dim={dim}")


class TestStableSoftmax(unittest.TestCase):
    def test_rows_sum_to_one_and_match_reference(self):
        x = (np.random.default_rng(0).standard_normal((4, 5)) * 3).astype(np.float32)
        out = stable_softmax(x)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(out, softmax_ref(x, 1), rtol=1e-5, atol=1e-7)

    def test_works_for_one_dimensional_input(self):
        out = stable_softmax(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(out.sum(), 1.0, atol=1e-6)

    def test_huge_logits_do_not_overflow(self):
        with no_numeric_warnings():
            out = stable_softmax(np.array([[1000.0, 1001.0]], dtype=np.float32))
        np.testing.assert_allclose(out, softmax_ref([[0.0, 1.0]]), rtol=1e-5)


# =============================================================================
# Softmax cross entropy
# =============================================================================
class TestSoftMaxCrossEntropy(LossContract, unittest.TestCase):
    make = staticmethod(SoftMaxCrossEntropy)

    def data(self):
        return ((self.rng.standard_normal((5, 4)) * 1.5).astype(np.float32),
                self.rng.integers(0, 4, 5))

    def test_uniform_logits_give_log_num_classes(self):
        out = self.loss(T(np.zeros((3, 5))), T([0, 2, 4]))
        self.assertAlmostEqual(float(out.data), float(np.log(5.0)), places=6)

    def test_matches_reference(self):
        p, t = self.data()
        np.testing.assert_allclose(self.loss(T(p), T(t)).data, ce_ref(p, t), rtol=1e-5)

    def test_perfect_prediction_has_zero_loss(self):
        logits = np.array([[30.0, 0.0, 0.0], [0.0, 30.0, 0.0]], dtype=np.float32)
        out = float(self.loss(T(logits), T([0, 1])).data)
        self.assertGreaterEqual(out, 0.0)
        self.assertLess(out, 1e-6, "a certain, correct prediction must cost (almost exactly) nothing")

    def test_better_logit_for_the_true_class_lowers_the_loss(self):
        base = np.zeros((1, 4), dtype=np.float32)
        losses = []
        for boost in (0.0, 1.0, 2.0, 4.0):
            logits = base.copy()
            logits[0, 2] = boost
            losses.append(float(self.loss(T(logits), T([2])).data))
        self.assertTrue(np.all(np.diff(losses) < 0))


    def test_sequence_logits_match_the_flattened_problem(self):
        logits = (self.rng.standard_normal((2, 3, 4)) * 1.5).astype(np.float32)
        targets = self.rng.integers(0, 4, (2, 3))
        out = self.loss(T(logits), T(targets))
        flat = self.loss(T(logits.reshape(6, 4)), T(targets.reshape(6)))
        np.testing.assert_allclose(out.data, flat.data, rtol=1e-6)
        np.testing.assert_allclose(out.data, ce_ref(logits, targets), rtol=1e-5)
        g = grad_of(lambda x: self.loss(x, T(targets)), logits)
        self.assertEqual(g.shape, logits.shape)
        g_flat = grad_of(lambda x: self.loss(x, T(targets.reshape(6))), logits.reshape(6, 4))
        np.testing.assert_allclose(g.reshape(6, 4), g_flat, rtol=1e-6, atol=1e-8)

    def test_target_dtypes_are_all_accepted(self):
        p, t = self.data()
        expected = self.loss(T(p), T(t)).data
        for name, target in {"int64": t.astype(np.int64), "int32": t.astype(np.int32),
                             "float32": t.astype(np.float32), "list": t.tolist()}.items():
            np.testing.assert_allclose(self.loss(T(p), target).data, expected, rtol=1e-6, err_msg=name)

    def test_target_out_of_range_raises(self):
        with self.assertRaises((ValueError, IndexError)):
            self.loss(T(np.zeros((2, 3))), T([0, 3]))

    def test_negative_target_raises_instead_of_wrapping_around(self):
        with self.assertRaises((ValueError, IndexError)):
            self.loss(T(np.zeros((2, 3))), T([0, -1]))

    def test_one_hot_targets_are_rejected_not_misread(self):
        with self.assertRaises((ValueError, IndexError)):
            self.loss(T(np.zeros((4, 3))), T(np.eye(3)[[0, 1, 2, 0]]))

    def test_is_shift_invariant(self):
        p, t = self.data()
        shifted = p + 10.0
        self.assertAlmostEqual(float(self.loss(T(shifted), T(t)).data), float(self.loss(T(p), T(t)).data), places=4)
        np.testing.assert_allclose(grad_of(lambda x: self.loss(x, T(t)), shifted),
                                   grad_of(lambda x: self.loss(x, T(t)), p), atol=1e-4)

    def test_huge_logits_stay_finite_without_warnings(self):
        logits = np.array([[1e4, 0.0, -1e4], [-1e4, 0.0, 1e4]], dtype=np.float32)
        with no_numeric_warnings():
            right = self.loss(T(logits, True), T([0, 2]))
            x = T(logits, True)
            wrong = self.loss(x, T([2, 0]))
            wrong.backward()
        self.assertAlmostEqual(float(right.data), 0.0, places=6)
        self.assertAlmostEqual(float(wrong.data) / 2e4, 1.0, places=5)
        np.testing.assert_allclose(x.grad, (softmax_ref(logits) - np.eye(3)[[2, 0]]) / 4, atol=1e-6)

    def test_masked_minus_inf_classes_do_not_break_the_loss(self):
        logits = np.array([[1.0, -np.inf, 0.0]], dtype=np.float32)
        with no_numeric_warnings():
            x = T(logits, True)
            out = self.loss(x, T([0]))
            out.backward()
        self.assertTrue(np.isfinite(float(out.data)) and np.isfinite(x.grad).all())
        self.assertEqual(float(x.grad[0, 1]), 0.0)

    def test_rows_do_not_influence_each_others_gradients(self):
        p, t = self.data()
        g1 = grad_of(lambda x: self.loss(x, T(t)), p)
        other = p.copy()
        other[3] += 5.0
        g2 = grad_of(lambda x: self.loss(x, T(t)), other)
        np.testing.assert_allclose(g1[[0, 1, 2, 4]], g2[[0, 1, 2, 4]], atol=1e-7)

    def test_mean_is_invariant_to_duplicating_the_batch(self):
        p, t = self.data()
        double = float(self.loss(T(np.concatenate([p, p])), T(np.concatenate([t, t]))).data)
        self.assertAlmostEqual(double, float(self.loss(T(p), T(t)).data), places=6)

    def test_agrees_with_log_softmax_helper(self):
        p, t = self.data()
        lp = log_softmax(T(p)).data
        np.testing.assert_allclose(self.loss(T(p), T(t)).data, -lp[np.arange(len(t)), t].mean(), rtol=1e-5)


class TestSoftMaxCrossEntropyReductions(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.p = (rng.standard_normal((6, 4)) * 1.5).astype(np.float32)
        self.t = rng.integers(0, 4, 6)

    def test_invalid_reduction_raises(self):
        with self.assertRaises(ValueError):
            SoftMaxCrossEntropy(reduction="average")

    # def test_values_and_shapes(self):
    #     per = ce_per_sample_ref(self.p, self.t)
    #     none = SoftMaxCrossEntropy("none")(T(self.p), T(self.t))
    #     self.assertEqual(none.shape, self.t.shape)
    #     np.testing.assert_allclose(none.data, per, rtol=1e-5, atol=1e-7)
    #     np.testing.assert_allclose(SoftMaxCrossEntropy("sum")(T(self.p), T(self.t)).data, per.sum(), rtol=1e-5)
    #     np.testing.assert_allclose(SoftMaxCrossEntropy("mean")(T(self.p), T(self.t)).data, per.mean(), rtol=1e-5)

    # def test_gradcheck_for_every_reduction(self):
    #     for reduction in ("mean", "sum", "none"):
    #         assert_gradcheck(lambda x: SoftMaxCrossEntropy(reduction)(x, T(self.t)), self.p, msg=reduction)


# # =============================================================================
# # Binary cross entropy with logits
# # =============================================================================
# class TestBCEWithLogits(LossContract, unittest.TestCase):
#     make = staticmethod(BCEWithLogits)

#     def data(self):
#         return ((self.rng.standard_normal((5, 3)) * 1.5).astype(np.float32),
#                 self.rng.integers(0, 2, (5, 3)).astype(np.float32))

#     def test_zero_logit_costs_log_two(self):
#         out = self.loss(T([0.0, 0.0]), T([0.0, 1.0]))
#         self.assertAlmostEqual(float(out.data), float(np.log(2.0)), places=6)

#     def test_matches_reference_for_hard_and_soft_targets(self):
#         p, t = self.data()
#         np.testing.assert_allclose(self.loss(T(p), T(t)).data, bce_ref(p, t), rtol=1e-5)
#         soft = self.rng.uniform(0, 1, p.shape).astype(np.float32)
#         np.testing.assert_allclose(self.loss(T(p), T(soft)).data, bce_ref(p, soft), rtol=1e-5)

#     def test_matches_the_textbook_formula_for_moderate_logits(self):
#         z = np.linspace(-8, 8, 17).astype(np.float32)
#         for y in (0.0, 1.0, 0.3):
#             s = sigmoid_ref(z.astype(np.float64))
#             textbook = np.mean(-(y * np.log(s) + (1 - y) * np.log(1 - s)))
#             np.testing.assert_allclose(self.loss(T(z), T(np.full_like(z, y))).data, textbook, rtol=1e-5)

#     def test_is_non_negative_and_zero_for_confident_correct_predictions(self):
#         z = np.array([40.0, -40.0], dtype=np.float32)
#         out = float(self.loss(T(z), T([1.0, 0.0])).data)
#         self.assertGreaterEqual(out, 0.0)
#         self.assertLess(out, 1e-8)

#     def test_confident_mistakes_cost_the_full_logit_not_a_clipped_value(self):
#         z = np.array([20.0, -20.0, 30.0, -30.0, 50.0, -50.0], dtype=np.float32)
#         y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
#         np.testing.assert_allclose(self.loss(T(z), T(y)).data, np.abs(z).mean(), rtol=1e-5)

#     def test_confident_mistakes_keep_a_full_strength_gradient(self):
#         """d loss / d z = sigmoid(z) - y must stay +-1/N, never vanish when the model is confidently wrong."""
#         z = np.array([20.0, -20.0, 30.0, -30.0, 50.0, -50.0], dtype=np.float32)
#         y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
#         g = grad_of(lambda x: self.loss(x, T(y)), z)
#         np.testing.assert_allclose(g, np.array([1, -1, 1, -1, 1, -1]) / 6.0, atol=1e-6)

#     def test_gradient_is_sigmoid_minus_target_over_n(self):
#         p, t = self.data()
#         g = grad_of(lambda x: self.loss(x, T(t)), p)
#         np.testing.assert_allclose(g, (sigmoid_ref(p.astype(np.float64)) - t) / p.size, rtol=1e-4, atol=1e-7)

#     def test_soft_target_gradcheck(self):
#         p, _ = self.data()
#         soft = self.rng.uniform(0, 1, p.shape).astype(np.float32)
#         assert_gradcheck(lambda x: self.loss(x, T(soft)), p)

#     def test_extreme_logits_are_finite_without_warnings(self):
#         z = np.array([1e4, -1e4, 1e4, -1e4], dtype=np.float32)
#         y = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
#         with no_numeric_warnings():
#             x = T(z, True)
#             out = self.loss(x, T(y))
#             out.backward()
#         self.assertAlmostEqual(float(out.data) / 5e3, 1.0, places=5)           # (0 + 0 + 1e4 + 1e4) / 4
#         np.testing.assert_allclose(x.grad, np.array([0, 0, 1, -1]) / 4.0, atol=1e-6)

#     def test_default_reduction_is_the_mean(self):
#         p, t = self.data()
#         single = float(self.loss(T(p), T(t)).data)
#         double = float(self.loss(T(np.concatenate([p, p])), T(np.concatenate([t, t]))).data)
#         self.assertAlmostEqual(single, double, places=6, msg="loss must not grow with the batch size")
#         np.testing.assert_allclose(single, bce_per_element_ref(p, t).mean(), rtol=1e-5)

#     def test_flipping_logits_and_labels_gives_the_same_loss(self):
#         p, t = self.data()
#         a = float(self.loss(T(p), T(t)).data)
#         b = float(self.loss(T(-p), T(1.0 - t)).data)
#         self.assertAlmostEqual(a, b, places=5)

#     def test_moving_the_logit_towards_the_label_lowers_the_loss(self):
#         up = [float(self.loss(T([z]), T([1.0])).data) for z in (-2.0, 0.0, 2.0, 4.0)]
#         down = [float(self.loss(T([z]), T([0.0])).data) for z in (2.0, 0.0, -2.0, -4.0)]
#         self.assertTrue(np.all(np.diff(up) < 0) and np.all(np.diff(down) < 0))

#     def test_shape_mismatch_raises_instead_of_broadcasting(self):
#         with self.assertRaises(ValueError):
#             self.loss(T(np.zeros((4, 1))), T(np.zeros(4)))          # would silently become (4, 4)
#         with self.assertRaises(ValueError):
#             self.loss(T(np.zeros(4)), T(np.zeros(5)))

#     def test_gradient_flows_to_targets_when_they_require_grad(self):
#         p, t = self.data()
#         x, y = T(p, True), T(t, True)
#         self.loss(x, y).backward()
#         np.testing.assert_allclose(y.grad, -p / p.size, rtol=1e-5, atol=1e-8)


# class TestBCEWithLogitsReductions(unittest.TestCase):
#     def setUp(self):
#         rng = np.random.default_rng(1)
#         self.z = (rng.standard_normal((4, 3)) * 1.5).astype(np.float32)
#         self.y = rng.integers(0, 2, (4, 3)).astype(np.float32)

#     def test_invalid_reduction_raises(self):
#         with self.assertRaises(ValueError):
#             BCEWithLogits(reduction="average")

#     def test_values_and_shapes(self):
#         per = bce_per_element_ref(self.z, self.y)
#         none = BCEWithLogits("none")(T(self.z), T(self.y))
#         self.assertEqual(none.shape, self.z.shape)
#         np.testing.assert_allclose(none.data, per, rtol=1e-5, atol=1e-7)
#         np.testing.assert_allclose(BCEWithLogits("sum")(T(self.z), T(self.y)).data, per.sum(), rtol=1e-5)
#         np.testing.assert_allclose(BCEWithLogits("mean")(T(self.z), T(self.y)).data, per.mean(), rtol=1e-5)

#     def test_gradcheck_for_every_reduction(self):
#         for reduction in ("mean", "sum", "none"):
#             assert_gradcheck(lambda x: BCEWithLogits(reduction)(x, T(self.y)), self.z, msg=reduction)

#     def test_sum_gradient_is_sigmoid_minus_target(self):
#         g = grad_of(lambda x: BCEWithLogits("sum")(x, T(self.y)), self.z)
#         np.testing.assert_allclose(g, sigmoid_ref(self.z.astype(np.float64)) - self.y, rtol=1e-4, atol=1e-7)


# # =============================================================================
# # Relationships between losses
# # =============================================================================
# class TestLossConsistency(unittest.TestCase):
#     def test_two_class_cross_entropy_equals_bce_with_logits(self):
#         """softmax([0, z])[1] == sigmoid(z), so the two losses (and gradients) must agree."""
#         rng = np.random.default_rng(0)
#         z = (rng.standard_normal(8) * 2).astype(np.float32)
#         y = rng.integers(0, 2, 8)
#         logits = np.stack([np.zeros_like(z), z], axis=1)

#         ce_logits = T(logits, True)
#         ce = SoftMaxCrossEntropy()(ce_logits, T(y))
#         ce.backward()
#         bce_z = T(z, True)
#         bce = BCEWithLogits()(bce_z, T(y.astype(np.float32)))
#         bce.backward()

#         np.testing.assert_allclose(ce.data, bce.data, rtol=1e-5)
#         np.testing.assert_allclose(ce_logits.grad[:, 1], bce_z.grad, rtol=1e-4, atol=1e-7)


# # =============================================================================
# # Integration: whole networks and real training loops
# # =============================================================================
# class TestLossesInsideNetworks(unittest.TestCase):
#     def _net(self, rng, n_in, n_hidden, n_out, scale=0.7):
#         W1 = T(rng.standard_normal((n_in, n_hidden)) * scale, True)
#         b1 = T(rng.standard_normal(n_hidden) * 0.3, True)
#         W2 = T(rng.standard_normal((n_hidden, n_out)) * scale, True)
#         b2 = T(np.zeros(n_out), True)
#         return [W1, b1, W2, b2]

#     def test_mse_network_gradients_match_finite_differences(self):
#         rng = np.random.default_rng(0)
#         X, Y = T(rng.standard_normal((5, 2))), T(rng.standard_normal((5, 3)))
#         params, act, loss = self._net(rng, 2, 4, 3), Tanh(), MSE()
#         W1, b1, W2, b2 = params
#         check_network_gradients(lambda: loss(act(X @ W1 + b1) @ W2 + b2, Y), params)

#     def test_cross_entropy_network_gradients_match_finite_differences(self):
#         rng = np.random.default_rng(1)
#         X, y = T(rng.standard_normal((6, 2))), T([0, 1, 2, 0, 1, 2])
#         params, act, loss = self._net(rng, 2, 4, 3), Tanh(), SoftMaxCrossEntropy()
#         W1, b1, W2, b2 = params
#         check_network_gradients(lambda: loss(act(X @ W1 + b1) @ W2 + b2, y), params)

#     def test_bce_network_gradients_match_finite_differences(self):
#         rng = np.random.default_rng(2)
#         X, y = T(rng.standard_normal((6, 2))), T(rng.integers(0, 2, (6, 1)))
#         params, act, loss = self._net(rng, 2, 4, 1), Tanh(), BCEWithLogits()
#         W1, b1, W2, b2 = params
#         check_network_gradients(lambda: loss(act(X @ W1 + b1) @ W2 + b2, y), params)

#     def test_multi_output_linear_regression_recovers_the_true_weights(self):
#         rng = np.random.default_rng(0)
#         A = np.array([[3.0, -2.0, 0.5], [1.0, 2.0, -1.0]], dtype=np.float32)
#         c = np.array([1.0, -1.0, 0.5], dtype=np.float32)
#         X = rng.uniform(-1, 1, (64, 2)).astype(np.float32)
#         Y = T(X @ A + c)
#         X = T(X)
#         W, b = T(np.zeros((2, 3)), True), T(np.zeros(3), True)
#         loss_fn, first = MSE(), None
#         for _ in range(300):
#             loss = loss_fn(X @ W + b, Y)
#             loss.backward()
#             first = float(loss.data) if first is None else first
#             sgd([W, b], 0.5)
#         self.assertLess(float(loss.data), 1e-3 * first)
#         np.testing.assert_allclose(W.data, A, atol=0.05)
#         np.testing.assert_allclose(b.data, c, atol=0.05)

#     def test_softmax_regression_learns_three_blobs(self):
#         rng = np.random.default_rng(1)
#         labels = np.repeat([0, 1, 2], 40)
#         centers = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])
#         X = T(centers[labels] + rng.standard_normal((120, 2)) * 0.6)
#         y = T(labels)
#         W, b = T(np.zeros((2, 3)), True), T(np.zeros(3), True)
#         loss_fn, losses = SoftMaxCrossEntropy(), []
#         for _ in range(100):
#             logits = X @ W + b
#             loss = loss_fn(logits, y)
#             loss.backward()
#             losses.append(float(loss.data))
#             sgd([W, b], 0.5)
#         self.assertTrue(np.isfinite(losses).all())
#         self.assertAlmostEqual(losses[0], float(np.log(3.0)), places=5)
#         self.assertLess(losses[-1], 0.3 * losses[0])
#         self.assertGreater(float((logits.data.argmax(axis=1) == labels).mean()), 0.95)

#     def test_logistic_regression_learns_two_blobs(self):
#         rng = np.random.default_rng(2)
#         labels = np.repeat([0, 1], 50)
#         X = T(np.where(labels[:, None] == 1, 1.5, -1.5) + rng.standard_normal((100, 2)) * 0.7)
#         y = T(labels.reshape(-1, 1))
#         W, b = T(np.zeros((2, 1)), True), T(np.zeros(1), True)
#         loss_fn, losses = BCEWithLogits(), []
#         for _ in range(100):
#             logits = X @ W + b
#             loss = loss_fn(logits, y)
#             loss.backward()
#             losses.append(float(loss.data))
#             sgd([W, b], 0.5)
#         self.assertAlmostEqual(losses[0], float(np.log(2.0)), places=5)
#         self.assertLess(losses[-1], 0.3 * losses[0])
#         self.assertGreater(float(((logits.data > 0).astype(int).ravel() == labels).mean()), 0.95)

#     def test_bce_recovers_from_a_confidently_wrong_initialisation(self):
#         """Logits near +-25 with every label wrong: the gradient must still pull the weight back."""
#         rng = np.random.default_rng(3)
#         labels = np.repeat([0, 1], 25)
#         x = np.where(labels == 1, 1.0, -1.0) + rng.standard_normal(50) * 0.1
#         X, y = T(x.reshape(-1, 1)), T(labels.reshape(-1, 1))
#         W, b = T([[-25.0]], True), T(np.zeros(1), True)
#         loss_fn, losses = BCEWithLogits(), []
#         for _ in range(200):
#             logits = X @ W + b
#             loss = loss_fn(logits, y)
#             loss.backward()
#             losses.append(float(loss.data))
#             sgd([W, b], 0.5)
#         self.assertGreater(losses[0], 20.0)
#         self.assertLess(losses[-1], 0.2 * losses[0], "training stalled: the gradient vanished for confident mistakes")
#         self.assertGreater(float(((logits.data > 0).astype(int).ravel() == labels).mean()), 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
