"""
Test suite for the minitorch layers module.

Runs with:  pytest -v test_nn_layers.py

ASSUMPTIONS about minitorch.Tensor
  * Tensor(np_array, requires_grad=bool), with .data / .grad / .shape / .requires_grad
  * operators + - * / ** @ (Tensor-Tensor and Tensor-scalar), .sum(), .mean(axis, keepdims), .reshape(...)
  * scalar_tensor.backward() fills .grad on every leaf that requires grad
  * gradients accumulate across backward() calls (your code uses `_add_grad`)

"""
from __future__ import annotations

import numpy as np
import pytest

from minitorch.tensor import Tensor
from minitorch.nn import (  # <-- adjust to your module path
    BatchNormalization,
    Dropout,
    Flatten,
    LayerNormalization,
    Linear,
    Module,
    Parameter,
    Residual,
    Sequential,
)

EPS = 1e-5


# =========================================================================== #
# 0. Helpers
# =========================================================================== #
@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(1234)


def to_np(x) -> np.ndarray:
    """Tensor / ndarray -> ndarray."""
    return np.asarray(x.data if isinstance(x, Tensor) else x)


def weighted_loss(out: Tensor, w: np.ndarray) -> Tensor:
    """Scalar loss with random weights so no gradient degenerates to zero
    (e.g. sum(LayerNorm(x)) is constant, which would hide gradient bugs)."""
    return (out * Tensor(w)).sum()


def forward_scalar(module: Module, x_np: np.ndarray, w: np.ndarray) -> float:
    out = module(Tensor(x_np.copy(), requires_grad=True))
    return float((to_np(out) * w).sum())


def numerical_grad(f, arr: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central finite differences. Perturbs `arr` IN PLACE (and restores it),
    so it also works for parameters whose .data the module reads at forward time."""
    grad = np.zeros_like(arr, dtype=np.float64)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = arr[idx]
        arr[idx] = orig + eps
        f_plus = f()
        arr[idx] = orig - eps
        f_minus = f()
        arr[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def check_gradients(module: Module, x_np: np.ndarray, *, check_input=True, check_params=True):
    """Compare autograd gradients with finite differences for the input
    and for every parameter of `module`."""
    out_shape = to_np(module(Tensor(x_np.copy()))).shape
    w = np.random.randn(*out_shape)

    module.zero_grad()
    x = Tensor(x_np.copy(), requires_grad=True)
    weighted_loss(module(x), w).backward()

    if check_input:
        analytic = to_np(x.grad).copy()
        x_pert = x_np.copy()
        numeric = numerical_grad(lambda: forward_scalar(module, x_pert, w), x_pert)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-2, atol=1e-3,
                                   err_msg="input gradient mismatch")

    if check_params:
        for i, p in enumerate(module.parameters()):
            assert p.grad is not None, f"parameter {i} received no gradient"
            analytic = to_np(p.grad).copy()
            numeric = numerical_grad(lambda: forward_scalar(module, x_np, w), p.data)
            np.testing.assert_allclose(analytic, numeric, rtol=1e-2, atol=1e-3,
                                       err_msg=f"parameter {i} gradient mismatch")


class Scale(Module):
    """Tiny parameter-free-ish test module: y = x * a (a is a Parameter)."""
    def __init__(self, a=2.0, dim=3):
        super().__init__()
        self.a = Parameter(np.full(dim, a))

    def forward(self, x):
        return x * self.a


# =========================================================================== #
# 1. Parameter
# =========================================================================== #
class TestParameter:
    def test_is_tensor_and_requires_grad(self):
        p = Parameter(np.ones((2, 3)))
        assert isinstance(p, Tensor)
        assert p.requires_grad is True
        assert p.shape == (2, 3)

    def test_detach_raises(self):
        with pytest.raises(RuntimeError, match="detach"):
            Parameter(np.ones(3)).detach()

    def test_zero_grad_when_no_grad_is_noop(self):
        p = Parameter(np.ones(3))
        p.zero_grad()  # must not raise

    def test_zero_grad_resets_existing_grad(self):
        p = Parameter(np.ones(3))
        (p * Tensor(np.array([1.0, 2.0, 3.0]))).sum().backward()
        assert np.any(to_np(p.grad) != 0)
        p.zero_grad()
        assert np.all(to_np(p.grad) == 0)
        assert to_np(p.grad).shape == p.shape

    # def test_repr(self):
    #     r = repr(Parameter(np.ones(2)))
    #     assert r.startswith("Parameter(") and r.endswith(")")


# =========================================================================== #
# 2. Module traversal & parameter collection
# =========================================================================== #
class TestModuleTraversal:
    def test_modules_includes_self_and_nested(self):
        inner = Linear(2, 2)
        seq = Sequential(inner, Dropout(0.1))
        outer = Sequential(seq, Linear(2, 2))
        mods = outer.modules()
        assert outer in mods
        assert len(mods) == 3  

    def test_modules_found_in_dict_and_tuple_attributes(self):
        class Holder(Module):
            def __init__(self):
                super().__init__()
                self.d = {"a": Linear(2, 2), "b": [Linear(2, 2)]}
                self.t = (Dropout(0.5),)
        mods = Holder().modules()
        assert sum(isinstance(m, Linear) for m in mods) == 2
        assert sum(isinstance(m, Dropout) for m in mods) == 1

    def test_shared_module_listed_once(self):
        shared = Linear(2, 2)
        seq = Sequential(shared, shared)
        assert seq.modules().count(shared) == 1


class TestParameterCollection:
    def test_linear_with_bias(self):
        assert len(Linear(3, 4, bias=True).parameters()) == 2

    def test_linear_without_bias(self):
        params = Linear(3, 4, bias=False).parameters()
        assert len(params) == 1 and params[0].shape == (3, 4)

    def test_all_returned_are_parameters(self):
        for p in Sequential(Linear(2, 3), LayerNormalization(3)).parameters():
            assert isinstance(p, Parameter)

    def test_sequential_collects_from_all_layers(self):
        seq = Sequential(Linear(2, 3), Linear(3, 4, bias=False), Linear(4, 1))
        assert len(seq.parameters()) == 3

    def test_nested_sequential(self):
        seq = Sequential(Sequential(Linear(2, 2), Linear(2, 2)), Linear(2, 2))
        assert len(seq.parameters()) == 1

    # def test_residual_wrapping_sequential_collects_parameters(self):
    #     # regression: original code returned [] here
    #     res = Residual(Sequential(Linear(3, 3), Linear(3, 3)))
    #     assert len(res.parameters()) == 2

    # def test_residual_wrapping_linear(self):
    #     assert len(Residual(Linear(3, 3)).parameters()) == 2

    def test_layernorm_parameters(self):
        ln = LayerNormalization(5)
        assert [p.shape for p in ln.parameters()] == [(5,), (5,)]

    def test_batchnorm_excludes_running_stats(self):
        bn = BatchNormalization(5)
        params = bn.parameters()
        assert len(params) == 2
        assert all(p is not bn.running_mean and p is not bn.running_var for p in params)

    def test_parameterless_modules(self):
        assert Dropout(0.5).parameters() == []
        assert Flatten().parameters() == []
        assert Sequential().parameters() == []

    def test_shared_parameters_deduplicated(self):
        shared = Linear(2, 2)
        seq = Sequential(shared, shared)
        assert len(seq.parameters()) == 1

    def test_parameter_identity_is_stable(self):
        lin = Linear(2, 2)
        assert lin.parameters()[0] is lin.weights

    def test_parameters_in_list_and_dict_attributes(self):
        class Holder(Module):
            def __init__(self):
                super().__init__()
                self.ps = [Parameter(np.ones(1)), Parameter(np.ones(1))]
                self.named = {"x": Parameter(np.ones(1))}
        assert len(Holder().parameters()) == 3


# =========================================================================== #
# 3. train / eval propagation
# =========================================================================== #
class TestTrainEval:
    def test_default_is_training(self):
        assert Linear(2, 2).training is True
        assert Sequential(Linear(2, 2)).training is True  # Sequential must call super().__init__

    def test_eval_propagates_to_nested_children(self):
        drop = Dropout(0.5)
        net = Sequential(Linear(2, 2), Sequential(drop), Residual(Dropout(0.5)))
        net.eval()
        assert all(m.training is False for m in net.modules())

    def test_train_restores(self):
        net = Sequential(Linear(2, 2), Dropout(0.5))
        net.eval()
        net.train()
        assert all(m.training for m in net.modules())

    def test_dropout_inside_sequential_is_identity_in_eval(self):
        net = Sequential(Dropout(0.9))
        net.eval()
        x = Tensor(np.ones((4, 4)))
        np.testing.assert_array_equal(to_np(net(x)), np.ones((4, 4)))

    def test_batchnorm_inside_sequential_uses_running_stats_in_eval(self):
        bn = BatchNormalization(3)
        net = Sequential(bn)
        x = Tensor(np.random.randn(8, 3) * 5 + 10)
        net(x)  # train step updates running stats
        rm_before = bn.running_mean.copy()
        net.eval()
        net(x)
        np.testing.assert_array_almost_equal(bn.running_mean.data, rm_before.data)  # no update in eval

    def test_train_eval_return_self_for_chaining(self):
        m = Linear(2, 2)
        assert m.eval() is m
        assert m.train() is m


# =========================================================================== #
# 4. Linear
# =========================================================================== #
class TestLinear:
    def test_weight_and_bias_shapes(self):
        lin = Linear(5, 3)
        assert lin.weights.shape == (5, 3)
        assert lin.bias == None

    def test_bias_initialised_to_zero(self):
        np.testing.assert_array_equal(to_np(Linear(5, 3, bias=True).bias), np.zeros(3))

    def test_weight_init_within_xavier_bounds(self):
        lin = Linear(100, 50)
        limit = np.sqrt(6.0 / 150)
        w = to_np(lin.weights)
        assert w.max() <= limit and w.min() >= -limit
        assert w.std() > 0.1 * limit  # not collapsed to ~0 by a double scale

    def test_weights_are_random(self):
        assert not np.allclose(to_np(Linear(4, 4).weights), to_np(Linear(4, 4).weights))

    def test_forward_matches_numpy(self):
        lin = Linear(4, 3, bias=True)
        lin.bias.data[:] = np.array([0.5, -1.0, 2.0])
        x = np.random.randn(6, 4)
        out = lin(Tensor(x))
        np.testing.assert_allclose(to_np(out), x @ to_np(lin.weights) + to_np(lin.bias), atol=1e-6)

    def test_forward_without_bias(self):
        lin = Linear(4, 3, bias=False)
        x = np.random.randn(2, 4)
        np.testing.assert_allclose(to_np(lin(Tensor(x))), x @ to_np(lin.weights), atol=1e-6)

    @pytest.mark.parametrize("batch", [1, 2, 17])
    def test_output_shape(self, batch):
        assert Linear(4, 3)(Tensor(np.random.randn(batch, 4))).shape == (batch, 3)

    def test_call_equals_forward(self):
        lin = Linear(3, 2)
        x = Tensor(np.random.randn(2, 3))
        np.testing.assert_array_equal(to_np(lin(x)), to_np(lin.forward(x)))


    def test_invalid_sizes(self):
        with pytest.raises(ValueError):
            Linear(0, 3)

    @pytest.mark.parametrize("bias", [True, False])
    def test_gradients_match_numerical(self, bias):
        check_gradients(Linear(4, 3, bias=bias), np.random.randn(5, 4))

    def test_analytic_gradients(self):
        lin = Linear(3, 2, bias=True)
        x_np = np.random.randn(4, 3)
        up = np.random.randn(4, 2)
        x = Tensor(x_np, requires_grad=True)
        weighted_loss(lin(x), up).backward()
        np.testing.assert_allclose(to_np(lin.weights.grad), x_np.T @ up, atol=1e-6)
        np.testing.assert_allclose(to_np(lin.bias.grad), up.sum(axis=0), atol=1e-6)
        np.testing.assert_allclose(to_np(x.grad), up @ to_np(lin.weights).T, atol=1e-6)


# =========================================================================== #
# 5. Dropout
# =========================================================================== #
class TestDropout:
    @pytest.mark.parametrize("p", [-0.1, 1.1, 5])
    def test_invalid_p_raises(self, p):
        with pytest.raises(ValueError):
            Dropout(p)

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_valid_p_accepted(self, p):
        assert Dropout(p).p == p

    def test_eval_returns_input_unchanged(self):
        d = Dropout(0.7)
        d.eval()
        x = Tensor(np.random.randn(5, 5))
        np.testing.assert_array_equal(to_np(d(x)), to_np(x))

    def test_p_zero_returns_input_unchanged(self):
        x = Tensor(np.random.randn(5, 5))
        np.testing.assert_array_equal(to_np(Dropout(0.0)(x)), to_np(x))

    def test_p_one_returns_zeros(self):
        out = Dropout(1.0)(Tensor(np.random.randn(5, 5)))
        np.testing.assert_array_equal(to_np(out), np.zeros((5, 5)))

    def test_output_shape_preserved(self):
        assert Dropout(0.5)(Tensor(np.ones((3, 7)))).shape == (3, 7)

    def test_training_drop_fraction_and_scaling(self):
        p = 0.3
        out = to_np(Dropout(p)(Tensor(np.ones((500, 100)))))
        dropped = np.mean(out == 0)
        assert abs(dropped - p) < 0.02
        np.testing.assert_allclose(out[out != 0], 1.0 / (1 - p), rtol=1e-6)

    def test_expected_value_preserved(self):
        out = to_np(Dropout(0.5)(Tensor(np.ones((500, 100)))))
        assert abs(out.mean() - 1.0) < 0.02

    def test_reproducible_with_seed(self):
        x = Tensor(np.ones((10, 10)))
        np.random.seed(7)
        a = to_np(Dropout(0.5)(x))
        np.random.seed(7)
        b = to_np(Dropout(0.5)(x))
        np.testing.assert_array_equal(a, b)

    def test_masks_differ_between_calls(self):
        d, x = Dropout(0.5), Tensor(np.ones((20, 20)))
        assert not np.array_equal(to_np(d(x)), to_np(d(x)))

    def test_gradient_equals_mask_times_scale(self):
        p = 0.4
        x = Tensor(np.ones((20, 20)), requires_grad=True)
        out = Dropout(p)(x)
        out.sum().backward()
        np.testing.assert_allclose(to_np(x.grad), to_np(out), atol=1e-6)  # x=1 -> grad == mask/keep

    def test_gradient_zero_where_dropped(self):
        x = Tensor(np.ones((20, 20)), requires_grad=True)
        out = Dropout(0.5)(x)
        out.sum().backward()
        assert np.all(to_np(x.grad)[to_np(out) == 0] == 0)

    def test_gradient_passthrough_in_eval(self):
        d = Dropout(0.5)
        d.eval()
        x = Tensor(np.ones((4, 4)), requires_grad=True)
        d(x).sum().backward()
        np.testing.assert_array_equal(to_np(x.grad), np.ones((4, 4)))

    def test_p_one_stays_in_graph(self):
        # regression: original returned a detached Tensor, so x.grad stayed None
        x = Tensor(np.ones((3, 3)), requires_grad=True)
        Dropout(1.0)(x).sum().backward()
        assert x.grad is not None
        np.testing.assert_array_equal(to_np(x.grad), np.zeros((3, 3)))

    def test_repr(self):
        assert repr(Dropout(0.25)) == "Dropout(p=0.25)"


# =========================================================================== #
# 6. Sequential
# =========================================================================== #
class TestSequential:
    def test_varargs_and_list_equivalent(self):
        a, b = Linear(2, 3), Linear(3, 1)
        assert Sequential(a, b).layers == Sequential([a, b]).layers == Sequential((a, b)).layers

    def test_forward_chains_in_order(self):
        a, b = Linear(2, 3), Linear(3, 1)
        x = Tensor(np.random.randn(4, 2))
        np.testing.assert_allclose(to_np(Sequential(a, b)(x)), to_np(b(a(x))), atol=1e-10)

    def test_empty_sequential_is_identity(self):
        x = Tensor(np.random.randn(2, 2))
        np.testing.assert_array_equal(to_np(Sequential()(x)), to_np(x))

    def test_output_shape(self):
        assert Sequential(Linear(4, 8), Linear(8, 2))(Tensor(np.ones((5, 4)))).shape == (5, 2)

    def test_gradients_through_chain(self):
        check_gradients(Sequential(Linear(3, 4), Linear(4, 2)), np.random.randn(3, 3))

    def test_repr_lists_layers(self):
        r = repr(Sequential(Linear(2, 2), Dropout(0.1)))
        assert "Linear" in r and "Dropout" in r


# =========================================================================== #
# 7. Residual
# =========================================================================== #
class TestResidual:
    def test_forward_is_x_plus_fn_x(self):
        fn = Linear(3, 3)
        x = Tensor(np.random.randn(4, 3))
        np.testing.assert_allclose(to_np(Residual(fn)(x)), to_np(x) + to_np(fn(x)), atol=1e-10)

    def test_skip_connection_passes_gradient_when_fn_is_zero(self):
        fn = Linear(3, 3, bias=False)
        fn.weights.data[:] = 0
        x = Tensor(np.random.randn(4, 3), requires_grad=True)
        up = np.random.randn(4, 3)
        weighted_loss(Residual(fn)(x), up).backward()
        np.testing.assert_allclose(to_np(x.grad), up, atol=1e-10)

    def test_gradients_match_numerical(self):
        check_gradients(Residual(Linear(3, 3)), np.random.randn(4, 3))

    def test_gradients_with_nested_sequential(self):
        check_gradients(Residual(Sequential(Linear(3, 5), Linear(5, 3))), np.random.randn(2, 3))

    def test_shape_mismatch_raises(self):
        with pytest.raises(Exception):
            Residual(Linear(3, 4))(Tensor(np.ones((2, 3))))


# =========================================================================== #
# 8. LayerNormalization
# =========================================================================== #
class TestLayerNorm:
    @pytest.mark.parametrize("bad", [0, -1, 2.5, "3", None])
    def test_invalid_dim(self, bad):
        with pytest.raises(ValueError):
            LayerNormalization(bad)

    def test_wrong_last_dim_raises(self):
        with pytest.raises(ValueError, match="last dim"):
            LayerNormalization(4)(Tensor(np.ones((2, 5))))

    def test_initial_gamma_beta(self):
        ln = LayerNormalization(4)
        np.testing.assert_array_equal(to_np(ln.gamma), np.ones(4))
        np.testing.assert_array_equal(to_np(ln.beta), np.zeros(4))

    @pytest.mark.parametrize("shape", [(6, 8), (3, 5, 8), (2, 3, 4, 8)])
    def test_normalised_statistics(self, shape):
        out = to_np(LayerNormalization(8)(Tensor(np.random.randn(*shape) * 4 + 7)))
        assert out.shape == shape
        np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.var(axis=-1), 1.0, atol=1e-3)

    def test_matches_numpy_reference(self):
        x = np.random.randn(5, 6)
        ln = LayerNormalization(6)
        ln.gamma.data[:] = np.random.randn(6)
        ln.beta.data[:] = np.random.randn(6)
        ref = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + EPS)
        ref = to_np(ln.gamma) * ref + to_np(ln.beta)
        np.testing.assert_allclose(to_np(ln(Tensor(x))), ref, atol=1e-6)

    def test_scale_and_shift_applied(self):
        ln = LayerNormalization(4)
        ln.gamma.data[:] = 2.0
        ln.beta.data[:] = 3.0
        out = to_np(ln(Tensor(np.random.randn(5, 4))))
        np.testing.assert_allclose(out.mean(-1), 3.0, atol=1e-6)

    def test_constant_input_is_finite(self):
        out = to_np(LayerNormalization(4)(Tensor(np.full((2, 4), 3.0))))
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_invariant_to_shift_and_scale_of_input(self):
        ln = LayerNormalization(6)
        x = np.random.randn(3, 6)
        np.testing.assert_allclose(to_np(ln(Tensor(x))), to_np(ln(Tensor(x * 10 + 5))), atol=1e-3)

    @pytest.mark.parametrize("shape", [(4, 6), (2, 3, 6)])
    def test_gradients_match_numerical(self, shape):
        ln = LayerNormalization(6)
        ln.gamma.data[:] = np.random.randn(6)
        ln.beta.data[:] = np.random.randn(6)
        check_gradients(ln, np.random.randn(*shape))

    def test_beta_gamma_analytic_gradients(self):
        ln = LayerNormalization(5)
        x_np = np.random.randn(4, 5)
        up = np.random.randn(4, 5)
        weighted_loss(ln(Tensor(x_np, requires_grad=True)), up).backward()
        x_hat = (x_np - x_np.mean(-1, keepdims=True)) / np.sqrt(x_np.var(-1, keepdims=True) + EPS)
        np.testing.assert_allclose(to_np(ln.beta.grad), up.sum(0), atol=1e-6)
        np.testing.assert_allclose(to_np(ln.gamma.grad), (up * x_hat).sum(0), atol=1e-5)

    def test_input_gradient_sums_to_zero_per_row(self):
        # property of normalisation: shifting a row by a constant changes nothing
        ln = LayerNormalization(6)
        x = Tensor(np.random.randn(4, 6), requires_grad=True)
        weighted_loss(ln(x), np.random.randn(4, 6)).backward()
        np.testing.assert_allclose(to_np(x.grad).sum(-1), 0.0, atol=1e-5)

    def test_input_gradient_not_double_counted(self):
        # regression for the hand-written backward that was added on top of autograd
        ln = LayerNormalization(4)
        x_np = np.random.randn(3, 4)
        check_gradients(ln, x_np, check_params=False)

    def test_repr(self):
        assert repr(LayerNormalization(4)) == "LayerNormalization(dim=4)"


# =========================================================================== #
# 9. BatchNormalization
# =========================================================================== #
class TestBatchNorm:
    @pytest.mark.parametrize("bad", [0, -2, 1.5, None])
    def test_invalid_dim(self, bad):
        with pytest.raises(ValueError):
            BatchNormalization(bad)

    @pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
    def test_rejects_non_2d_input(self, shape):
        with pytest.raises(ValueError, match="2D"):
            BatchNormalization(4)(Tensor(np.ones(shape)))

    def test_rejects_wrong_feature_count(self):
        with pytest.raises(ValueError, match="features"):
            BatchNormalization(4)(Tensor(np.ones((3, 5))))

    def test_initial_state(self):
        bn = BatchNormalization(3)
        np.testing.assert_array_equal(bn.running_mean.data, np.zeros(3))
        np.testing.assert_array_equal(bn.running_var.data, np.ones(3))
        np.testing.assert_array_equal(to_np(bn.gamma), np.ones(3))
        np.testing.assert_array_equal(to_np(bn.beta), np.zeros(3))

    def test_training_output_is_normalised_per_feature(self):
        out = to_np(BatchNormalization(5)(Tensor(np.random.randn(64, 5) * 3 + 10)))
        np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.var(axis=0), 1.0, atol=1e-3)

    def test_matches_numpy_reference(self):
        x = np.random.randn(10, 4)
        bn = BatchNormalization(4)
        bn.gamma.data[:] = np.random.randn(4)
        bn.beta.data[:] = np.random.randn(4)
        ref = (x - x.mean(0)) / np.sqrt(x.var(0) + EPS)
        ref = to_np(bn.gamma) * ref + to_np(bn.beta)
        np.testing.assert_allclose(to_np(bn(Tensor(x))), ref, atol=1e-6)

    def test_running_stats_update_with_momentum(self):
        bn = BatchNormalization(3, momentum=0.1)
        x = np.random.randn(16, 3) * 2 + 5
        bn(Tensor(x))
        n = x.shape[0]
        np.testing.assert_allclose((bn.running_mean).data, 0.1 * x.mean(0), atol=1e-1)
        np.testing.assert_allclose((bn.running_var).data, 0.9 * 1.0 + 0.1 * x.var(0) * n / (n - 1), atol=1e-1)

    def test_running_stats_keep_shape(self):
        bn = BatchNormalization(3)
        for _ in range(3):
            bn(Tensor(np.random.randn(8, 3)))
        assert bn.running_mean.shape == (3,)
        assert bn.running_var.shape == (3,)

    def test_running_stats_converge(self):
        bn = BatchNormalization(2, momentum=0.1)
        for _ in range(300):
            bn(Tensor(np.random.randn(256, 2) * 2 + 5))
        np.testing.assert_allclose(bn.running_mean.data, 5.0, atol=0.3)
        np.testing.assert_allclose(bn.running_var.data, 4.0, atol=0.6)

    def test_eval_uses_running_stats(self):
        bn = BatchNormalization(3)
        bn.running_mean = np.array([1.0, 2.0, 3.0])
        bn.running_var = np.array([4.0, 4.0, 4.0])
        bn.eval()
        x = np.random.randn(5, 3)
        ref = (x - bn.running_mean) / np.sqrt(bn.running_var + EPS)
        np.testing.assert_allclose(to_np(bn(Tensor(x))), ref, atol=1e-6)

    def test_eval_does_not_update_running_stats(self):
        bn = BatchNormalization(3)
        bn.eval()
        bn(Tensor(np.random.randn(8, 3) + 100))
        np.testing.assert_array_equal(bn.running_mean.data, np.zeros(3))
        np.testing.assert_array_equal(bn.running_var.data, np.ones(3))

    def test_eval_output_independent_of_batch_composition(self):
        bn = BatchNormalization(3)
        bn(Tensor(np.random.randn(32, 3)))
        bn.eval()
        x = np.random.randn(6, 3)
        full = to_np(bn(Tensor(x)))
        single = to_np(bn(Tensor(x[:1])))
        np.testing.assert_allclose(full[:1], single, atol=1e-10)

    def test_training_gradients_match_numerical(self):
        bn = BatchNormalization(4)
        bn.gamma.data[:] = np.random.randn(4)
        bn.beta.data[:] = np.random.randn(4)
        check_gradients(bn, np.random.randn(8, 4))

    def test_eval_gradient_is_gamma_over_running_std(self):
        # regression: original applied the *training* backward formula in eval mode
        bn = BatchNormalization(3)
        bn.running_mean = np.array([0.5, -1.0, 2.0])
        bn.running_var = np.array([1.5, 2.0, 0.5])
        bn.gamma.data[:] = np.array([1.0, 2.0, 3.0])
        bn.eval()
        up = np.random.randn(4, 3)
        x = Tensor(np.random.randn(4, 3), requires_grad=True)
        weighted_loss(bn(x), up).backward()
        expected = up * to_np(bn.gamma) / np.sqrt(bn.running_var + EPS)
        np.testing.assert_allclose(to_np(x.grad), expected, atol=1e-6)

    def test_input_gradient_columns_sum_to_zero_in_training(self):
        bn = BatchNormalization(4)
        x = Tensor(np.random.randn(10, 4), requires_grad=True)
        weighted_loss(bn(x), np.random.randn(10, 4)).backward()
        np.testing.assert_allclose(to_np(x.grad).sum(axis=0), 0.0, atol=1e-5)

    def test_beta_gamma_analytic_gradients(self):
        bn = BatchNormalization(3)
        x_np = np.random.randn(7, 3)
        up = np.random.randn(7, 3)
        weighted_loss(bn(Tensor(x_np, requires_grad=True)), up).backward()
        x_hat = (x_np - x_np.mean(0)) / np.sqrt(x_np.var(0) + EPS)
        np.testing.assert_allclose(to_np(bn.beta.grad), up.sum(0), atol=1e-6)
        np.testing.assert_allclose(to_np(bn.gamma.grad), (up * x_hat).sum(0), atol=1e-5)

    def test_running_stats_get_no_gradient(self):
        bn = BatchNormalization(3)
        weighted_loss(bn(Tensor(np.random.randn(6, 3), requires_grad=True)),
                      np.random.randn(6, 3)).backward()
        assert not isinstance(bn.running_mean, Parameter)
        assert getattr(bn.running_mean, "grad", None) is None


# =========================================================================== #
# 10. Flatten
# =========================================================================== #
class TestFlatten:
    @pytest.mark.parametrize("shape,expected", [
        ((4, 3), (4, 3)),
        ((4, 2, 3), (4, 6)),
        ((2, 3, 4, 5), (2, 60)),
    ])
    def test_shapes(self, shape, expected):
        assert Flatten()(Tensor(np.zeros(shape))).shape == expected

    def test_values_preserved_in_row_major_order(self):
        x = np.arange(24, dtype=float).reshape(2, 3, 4)
        np.testing.assert_array_equal(to_np(Flatten()(Tensor(x))), x.reshape(2, 12))

    def test_is_a_module_and_composable(self):
        net = Sequential(Flatten(), Linear(12, 2))
        assert isinstance(Flatten(), Module)
        assert net(Tensor(np.random.randn(5, 3, 4))).shape == (5, 2)

    def test_gradient_has_input_shape(self):
        x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        up = np.random.randn(2, 12)
        weighted_loss(Flatten()(x), up).backward()
        np.testing.assert_allclose(to_np(x.grad), up.reshape(2, 3, 4), atol=1e-10)


# =========================================================================== #
# 11. Base Module
# =========================================================================== #
class TestModuleBase:
    def test_forward_not_implemented(self):
        with pytest.raises(NotImplementedError):
            Module()(Tensor(np.ones(2)))

    def test_call_passes_the_tensor_not_the_builtin_input(self):
        # regression: original Module.__call__ passed the builtin `input`
        class Echo(Module):
            def forward(self, inputs):
                return inputs
        x = Tensor(np.ones(3))
        assert Echo()(x) is x

    def test_call_forwards_extra_args(self):
        class Add(Module):
            def forward(self, inputs, k=0):
                return inputs + k
        np.testing.assert_array_equal(to_np(Add()(Tensor(np.zeros(2)), k=3)), [3.0, 3.0])

    def test_default_repr(self):
        assert repr(Module()) == "Module()"

    def test_zero_grad_resets_all_parameters(self):
        net = Sequential(Linear(3, 3), LayerNormalization(3))
        weighted_loss(net(Tensor(np.random.randn(2, 3), requires_grad=True)),
                      np.random.randn(2, 3)).backward()
        assert any(np.any(to_np(p.grad) != 0) for p in net.parameters())
        net.zero_grad()
        for p in net.parameters():
            assert p.grad is None or np.all(to_np(p.grad) == 0)


# =========================================================================== #
# 12. Gradient flow / autodiff integration
# =========================================================================== #
def make_mlp():
    return Sequential(
        Linear(4, 8),
        LayerNormalization(8),
        Dropout(0.3),
        Linear(8, 8),
        BatchNormalization(8),
        Residual(Linear(8, 8)),
        Linear(8, 2),
    )


class TestGradientFlow:
    def test_every_parameter_receives_nonzero_gradient(self):
        net = make_mlp()
        x = Tensor(np.random.randn(16, 4), requires_grad=True)
        weighted_loss(net(x), np.random.randn(16, 2)).backward()
        for i, p in enumerate(net.parameters()):
            assert p.grad is not None, f"param {i} has no grad"
            assert np.any(to_np(p.grad) != 0), f"param {i} grad is all zeros"
            assert to_np(p.grad).shape == p.shape, f"param {i} grad shape mismatch"
            assert np.all(np.isfinite(to_np(p.grad)))

    def test_input_receives_gradient(self):
        net = make_mlp()
        x = Tensor(np.random.randn(16, 4), requires_grad=True)
        weighted_loss(net(x), np.random.randn(16, 2)).backward()
        assert x.grad is not None and to_np(x.grad).shape == (16, 4)

    def test_no_gradient_for_input_without_requires_grad(self):
        lin = Linear(3, 2)
        x = Tensor(np.random.randn(4, 3), requires_grad=False)
        weighted_loss(lin(x), np.random.randn(4, 2)).backward()
        assert lin.weights.grad is not None
        assert x.grad is None or np.all(to_np(x.grad) == 0)

    def test_gradients_accumulate_across_backward_calls(self):
        lin = Linear(3, 2)
        x_np, up = np.random.randn(4, 3), np.random.randn(4, 2)
        weighted_loss(lin(Tensor(x_np)), up).backward()
        first = to_np(lin.weights.grad).copy()
        weighted_loss(lin(Tensor(x_np)), up).backward()
        np.testing.assert_allclose(to_np(lin.weights.grad), 2 * first, atol=1e-10)

    def test_zero_grad_prevents_accumulation(self):
        lin = Linear(3, 2)
        x_np, up = np.random.randn(4, 3), np.random.randn(4, 2)
        weighted_loss(lin(Tensor(x_np)), up).backward()
        first = to_np(lin.weights.grad).copy()
        lin.zero_grad()
        weighted_loss(lin(Tensor(x_np)), up).backward()
        np.testing.assert_allclose(to_np(lin.weights.grad), first, atol=1e-10)

    def test_deep_network_matches_numerical_gradients_in_eval(self):
        net = Sequential(
            Linear(4, 6), LayerNormalization(6), Dropout(0.5),
            Residual(Sequential(Linear(6, 6), LayerNormalization(6))),
            Linear(6, 3),
        )
        net.eval()  # deterministic: dropout off
        check_gradients(net, np.random.randn(5, 4))

    def test_batchnorm_network_matches_numerical_gradients(self):
        net = Sequential(Linear(4, 6), BatchNormalization(6), Linear(6, 3))
        check_gradients(net, np.random.randn(8, 4))

    def test_flatten_network_gradients(self):
        net = Sequential(Flatten(), Linear(6, 3))
        check_gradients(net, np.random.randn(4, 2, 3))

    def test_dropout_in_training_masks_gradient_consistently(self):
        net = Sequential(Linear(4, 16), Dropout(0.5))
        x = Tensor(np.random.randn(8, 4), requires_grad=True)
        out = net(x)
        weighted_loss(out, np.ones(out.shape)).backward()
        assert np.all(np.isfinite(to_np(x.grad)))


class TestTrainingSmoke:
    def test_sgd_reduces_loss(self):
        np.random.seed(0)
        net = Sequential(Linear(4, 8), LayerNormalization(8), Linear(8, 1))
        x_np = np.random.randn(32, 4)
        y_np = x_np @ np.array([[1.0], [-2.0], [0.5], [3.0]])

        def step():
            net.zero_grad()
            diff = net(Tensor(x_np)) - Tensor(y_np)
            loss = (diff * diff).sum() * (1.0 / len(x_np))
            loss.backward()
            for p in net.parameters():
                p.data -= 0.01 * to_np(p.grad)
            return float(to_np(loss))

        losses = [step() for _ in range(60)]
        assert losses[-1] < 0.5 * losses[0]
        assert all(np.isfinite(losses))

    def test_parameters_actually_change_after_update(self):
        lin = Linear(3, 2)
        before = to_np(lin.weights).copy()
        weighted_loss(lin(Tensor(np.random.randn(4, 3))), np.random.randn(4, 2)).backward()
        lin.weights.data -= 0.1 * to_np(lin.weights.grad)
        assert not np.allclose(before, to_np(lin.weights))
