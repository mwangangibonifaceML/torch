import math

import numpy as np

from minitorch.nn import Module
from minitorch.tensor import Tensor


class Sigmoid(Module):
    """
    Sigmoid activation: σ(x) = 1/(1 + e^(-x))

    Maps input to range (0,1). It's derivative is at most 0.25 and
    might causes gradients to vanish for large |x|, therefore it
    is best used in output layers (for probabilities and binary
    classificationa) rather than in deep hidden stacks

    Example:
        >>> y = Sigmoid()(Tensor(np.array([-1.0, 0.0, 1.0])))
        >>> y
        Tensor(data=[0.26894143, 0.5       , 0.7310586 ])
    """

    def forward(self, inputs: Tensor) -> Tensor:
        x = Tensor._ensure_tensor(inputs)
        d = x.data
        z = np.exp(-np.abs(d))
        out_data = np.where(d >= 0, 1.0 / (1.0 + z), z / (1 + z)).astype(np.float32)
        out = Tensor(
            data=out_data,
            requires_grad=x.requires_grad,
            _parents=(x,) if x.requires_grad else (),
            dtype=out_data.dtype,
            device=x.device,
        )

        if x.requires_grad:

            def _backward():
                if out.grad is not None:
                    x._add_grad(out.grad * out.data * (1 - out.data))

            out._backward = _backward

        return out


class ReLU(Module):
    """ReLU activation: f(x) = max(0, x)

    Sets negative values to zero, keeps positive values unchanged.

    Example:
    >>> relu = ReLu()
    >>> x = Tensor(np.array([-1.0, 0.0, 1.0]))
    >>> y = relu(x)
    >>> print(y)
    Tensor(data=[0.0, 0.0, 1.0])
    """

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of the ReLU activation function.

        Approach:
        1. use np.maximum(0, x.data) for element-wise max.
        2. Return the result wrapped in new tensor.

        Args:
            X (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying ReLU activation.
        """
        X = Tensor._ensure_tensor(X)
        result = Tensor(
            np.maximum(0, X.data),
            requires_grad=X.requires_grad,
            _parents=(X,) if X.requires_grad else (),
            dtype=np.float32,
            device=X.device,
        )

        if X.requires_grad:

            def _backward():
                if result.grad is not None:
                    X._add_grad(np.where(result.data > 0, result.grad, np.float32(0.0)))

            result._backward = _backward
        return result


class Tanh(Module):
    """
    Tanh activation: f(x) = tanh(x)

    Applies the hyperbolic tangent function element-wise.

    Example:
    >>> tanh = Tanh()
    >>> x = Tensor(np.array([-1.0, 0.0, 1.0]))
    >>> y = tanh(x)
    >>> print(y)
    Tensor(data=[-0.76159416,  0.0       ,  0.76159416])
    """

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of the Tanh activation function.

        Args:
            X (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying Tanh activation.
        """
        X = Tensor._ensure_tensor(X)
        result = np.tanh(X.data)
        result = Tensor(
            result,
            requires_grad=X.requires_grad,
            _parents=(X,) if X.requires_grad else (),
            device=X.device,
            dtype=np.float32,
        )

        if X.requires_grad:

            def _backward():
                if result.grad is not None:
                    Tanh_gradient = 1 - result.data**2
                    X._add_grad(result.grad * Tanh_gradient)

            result._backward = _backward
        return result


def _erf(x):
    x = np.asarray(x)
    s, a = np.sign(x), np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    poly = (
        (((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736
    ) * t + 0.254829592
    return s * (1.0 - poly * t * np.exp(-a * a))


def _sigmoid(d):
    z = np.exp(-np.abs(d))
    return np.where(d >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def _gelu_np(d, approximate, with_grad):
    """Return (gelu(d), d gelu/d d or None) as NumPy arrays."""
    if approximate == "none":  # exact: x * Phi(x)
        cdf = 0.5 * (1.0 + _erf(d / math.sqrt(2)))
        out = d * cdf
        grad = (
            cdf + d * np.exp(-0.5 * d * d) / math.sqrt(2.0 * math.pi)
            if with_grad
            else None
        )
    elif approximate == "tanh":
        t = np.tanh(math.sqrt(2.0 / math.pi) * (d + 0.044715 * d**3))
        out = 0.5 * d * (1.0 + t)
        grad = (
            (
                0.5 * (1.0 + t)
                + 0.5
                * d
                * (1.0 - t * t)
                * math.sqrt(2.0 / math.pi)
                * (1.0 + 3 * 0.044715 * d * d)
            )
            if with_grad
            else None
        )
    else:  # "sigmoid": x * sigmoid(1.702 x)
        s = _sigmoid(1.702 * d)
        out = d * s
        grad = s + 1.702 * d * s * (1.0 - s) if with_grad else None
    return out, grad


class GELU(Module):
    """GELU activation"""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        if approximate not in ("none", "tanh", "sigmoid"):
            raise ValueError("approximate must be 'none', 'tanh' or 'sigmoid'")
        self.approximate = approximate

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of the GELU activation function.

        Approach:
        1. Use approximation : x * sigmoid(1.702 * x).
        2. Compute sigmoid part: 1 / (1 + exp(-1.702 * x)).
        3. Multiply input x with sigmoid result.
        4. Return the result wrapped in new tensor.

        """
        X = Tensor._ensure_tensor(X)
        result_data, local_grad = _gelu_np(
            X.data, self.approximate, with_grad=X.requires_grad
        )

        result = Tensor(
            result_data,
            requires_grad=X.requires_grad,
            _parents=(X,) if X.requires_grad else (),
            dtype=np.float32,
            device=X.device,
        )

        if X.requires_grad:

            def _backward():
                if result.grad is not None:
                    X._add_grad(result.grad * local_grad)

            result._backward = _backward
        return result


class Softmax(Module):
    """Softmax activation: f(x_i) = e^(x_i) / Σ e^(x_j)

    Converts logits to probabilities across classes."""

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, X: Tensor) -> Tensor:
        """Forward pass of the Softmax activation function.
        Approach:
        1. Subtract max for numerical stability: x - max(x).
        2. Compute exponentials: exp(x - max(x)).
        3. sum along dimension: sum(exp_values).
        3. Normalize: exp_values / sum(exp_values).
        4. Return the result wrapped in new tensor.
        """
        X = Tensor._ensure_tensor(X)
        x = X.data
        x_shifted = x - np.max(x, axis=self.dim, keepdims=True)
        exp_x = np.exp(x_shifted)
        sum_exp_x = np.sum(exp_x, axis=self.dim, keepdims=True)
        out_data = exp_x / sum_exp_x

        result = Tensor(
            out_data,
            requires_grad=X.requires_grad,
            _parents=(X,) if X.requires_grad else (),
            device=X.device,
            dtype=np.float32,
        )

        if X.requires_grad:

            def _backward():
                if result.grad is not None:
                    sum_term = np.sum(
                        result.grad * result.data, axis=self.dim, keepdims=True
                    )
                    gradient = result.data * (result.grad - sum_term)
                    X._add_grad(gradient)

            result._backward = _backward

        return result
