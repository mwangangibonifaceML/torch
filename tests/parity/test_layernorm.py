
import numpy as np
import torch

from minitorch.nn.layers import LayerNormalization
from minitorch.tensor import Tensor
from minitorch.conftest import assert_close


def test_layernorm_forward_backward():
    rng = np.random.default_rng(42)

    x_data = rng.standard_normal((4, 8, 16)).astype(np.float32)

    layer = LayerNormalization(16)
    torch_layer = torch.nn.LayerNorm(16, eps=layer.eps)

    # Align gamma and beta.
    torch_layer.weight.data.copy_(torch.tensor(layer.gamma.data))
    torch_layer.bias.data.copy_(torch.tensor(layer.beta.data))

    x_custom = Tensor(x_data.copy(), requires_grad=True)
    x_torch = torch.tensor(x_data.copy(), requires_grad=True)

    y_custom = layer(x_custom)
    y_torch = torch_layer(x_torch)

    assert_close(y_custom, y_torch)

    upstream = rng.standard_normal(y_torch.shape).astype(np.float32)

    (y_custom * Tensor(upstream)).sum().backward()
    (y_torch * torch.tensor(upstream)).sum().backward()

    assert_close(x_custom.grad, x_torch.grad)
    assert_close(layer.gamma.grad, torch_layer.weight.grad)
    assert_close(layer.beta.grad, torch_layer.bias.grad)
