
import numpy as np
import torch

from minitorch.tensor import Tensor
from minitorch.nn import Linear
from minitorch.conftest import assert_close


def test_linear_forward_backward():
    rng = np.random.default_rng(42)

    batch_size = 8
    in_features = 16
    out_features = 10

    x_data = rng.standard_normal(
        (batch_size, in_features)
    ).astype(np.float32)

    layer = Linear(in_features, out_features, bias=True)

    torch_layer = torch.nn.Linear(
        in_features, out_features
    )

    # Copy custom parameters to PyTorch.
    torch_layer.weight.data.copy_(
        torch.tensor(layer.weights.data.T)
    )
    torch_layer.bias.data.copy_(
        torch.tensor(layer.bias.data)
    )

    x_custom = Tensor(x_data.copy(), requires_grad=True)
    x_torch = torch.tensor(x_data.copy(), requires_grad=True)

    y_custom = layer(x_custom)
    y_torch = torch_layer(x_torch)

    assert_close(y_custom, y_torch)

    upstream = rng.standard_normal(
        y_torch.shape
    ).astype(np.float32)

    (y_custom * Tensor(upstream)).sum().backward()
    (y_torch * torch.tensor(upstream)).sum().backward()

    assert_close(x_custom.grad, x_torch.grad)
    assert_close(layer.weights.grad, torch_layer.weight.grad.T)
    assert_close(layer.bias.grad, torch_layer.bias.grad)

