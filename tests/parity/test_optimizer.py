
import numpy as np
import torch

from minitorch.nn import Parameter
from minitorch.tensor import Tensor
from minitorch.optim import SGD, AdamW
from minitorch.conftest import assert_close


def test_sgd_single_update():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gradient = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    custom_param = Parameter(data.copy())
    torch_param = torch.nn.Parameter(torch.tensor(data.copy()))

    custom_optimizer = SGD([custom_param], lr=0.01)
    torch_optimizer = torch.optim.SGD([torch_param], lr=0.01)

    custom_param.grad = Tensor(gradient.copy())
    torch_param.grad = torch.tensor(gradient.copy())

    custom_optimizer.step()
    torch_optimizer.step()

    assert_close(custom_param.data, torch_param.data)


def test_adamw_single_update():
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gradient = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    custom_param = Parameter(data.copy())
    torch_param = torch.nn.Parameter(torch.tensor(data.copy()))

    lr = 0.001
    weight_decay = 0.01

    custom_optimizer = AdamW(
        [custom_param],
        lr=lr,
        weight_decay=weight_decay,
    )

    torch_optimizer = torch.optim.AdamW(
        [torch_param],
        lr=lr,
        weight_decay=weight_decay,
    )

    custom_param.grad = Tensor(gradient.copy())
    torch_param.grad = torch.tensor(gradient.copy())

    custom_optimizer.step()
    torch_optimizer.step()

    assert_close(custom_param.data, torch_param.data)


if __name__ == "__main__":
    test_adamw_single_update()
    test_sgd_single_update()