
import numpy as np
import torch

from minitorch.tensor import Tensor
from minitorch.losses import MSE, SoftMaxCrossEntropy
from minitorch.conftest import assert_close, assert_same_grad


def test_mse_loss():
    rng = np.random.default_rng(42)

    pred_data = rng.standard_normal((8, 4)).astype(np.float32)
    target_data = rng.standard_normal((8, 4)).astype(np.float32)

    pred_custom = Tensor(pred_data.copy(), requires_grad=True)
    pred_torch = torch.tensor(pred_data.copy(), requires_grad=True)

    target_custom = Tensor(target_data.copy())
    target_torch = torch.tensor(target_data.copy())

    custom_loss = MSE().forward(pred_custom, target_custom)
    torch_loss = torch.nn.functional.mse_loss(
        pred_torch, target_torch
    )

    assert_close(custom_loss, torch_loss)

    custom_loss.backward()
    torch_loss.backward()

    assert_same_grad(pred_custom, pred_torch)


def test_cross_entropy():
    rng = np.random.default_rng(42)

    logits_data = rng.standard_normal((8, 10)).astype(np.float32)
    targets_data = rng.integers(0, 10, size=(8,))

    logits_custom = Tensor(logits_data.copy(), requires_grad=True)
    logits_torch = torch.tensor(logits_data.copy(), requires_grad=True)

    targets_custom = Tensor(targets_data)
    targets_torch = torch.tensor(targets_data, dtype=torch.long)

    custom_loss = SoftMaxCrossEntropy().forward(
        logits_custom, targets_custom
    )

    torch_loss = torch.nn.functional.cross_entropy(
        logits_torch, targets_torch
    )

    assert_close(custom_loss, torch_loss)

    custom_loss.backward()
    torch_loss.backward()

    assert_same_grad(logits_custom, logits_torch)


def test_cross_entropy_sequence_logits():
    rng = np.random.default_rng(42)

    logits_data = rng.standard_normal((2, 4, 10)).astype(np.float32)
    targets_data = rng.integers(0, 10, size=(2, 4))

    logits_custom = Tensor(logits_data.copy(), requires_grad=True)
    logits_torch = torch.tensor(logits_data.copy(), requires_grad=True)

    targets_custom = Tensor(targets_data)
    targets_torch = torch.tensor(targets_data, dtype=torch.long)

    custom_loss = SoftMaxCrossEntropy().forward(
        logits_custom, targets_custom
    )

    torch_loss = torch.nn.functional.cross_entropy(
        logits_torch.reshape(-1, 10),
        targets_torch.reshape(-1),
    )

    assert_close(custom_loss, torch_loss)

    custom_loss.backward()
    torch_loss.backward()

    assert_same_grad(logits_custom, logits_torch)
