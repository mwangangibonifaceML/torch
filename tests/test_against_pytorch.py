import numpy as np
import torch

from minitorch.tensor import Tensor

# Your engine
A = Tensor(np.random.randn(2, 3, 4).astype(np.float32),requires_grad=True)

B = Tensor(np.random.randn(2, 4, 5).astype(np.float32),requires_grad=True)

C = A @ B

loss = C.sum()

loss.backward()

grad_A = A.grad.copy()
grad_B = B.grad.copy()


# PyTorch reference
A_torch = torch.tensor(A.data, requires_grad=True)

B_torch = torch.tensor(B.data, requires_grad=True)

C_torch = A_torch @ B_torch

loss_torch = C_torch.sum()

loss_torch.backward()


# Compare
np.testing.assert_allclose(grad_A,A_torch.grad.numpy(),rtol=1e-4,atol=1e-5)

np.testing.assert_allclose(grad_B,B_torch.grad.numpy(),rtol=1e-4,atol=1e-5)

print("Matmul gradients match PyTorch!")