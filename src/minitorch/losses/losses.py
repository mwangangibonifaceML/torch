from typing import Any

import numpy as np

from minitorch.nn import Linear
from minitorch.tensor import Tensor

EPSILON =  1e-7
_REDUCTIONS = ("mean", "sum", "none")

def _check_reduction(reduction: str) -> str:
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction}")
    return reduction


def _reduce(values: np.ndarray, reduction: str):
    if reduction == "mean":
        values.mean(axis=0)
    if reduction == "sum":
        values.sum()
    return values


def _upstream(loss: Tensor, reduction: str, n:int):
    return loss.grad / n if reduction == "mean" else loss.grad


def log_softmax(tensor: Tensor, dim: int | None = -1) -> Tensor:
    """Comptute log-softmax with numerical stability

    Args:
        tensor (Tensor): Input tensor to perform log-softmax
        dim (Optional[int], optional): dimension to perform log-softmax. Defaults to None

    Returns:
        Tensor: log-softmax tensor

    """
    #* find the max value in the tensor
    x = tensor.data
    max_val = np.max(x, axis=dim, keepdims=True)
    
    #* subtract the max to prevent numerical overflow
    #* then calculate the log-softmax
    x_shifted = x - max_val
    exp_x = np.exp(x_shifted)
    exp_x_sum = np.sum(exp_x, axis=dim, keepdims=True)
    
    log_softmax_data = np.log(exp_x_sum + EPSILON)

    #* return log-softmax = tensor - max_val - log_softmax_data
    return Tensor(x_shifted - log_softmax_data, requires_grad=True, _parents=(tensor,))


def stable_softmax(logits_data):
    """compute softmax probabilities with numerical stability
    
    Subtracts the max value per row before exponentiating to prevent overflow
    
    """
    max_logits = np.max(logits_data, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_data - max_logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

class Loss:
    "Base class inherited by all losses."
    
    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = _check_reduction(reduction)
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
        
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Forward pass throught the loss functions.

        Returns:
            Tensor: The output tensor.
        """
        raise NotImplementedError("Forward method not implemented.")
    
class MSE(Loss):
    "Mean Squared Error for regression tasks."
    
    def forward(self, predictions: Tensor, targets: Tensor) -> np.any:
        predictions = Tensor._ensure_tensor(predictions)
        targets = Tensor._ensure_tensor(targets)
        
        if predictions.shape != targets.shape:
            raise ValueError(f"predictions {predictions.shape} and targets {targets.shape} should have the same shape.")
        
        squared_error = np.mean((predictions.data - targets.data) ** 2)
        inputs = (predictions, targets)
        require_grad = any(t.requires_grad for t in inputs)
        out = Tensor(squared_error,
                    requires_grad=require_grad,
                    _parents=tuple(t for t in inputs if t.requires_grad),
                    dtype=inputs[0].dtype,
                    device=inputs[0].device
                    )
        
        if out.requires_grad:
            def _backward():
                if out.grad is None:
                    return 
                predictions.data.shape[0]
                mse_grad = 2 * (
                    predictions.data - targets.data
                    ) * _upstream(out, self.reduction, predictions.data.size)
                
                if predictions.requires_grad:
                    predictions._add_grad(mse_grad)
                if targets.requires_grad:
                    targets._add_grad(-mse_grad)
                
            out._backward = _backward
        return out
    
class SoftMaxCrossEntropy(Loss):
    "Cross-Entropy loss for multi-class classififcation."
        
    def forward(self, logits: Tensor, targets: Tensor) -> np.any:        
        logits = Tensor._ensure_tensor(logits)
        targets = Tensor._ensure_tensor(targets)
        
        inputs = (logits, targets)
        
        #* calculate log probs
        logits_reshape = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        n = logits_reshape.shape[0]
        
        if targets.size != n:
            raise ValueError(
                f"targets {targets.shape} must hold one class index per row of logits {logits_reshape.shape} "
                f"(expected {n} indices, got {targets.size}); one-hot targets are not supported"
            )
            
        log_probs = log_softmax(logits_reshape, dim=-1)
        
        #* get batch_size and make targets ints
        batch_size, num_classes = log_probs.shape[0], log_probs.shape[1]
        target_indices = targets.data.astype(int)
        
        if n and target_indices.min() < 0 or target_indices.max() >= num_classes:
            raise ValueError(
                f"target indices must be in range (0 {num_classes - 1})."
            )
        
        
        #* get selected log probs
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]
        
        #* calculate loss
        neg_log_probs = -np.mean(selected_log_probs, keepdims=False) #* pytorch's NLLoss
        
        loss = Tensor(neg_log_probs,
                    requires_grad=logits.requires_grad,
                    _parents=tuple(t for t in inputs if t.requires_grad),
                    device =inputs[0].device,
                    dtype=inputs[0].dtype)
    
        if loss.requires_grad:
            def _backward():
                    if loss.grad is None:
                        return 
                    
                    loss_grad = _upstream(loss, self.reduction, n)
                    stable_probs = stable_softmax(logits_reshape.data)
                    one_hot_encode = np.zeros((batch_size, num_classes), dtype=np.float32)
                    one_hot_encode[np.arange(batch_size), target_indices] = 1.0
                    grad_logits = (stable_probs - one_hot_encode) / batch_size
                    logits_grad = (grad_logits * loss_grad).reshape(logits.shape)
                    logits._add_grad(logits_grad)
                
            loss._backward = _backward
        return loss


class BCEWithLogits(Loss):
    "Binary cross-entropy loss for binary classification."
    
    def forward(self, logits: Tensor, targets: Tensor) -> np.any:
        logits = Tensor._ensure_tensor(logits)
        targets = Tensor._ensure_tensor(targets)
        
        inputs = (logits, targets)
        n = logits.data.size
        if logits.shape != targets.shape:
            raise ValueError(f"logits {logits.shape} and targtes {targets.shape} must have the same shape.")
        
        #* apply sigmoid to logits to get probabilities
        #* numerical stable sigmoid
        sigmoid_logits = 1 / (1 + np.exp(-np.clip(logits.data, -500, 500)))
        
        #* clip the probabilities to avoid log(0)
        probs_clipped = np.clip(sigmoid_logits, EPSILON, 1 - EPSILON)
        
        #* calculate the binary cross entropy
        #* bce_per_sample = -[y * log(p) + (1-y) * log(1-p)]
        bce_per_sample = -np.sum(targets.data * np.log(probs_clipped) + \
        (1 - targets.data) * np.log(1 - probs_clipped))
        
        require_grad = any(t.requires_grad for t in inputs)
        loss = Tensor(bce_per_sample, 
                    requires_grad=require_grad,
                    _parents=tuple(t for t in inputs if t.requires_grad),
                    device=inputs[0].device,
                    dtype=inputs[0].dtype)
        
        if loss.requires_grad:
            def _backward():
                if loss.grad is None:
                    return 
                
                loss_grad = _upstream(loss, self.reduction, n)
                #* gradient w.r.t probabilities
                prob_grad = -(targets.data / probs_clipped) + ((1 - targets.data) / (1 - probs_clipped))
                #* gradient w.r.t logits: prob_grad * sigmoid'(logits)
                sigmoid_deriv = sigmoid_logits * (1 - sigmoid_logits)
                logits_grad = prob_grad * sigmoid_deriv
                if logits.requires_grad:
                    logits._add_grad(logits_grad * loss_grad)
                if targets.requires_grad:
                    targets._add_grad(logits.data * loss_grad)
                
            loss._backward = _backward
        return loss
    
    
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    batch, in_feat, out_feat = 32,10,2
    max_epochs = 5000
    eval_iters = 100
    
    x = Tensor(np.random.rand(batch, in_feat))
    x = x.reshape(-1, x.shape[-1])
    targets = Tensor(np.array([np.random.randint(0,out_feat, size=(batch))]))
    targets = targets.reshape(-1)
    

    nn = Linear(in_feat, out_feat, bias=True)
    loss_fn = SoftMaxCrossEntropy(reduction="none")
    params = nn.parameters()
    
    for epoch in range(1,max_epochs+1):
        for param in params:
            param.zero_grad()
        
        predictions = nn(x)
        loss = loss_fn(predictions, targets)
        loss.backward()
        
        for param in params:
            param.data -= 0.1 * param.grad
        
        if (epoch % eval_iters) == 0 or epoch ==1:
            print(f"Epoch: {epoch}, loss: {loss}")
    
    # predictions = nn(x)
    print(predictions.requires_grad, targets.requires_grad)
    print(predictions.grad, targets.grad)