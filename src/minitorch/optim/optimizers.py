from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from minitorch.tensor import Tensor
from minitorch.nn import Parameter

DEFAULT_EPS = 1e-8
DEFAULT_BETA1 = 0.9
DEFAULT_BETA2 = 0.999
DEFAULT_SGD_LR = 0.01
DEFAULT_ADAM_LR = 0.001

class Optimizer(object):
    "Base class for all optimizers."
    
    def __init__(self, params: list[Parameter], lr: float) -> None:
        self.params = params
        self.lr = lr  
        self.step_count = 0.0
        self.momentum_buffers = [None for _ in self.params]
        
    def step(self):
        """
        Update the parameters of the model based on the gradient
        Each optimizer implements its own update rule
        """
        
        raise NotImplementedError(
            f"Abstract method step() not implemented\n"
            f"{self.__class__.__name__} inherits from Optimizer but doesn't define step()\n"
            f"Each optimizer must implement its own update rule (SGD, Adam, etc.)\n"
            f"Override step() in your optimizer subclass:\n"
            f"    def step(self):\n"
            f"        for param in self.params:\n"
            f"            if param.grad is not None:\n"
            f"                 param.data -= self.lr * param.grad.data"
        )
        
    def _extract_grad_data(self, gradient: Tensor | NDArray) -> NDArray:
        """Extract the gradient of the parameters passed to the optimizer"""
        if isinstance(gradient, Tensor):
            return gradient.data
        return gradient

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()
    
    
class SGD(Optimizer):
    "Stochastic Gradient Descent."
    
    def __init__(self, params: list[Parameter], lr: float= DEFAULT_SGD_LR,
                momentum: float=0.0, weight_decay: float=0.0) -> None:
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        
    def step(self) -> None:
        """Perform Stochastic Gradient Descent to update the parameters"""
        #* iterate through the params updating them
        for i,param in enumerate(self.params):
            if param is None or param.grad is None:
                continue
            
            #* extract gradient for the current paramter.
            grad_data = self._extract_grad_data(param.grad)
            
            #* apply weight decay for the current parameter
            if self.weight_decay != 0.0:
                grad_data += self.weight_decay * param.data
                
            #* update momentum buffers
            if self.momentum != 0.0:
                if self.momentum_buffers[i] is None:
                    self.momentum_buffers[i] = np.zeros_like(param.data)
                self.momentum_buffers[i] = self.momentum * self.momentum_buffers[i] + grad_data
                grad_data = self.momentum_buffers[i]
                
            #* update parameter:
            param.data -= self.lr * grad_data
        
        self.step_count += 1
        
    def has_momentum(self):
        """
        Check if this optimizer uses momentum

        Returns:
            bool: True if momentum > 0.0, False otherwise
        """
        if self.momentum > 0.0:
            return True
        return False
        
    def set_momentum_state(self, states: list | None) -> None:
        if states is None or not self.has_momentum():
            return 
        
        if len(states) != len(self.momentum_buffers):
            raise ValueError(
                f"Momentum state length mismatch\n"
                f"State has {len(state)} buffers, but optimizer has {len(self.momentum_buffers)} parameters\n"
                f"Checkpoint was saved with a different model architecture or parameter count\n"
                f" Ensure you're loading state into an optimizer with the same number of parameters:\n"
                f"# Check parameter counts match before restoring\n"
                f"assert len(saved_state) == len(optimizer.params)"
            )
        
        for i,buffer in enumerate(states):
            if buffer is not None:
                self.momentum_buffers[i] = buffer.copy()
        
        
class Adam(Optimizer):
    def __init__(self, params: list[Parameter],
                lr: float = DEFAULT_ADAM_LR,
                betas: tuple[float, float] = (DEFAULT_BETA1, DEFAULT_BETA2),
                weight_decay: float=0.0,
                eps: float = DEFAULT_EPS) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas[0], betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]
        
    def step(self)-> None:
        """
        Update the parameters using adaptive learning rate
        
        Approach:
                Does three things for each parameter
                1) extracts gradient from the parameter.
                2) updates moments for adaptice scaling.
                3) updates the parameters.
        
        """
        self.step_count += 1.0
        
        for i, param in enumerate(self.params):
            if param is None or param.grad is None:
                continue
            grad_data = self._extract_grad_data(param)
            
            if self.weight_decay > 0.0:
                grad_data += self.weight_decay * param.data
            m,v = self._update_moments(i, grad_data)
            
            param.data -= self.lr * m / (np.sqrt(v) + self.eps)
                
                
                
    def _update_moments(self, param_index: int, gradient: NDArray) -> tuple(NDArray):
        """
        Update first and second moment estimates with bias correction.
        
        Computes the exponential moving averages(EMA) of the gradient (first moment)
        and the squared gradient (second moment), then applies bias correction
        to counteract the zero-initialization bias in early training steps.
        """
        
        if self.m_buffers[param_index] is None:
            self.m_buffers[param_index] = np.zeros_like(gradient)
            self.v_buffers[param_index] = np.zeros_like(gradient)
            
        self.m_buffers[param_index] = self.beta1 * self.m_buffers[param_index] + (1 - self.beta1) * gradient
        self.v_buffers[param_index] = self.beta2 * self.v_buffers[param_index] + (1 - self.beta2) * gradient ** 2

        m_bias = 1 - self.beta1 ** self.step_count
        v_bias = 1 - self.beta2 ** self.step_count
        
        m_hat = self.m_buffers[param_index] / m_bias
        v_hat = self.v_buffers[param_index] / v_bias
        return m_hat, v_hat
        
        
class AdamW(Optimizer):
    def __init__(self, params: list[Parameter],
                lr: float = DEFAULT_ADAM_LR,
                betas: tuple[float, float] = (DEFAULT_BETA1, DEFAULT_BETA2),
                weight_decay: float=0.0,
                eps: float = DEFAULT_EPS) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas[0], betas[1]
        self.weight_decay = weight_decay
        self.eps = eps
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]
        
    def step(self)-> None:
        """
        Update the parameters using adaptive learning rate
        
        Approach:
                Does three things for each parameter
                1) extracts gradient from the parameter.
                2) updates moments for adaptice scaling.
                3) updates the parameters.
        
        """
        self.step_count += 1.0
        
        for i, param in enumerate(self.params):
            if param is None or param.grad is None:
                continue
            grad_data = self._extract_grad_data(param)
            m,v = self._update_moments(i, grad_data)
            
            param.data -= self.lr * m / (np.sqrt(v) + self.eps)
            if self.weight_decay > 0.0:
                param.data *= (1 - self.lr * self.weight_decay)
                
    def _update_moments(self, param_index: int, gradient: NDArray) -> tuple(NDArray):
        """
        Update first and second moment estimates with bias correction.
        
        Computes the exponential moving averages(EMA) of the gradient (first moment)
        and the squared gradient (second moment), then applies bias correction
        to counteract the zero-initialization bias in early training steps.
        """        
        if self.m_buffers[param_index] is None:
            self.m_buffers[param_index] = np.zeros_like(gradient)
            self.v_buffers[param_index] = np.zeros_like(gradient)
            
        self.m_buffers[param_index] = self.beta1 * self.m_buffers[param_index] + (1 - self.beta1) * gradient
        self.v_buffers[param_index] = self.beta2 * self.v_buffers[param_index] + (1 - self.beta2) * gradient ** 2

        m_bias = 1 - self.beta1 ** self.step_count
        v_bias = 1 - self.beta2 ** self.step_count
        
        m_hat = self.m_buffers[param_index] / m_bias
        v_hat = self.v_buffers[param_index] / v_bias
        return m_hat, v_hat
        
        

        
if __name__ == "__main__":
    import numpy as np
    from minitorch.nn import Linear, Sequential
    from minitorch.activations import Tanh
    
    batch, in_feat, out_feat, seed = 5,3,2,42
    rng = np.random.default_rng(seed)
    inputs = Tensor(rng.standard_normal(size=(batch, in_feat)))
    
    nn = Sequential(Linear(in_feat, out_feat, bias=False), Linear(out_feat, 2, bias=False))
    params = nn.parameters()
    optim = Adam(params, lr=0.001)
    tanh = Tanh()
    
    
    print("Before backward pass\n")
    for param in params:
        print("Parameter\n")
        print(param)
        print("\ngradients")
        print(param.grad)
        print("\n")
        
    pre_act = nn(inputs)
    act = tanh(pre_act)
    act.backward()
    optim.step()
    
    print("After backward pass\n")
    for param in params:
        print("Parameter\n")
        print(param)
        print("\ngradients")
        print(param.grad)
        print("\n")
        
    print(optim.step_count)