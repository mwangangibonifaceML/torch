from __future__ import annotations

import math
import numpy as np

from typing import Iterator, Any
from minitorch.tensor import Tensor

EPSILON = 1e-5
DROPOUT_MIN_PROB = 0.0
DROPOUT_MAX_PROB = 1.0

def _walk(value, kind) -> list[Module]:
    """
    A simple recursive function that gets all the instances of value found in kind
    by searching through it's attributes, lists, tuples and dictionaries
    """
    if isinstance(value, kind):
        yield value
    
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v, kind)
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item, kind)
    
class Parameter(Tensor):
    """
    A trainable Tensor (weights and biases). Always requires grad
    """
    def __init__(self, data: np.ndarray) -> None:
        super().__init__(data=data, requires_grad=True)
        
    def zero_grad(self):
        if self.grad is not None:
            self.grad = np.zeros_like(self.data)
            
    def detach(self):
        raise RuntimeError(
            'Cannot detach a Parameter from the computational graph.',
            'Convert to Tensor if intentional'
        )
        
    def __repr__(self) -> str:
        return (
            f'Parameter(Tensor(data={self.data}), \n'
            f'shape={self.shape}, \n'
            f'requires_grad={self.requires_grad}'
                )
        

class Module:
    """Base class for all layers in the neural network.
    
    All layers should inherit from this class and implement the forward and backward methods.
    
    Methods:
        forward(input): Computes the forward pass of the layer.
        parameters(): Returns the parameters of the layer.
        The __call__ method is provided to allow instances of Layer to be called like functions.
        zero_grad(): Resets all the gradients to zero.
    """
    def __init__(self) -> None:
        self.training =True
        
    def eval(self, mode: bool=False):
        """Sets the layers to evaluation mode."""
        return self.train(mode)
        
    def train(self, mode: bool = True):
        """Sets the layers to training mode."""
        for module in self.modules():
            module.training = mode
        return self
    
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass throught the layer.
        
        Args:
            input (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor.
        """
        raise NotImplementedError("Forward method not implemented.")
    
    def __call__(self, inputs: Tensor,  *args: Any, **kwds: Any) -> Tensor:
        """Allows the layer to be called like a function.
        
        Args:
            input (Tensor): The input tensor.
            *args: Additional positional arguments.
            **kwds: Additional keyword arguments.
        
        Returns:
            Tensor: The output tensor from the forward pass.
        """
        return self.forward(inputs, *args, **kwds)
    
    def _children(self) -> Iterator[Module]:
        for value in self.__dict__.values():
            yield from _walk(value, Module)
    
    def modules(self) -> list[Module]:
        "self + all descedants, depth first, each descedant once"
        seen: set[int] = set()
        stack: list[Module] = [self]
        out: list[Module] = []
        
        while stack:
            module = stack.pop()
            if id(module) in seen:
                continue
            seen.add(id(module))
            out.append(module)
            stack.extend(reversed(list(self._children())))
        return out
    
    def parameters(self) -> list[Parameter]:
        """Returns the unique parameters of the module and it's descendants.
        
        Returns:
            list[Tensor]: A list of tensors representing the parameters of the module.
        """
        
        seen: set[int] = set()
        params: list[Parameter] = []
        
        for module in self.modules():
            for value in module.__dict__.values():
                for p in _walk(value, Parameter):
                    if id(p) not in seen:
                        seen.add(id(p))
                        params.append(p)
        return params
        
    def zero_grad(self):
        for p in self.parameters():
            p.grad = p.zero_grad()
    
    def __repr__(self) -> str:
        """String representation of the layer."""
        return f"{self.__class__.__name__}()"
    
    
class Linear(Module):
    """A fully connected linear layer: implements: y = xW + b. 
    
    Applies a linear transformation to the incoming data.
    
    Approach:
        - create weight matrix of shape (in_features, out_features)
        - create bias vector of shape (out_features), initialized to zeros if bias=True
        - set requires_grad=True for both weight and bias tensors
        
    HINTS:
        - Xavier init: scale = sqrt(6/in_features+out_features)
        - use np.random.randn to create weight and bias tensors
        -
    
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        
    Attributes:
        weights (Parameter): The learnable weights of the module of shape
                            `(in_features, out_features)`.
        bias (Parameter):   The learnable bias of the module
                            of shape `(out_features,)`.
    
    Methods:
        forward(input): Computes the forward pass of the linear layer.
        parameters(): Returns the parameters of the linear layer.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool= False) -> None:
        super().__init__()
        
        if (in_features <=0) or (out_features <=0):
            raise ValueError(
                "Input and out features must be positive."
            )
            
        self.in_features = in_features
        self.out_features = out_features
        
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weights = Parameter(
            np.random.uniform(-limit, limit, size=(in_features, out_features))
        )
        
        self.bias = Parameter(np.zeros(out_features)) if bias else None
    
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the linear layer.
        
        Args:
            input (Tensor): The input tensor of shape (batch_size, in_features).
        
        Returns:
            Tensor: The output tensor of shape (batch_size, out_features).
        """
        if self.bias is not None:
            return inputs @ self.weights + self.bias
        return inputs @ self.weights
    
    def __repr__(self) -> str:
        """String representation of the linear layer."""
        bias_str = "True" if self.bias is not None else None
        return (f"Linear(in_features={self.in_features}, out_features= {self.out_features}, "
                f"bias={bias_str})")
    

class Dropout(Module):
    """
    Initialize dropout layer.

    """
    def __init__(self, p: float) -> None:
        super().__init__()
        if not (DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB):
            raise ValueError(f"Dropout probability must be in the range [{DROPOUT_MIN_PROB}, {DROPOUT_MAX_PROB}), got {p}.")
        self.p = p
        
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through the dropout layer.
        
        During training, randomly sets a fraction p of inputs to zero.
        During inference, passes the input unchanged.
        
        Approach:
            - If training mode=False or p=0, return input unchanged
            - If p=1, return a tensor of zeros with the same shape as input
            - If training mode = True and p > 0, set random elements to 0 with
            probability p, and scale remaining elements by 1/(1-p)
            - If in evaluation mode, return input unchanged
        Hints:
            - Use np.random.random() < keep_prob to create dropout mask
            - training mode=False should return input unchanged
        """
            
        #* during evaluation or no dropout, return input unchanged
        if not self.training or self.p == DROPOUT_MIN_PROB:
            return inputs
        
        if self.p == DROPOUT_MAX_PROB:
            return inputs * 0.0
        
        #* during training, apply dropout
        keep_prob = DROPOUT_MAX_PROB - self.p
        mask = (np.random.random(inputs.data.shape) < keep_prob).astype(inputs.data.dtype)
        
        #* apply mask and scale to preserve gradients
        mask_tensor = Tensor(mask.astype(np.float32), requires_grad=False)
        scale = 1.0 / keep_prob
        output = (inputs * mask_tensor) * scale
        return output
    
    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"
    
class Sequential(Module):
    """" 
    Chain Module and Layers together. Accepts Sequential(a,b) and Sequential[a,b].
    
    """
    def __init__(self, *layers: Tuple[Layer] | list[Layer]) -> None:
        """Initialize the Sequential container with a list of layers.
        Accepts both (layer1, layer2, ...) and ([layer1, layer2, ...]) formats.
        """
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers = list(layers[0])
        else:
            self.layers = list(layers)
            
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass through all layers in sequence."""
        current = inputs
        for layer in self.layers:
            current = layer(current)
        return current
        
    def __repr__(self) -> str:
        layers_repr = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {layers_repr}\n)"
    
    
class Residual(Module):
    "y = X + fn(X)"
    
    def __init__(self, fn: Linear) -> None:
        super().__init__()
        self.fn = fn
        
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.fn(inputs)
    
    def __repr__(self) -> str:
        return f"Residual({self.fn})"
    

class LayerNormalization(Module):
    """
    Normalizes across the last dimension of the input tensor. Built purely from 
    differentiable Tensor operations.
    """
    def __init__(self, dim: int, eps: float= EPSILON) -> None:
        super().__init__()
        if not isinstance(dim, (int, np.integer)) or dim <= 0:
            raise ValueError(
                f"dim must be a positive integer, got {dim}"
            )
            
        self.dim = int(dim)
        self.eps = eps
        
        #* Learnable parameters: scale and shift
        self.gamma = Parameter(np.ones(shape=self.dim))
        self.beta = Parameter(np.zeros(shape=self.dim))
        
    def forward(self, X: Tensor) -> Tensor:
        if X.shape[-1] != self.dim:
            raise ValueError(
                f'last dim of input should equal dim, received: {X.shape[-1]} expected: {self.dim}'
            )
            
        mean = X.mean(axis=-1, keepdims=True)
        centred = X - mean
        var = (centred * centred).mean(axis=-1, keepdims=True)
        std = (var + self.eps) ** 0.5
        
        norm = centred / std
        shifted_norm = self.gamma * norm + self.beta #* auto backward from Tensor operations
                                                    #* therefore gamma.grad, beta.grad and norm.grad calculated automatically
        
        return shifted_norm
    
    def __repr__(self) -> str:
        return f'LayerNormalization(dim={self.dim})'
    
    
class BatchNormalization(Module):
    """
    Normalizes across the first dimension. 
    
    Supports 2D inputs of shape (batch_size, dim)
    """
    def __init__(
        self,
        dim: int,
        eps: float = EPSILON,
        momentum: float = 0.1
        ) -> None:
        
        super().__init__()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError(
                f"dim must be a positive integer, got {dim}"
            )
            
        #* store the dims, epsilon and the momentum
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        #* learnable parameters
        self.gamma = Parameter(np.ones(shape=self.dim))
        self.beta = Parameter(np.zeros(shape=self.dim))
        
        #* running mean and var (statistics not learnable)
        self.running_mean = Tensor.zeros(shape=(self.dim,))
        self.running_var = Tensor.ones(shape=(self.dim,))
    
    def forward(self, X: Tensor) -> Tensor:
        if len(X.shape) != 2:#, \
            raise ValueError(
                f'Batch Normalization only supports 2D inputs (N,C), got shape {X.shape}'
            )
            
        if X.shape[1] != self.dim:
            raise ValueError(
                f"Expected: {self.dim} features, got: {X.shape[1]}"
            )
            
        if self.training:
            batch_mean = X.mean(axis=0, keepdims=True)
            centred = X - batch_mean
            batch_var = (centred * centred).mean(axis=0, keepdims=True)
            
            #* update running statistics
            self.running_mean = (
                (1 - self.momentum) * self.running_mean +
                self.momentum * batch_mean.detach().data.reshape(-1)
            )
            
            self.running_var = (
                (1 - self.momentum) * self.running_var + 
                self.momentum * batch_var.detach().data.reshape(-1)
            )
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean.reshape((1,-1))
            var = self.running_var.reshape((1,-1))
        
        x_hat = (X - mean) / ((var + self.eps) **0.5)
        out = self.gamma * x_hat + self.beta
        return out

        
class Flatten(Module):
    "(N,d1,d2, ...) => (N *d1 * d2 * ....)"
    
    def __init__(self)-> None:
        super().__init__()
    
    def forward(self, inputs: Tensor) -> Tensor:
        n = inputs.shape[0]
        return inputs.reshape(n, int(np.prod(inputs.shape[1:])))

    def __repr__(self) -> str:
        return "Flatten()"
    
    
    


x = (np.random.randn(3, 5, 8) * 4 + 7).astype(np.float32)
t = Tensor(x)

def diff(name, tensor, ref):
    print(f"{name:12s} shape={tensor.shape}  max err={np.abs(tensor.data - ref).max():.2e}")

m_neg = t.mean(axis=-1, keepdims=True)
m_pos = t.mean(axis=2, keepdims=True)
ref_m = x.mean(-1, keepdims=True)
diff("mean axis=-1", m_neg, ref_m)
diff("mean axis=2", m_pos, ref_m)

c = t - m_pos
diff("X - mean", c, x - ref_m)

v = (c * c).mean(axis=2, keepdims=True)
diff("var", v, ((x - ref_m) ** 2).mean(-1, keepdims=True))

r = (v + 1e-5) ** 0.5
diff("sqrt", r, np.sqrt(((x - ref_m) ** 2).mean(-1, keepdims=True) + 1e-5))

diff("divide", c / r, (x - ref_m) / np.sqrt(((x - ref_m) ** 2).mean(-1, keepdims=True) + 1e-5))