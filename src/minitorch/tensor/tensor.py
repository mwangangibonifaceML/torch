from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def unbroadcast(grad: NDArray, shape: tuple[int]) -> NDArray:
    """Sum `grad` down to `shape`, undoing numpy broadcasting."""
    grad = np.asarray(grad)
    shape = tuple(shape)

    if grad.shape == shape:
        return grad
    if grad.ndim < len(shape):
        raise ValueError(
            f"Cannot unbroadcast gradient of shape {grad.shape} to {shape}: "
            f"the gradient has fewer dimensions than the target. "
            f"The op's backward should broadcast it first."
        )

    # * 1) sum away extra leading dims added by broadcasting
    extra = grad.ndim - len(shape)
    if extra:
        grad = grad.sum(axis=tuple(range(extra)))

    # * 2) sum over dims that were size 1 in the original
    axes = tuple(i for i, s in enumerate(shape) if s == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)

    return grad


class Tensor:
    """A tensor class that tries to mimick a pytorch tensor

    This class starts simple but includes features for future modules:
    - device: To support CPU/GPU tensors in the future.
    - _parents: To track the computational graph for autograd.
    - data: The underlying NumPy array storing the tensor values.
    - requires_grad: Will be used for automatic differentiation.
    - grad: Will store computed gradients.
    - backward(): Will compute gradients (the core idea in autograd).

    This class focuses on: data, shape, and basic operations.
    """

    def __init__(
        self,
        data: NDArray | list[int] | list[float] | float,
        requires_grad=False,
        dtype=np.float32,
        device: str | None = None,
        _parents: tuple[Tensor, ...] = (),
    ) -> None:
        if not isinstance(data, (float, int, list, tuple, np.generic)) and not (
            hasattr(data, "shape") and hasattr(data, "dtype")
        ):
            raise ValueError(
                f"Data must be a float, int, an iterable of either floats or integers or "
                f"a numpy array, you provided {type(data)}"
            )

        if isinstance(data, (float, int, list, tuple)):
            data = np.array(data, dtype=dtype)
        elif hasattr(data, "shape") and hasattr(data, "dtype"):
            data = data.astype(dtype)

        self.data = data
        self.requires_grad = requires_grad
        self.device = device if device is not None else "cpu"
        self._parents = _parents
        self.grad = (
            np.zeros_like(self.data, dtype=np.float32) if requires_grad else None
        )
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Tensor(data={self.data}, requires_grad= {self.requires_grad})"

    def __str__(self) -> str:
        return f"Tensor(data={self.data})"

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @staticmethod
    def _ensure_tensor(x: float | np.ndarray | Tensor) -> Tensor:
        """Check whether an argument is a Tensor, if not
        wrap it in Tensor class
        """
        if isinstance(x, Tensor):
            return x
        else:
            if isinstance(x, (int, float)):
                x = Tensor(np.array(x))
                return x
            else:
                x = Tensor(x)
                return x

    def numpy(self) -> np.ndarray:
        """Return the underlying numpy array"""
        return self.data.copy()

    def copy(self) -> Tensor:
        return Tensor(self.data.copy())

    def _determine_gradient_requirement(self, other: Tensor) -> bool:
        if isinstance(other, Tensor):
            return self.requires_grad or other.requires_grad
        return self.requires_grad

    def __getitem__(self, key):
        """
        Enable Tensor indexing and slicing.
        Supports:
            tensor[i]
            tensor[i:j]
            tensor[:, 1:-1]
            tensor[:, -1]
            tensor[i, j]
        """

        # forward pass
        result_data = self.data[key]

        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)

        result = Tensor(
            result_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if not self.requires_grad:
                return

            grad_input = np.zeros_like(self.data)

            # Scatter gradient back to the indexed positions
            grad_input[key] += result.grad

            self._add_grad(grad_input)

        result._backward = _backward
        return result

    def __len__(self) -> int:
        return len(self.data)

    def __add__(self, other) -> Tensor:
        """Add two tensors element-wise with broadcasting support"""
        other = Tensor(other) if not isinstance(other, Tensor) else other
        result = Tensor(
            self.data + other.data,
            requires_grad=self._determine_gradient_requirement(other),
            _parents=(self, other),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if self.requires_grad and result.grad is not None:
                    self._add_grad(result.grad)

            if other.requires_grad and result.grad is not None:
                    other._add_grad(result.grad)

        result._backward = _backward
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other) -> Tensor:
        """Multiply two tensors element-wise (NOT matrix multiplication)."""
        other = Tensor(other) if not isinstance(other, Tensor) else other
        result = Tensor(
            self.data * other.data,
            requires_grad=self._determine_gradient_requirement(other),
            _parents=(self, other),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if self.requires_grad:
                self._add_grad(result.grad * other.data)

            if other.requires_grad:
                other._add_grad(result.grad * self.data)

        result._backward = _backward
        return result

    def __rmul__(self, other) -> Tensor:
        return self.__mul__(other)

    def __sub__(self, other) -> Tensor:
        """Subtract two tensors element-wise."""
        other = Tensor(other) if not isinstance(other, Tensor) else other
        result = Tensor(
            self.data - other.data,
            requires_grad=self._determine_gradient_requirement(other),
            _parents=(self, other),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if self.requires_grad:
                self._add_grad(unbroadcast(result.grad, self.shape))

            if other.requires_grad:
                other._add_grad(-unbroadcast(result.grad, other.shape))

        result._backward = _backward
        return result

    def __truediv__(self, other) -> Tensor:
        """Divide two tensors element-wise."""
        other = Tensor(other) if not isinstance(other, Tensor) else other
        result = Tensor(
            self.data / other.data,
            requires_grad=self._determine_gradient_requirement(other),
            _parents=(self, other),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if self.requires_grad and result.grad is not None:
                    self._add_grad(result.grad / other.data)

            if other.requires_grad and result.grad is not None:
                    other._add_grad(-result.grad * self.data / other.data**2)

        result._backward = _backward
        return result

    def __matmul__(self, other) -> Tensor:
        """Enable @ operator for matrix multiplication"""
        return self.matmul(other)

    def __neg__(self):
        result = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if self.requires_grad and result.grad is not None:
                    self._add_grad(-result.grad)

        result._backward = _backward
        return result

    def __rsub__(self, other):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        result = Tensor(
            other.data - self.data,
            requires_grad=self._determine_gradient_requirement(other),
            _parents=(other, self),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if other.requires_grad and result.grad is not None:
                    other._add_grad(result.grad)

            if self.requires_grad and result.grad is not None:
                    self._add_grad(-result.grad)

        result._backward = _backward
        return result

    def __rtruediv__(self, other):
        other = Tensor(other) if not isinstance(other, Tensor) else other
        result = Tensor(
            other.data / self.data,
            requires_grad=self._determine_gradient_requirement(other),
            _parents=(self, other),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if other.requires_grad:
                other._add_grad(result.grad / self.data)

            if self.requires_grad and result.grad is not None:
                    self._add_grad(-result.grad * other.data / self.data**2)

        result._backward = _backward
        return result

    def __pow__(self, other):
        if not isinstance(other, (float, int)):
            raise TypeError("Power must be either integer or a float")
        
        result = Tensor(
            self.data**other,
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if self.requires_grad and result.grad is not None:
                self._add_grad(other * (self.data ** (other - 1)) * result.grad)

        result._backward = _backward
        return result

    def matmul(self, other) -> Tensor:
        other = Tensor(other) if not isinstance(other, Tensor) else other
        if (
            self.data.ndim == 0 or other.data.ndim == 0
        ):  # * 1D or scalar case, fallback to element-wise multiplication
            result = Tensor(
                self.data * other.data,
                requires_grad=self._determine_gradient_requirement(other),
                _parents=(self, other),
                dtype=np.float32,
                device=self.device,
            )

        else:  # * 2D or higher case, perform matrix multiplication with shape checks
            if len(self.shape) >= 2 and len(other.shape) >= 2 and self.shape[-1] != other.shape[-2]:
                    raise ValueError(
                        f"Incompatible shapes for matmul: {self.shape} @ {other.shape}"
                    )

            result_data = np.matmul(self.data, other.data)

            result = Tensor(
                result_data,
                requires_grad=self._determine_gradient_requirement(other),
                _parents=(self, other),
                dtype=np.float32,
                device=self.device,
            )

        def _backward():
            if result.grad is None:
                return

            g = result.grad
            a, b = self.data, other.data

            # 0-d operand: forward was element-wise multiplication
            if a.ndim == 0 or b.ndim == 0:
                if self.requires_grad:
                    self._add_grad(g * b)
                if other.requires_grad:
                    other._add_grad(g * a)
                return

            # promote 1-D operands to 2-D so one formula covers every case
            a2 = a[None, :] if a.ndim == 1 else a      # (k,)  -> (1, k)
            b2 = b[:, None] if b.ndim == 1 else b      # (k,)  -> (k, 1)

            # promote the output gradient to match the shape of a2 @ b2
            if a.ndim == 1 and b.ndim == 1:
                g2 = np.reshape(g, (1, 1))
            elif a.ndim == 1:
                g2 = np.expand_dims(g, -2)             # (..., n) -> (..., 1, n)
            elif b.ndim == 1:
                g2 = np.expand_dims(g, -1)             # (..., m) -> (..., m, 1)
            else:
                g2 = g

            # two independent blocks (not nested, no else)
            if self.requires_grad:
                grad_a = g2 @ b2.swapaxes(-1, -2)      # (..., m, k)
                if a.ndim == 1:
                    grad_a = grad_a.squeeze(-2)        # back to (..., k)
                self._add_grad(grad_a)                 # _add_grad unbroadcasts batch dims

            if other.requires_grad:
                grad_b = a2.swapaxes(-1, -2) @ g2      # (..., k, n)
                if b.ndim == 1:
                    grad_b = grad_b.squeeze(-1)        # back to (..., k)
                other._add_grad(grad_b)

        result._backward = _backward
        return result

    def sum(self, axis=None, keepdims=False) -> Tensor:
        """Sum tensor along specified axis"""
        result = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if result.grad is None or not self.requires_grad:
                return

            grad = result.grad

            # If dimensions were removed, restore them
            if axis is not None and not keepdims:
                if isinstance(axis, int):
                    axes = (axis,)
                else:
                    axes = axis

                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)

            # Broadcast to input shape
            grad = np.broadcast_to(grad, self.shape)

            self._add_grad(grad)

        result._backward = _backward
        return result

    def mean(self, axis=None, keepdims=False) -> Tensor:
        result = Tensor(
            np.array(np.mean(self.data, axis=axis, keepdims=keepdims)),
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if result.grad is None or not self.requires_grad:
                return
            axes = self._normalize_axes(axis)
            denom = int(np.prod([self.shape[ax] for ax in axes]))

            grad = result.grad
            if not keepdims:  # * restore the reduced dims
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, ax)

            self._add_grad(np.broadcast_to(grad, self.shape) / denom)

        result._backward = _backward
        return result

    def var(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        result = Tensor(
            np.array(np.var(self.data, axis=axis, keepdims=keepdims, ddof=1)),
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if self.requires_grad and result.grad is not None:
                    self._add_grad(result.grad / self.data.size)

        result._backward = _backward
        return result

    def max(self, axis: int | None = None, keepdims=False) -> Tensor:
        """Find the max value of a tensor along specified axis"""
        return Tensor(
            np.array(np.max(self.data, axis=axis, keepdims=keepdims)),
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=np.float32,
            device=self.device,
        )

    def reshape(self, *shape):
        """Reshape the tensor to a new dimensions"""
        original_shape = self.shape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            new_shape = tuple(shape[0])
        else:
            new_shape = shape
        if -1 in new_shape:
            if new_shape.count(-1) > 1:
                raise ValueError("Can only specify one unknown dimension with -1")
            known_size = 1
            unknown_idx = new_shape.index(-1)
            for i, dim in enumerate(new_shape):
                if i != unknown_idx:
                    known_size *= dim
            unknown_dim = self.size // known_size
            new_shape = list(new_shape)
            new_shape[unknown_idx] = unknown_dim
            new_shape = tuple(new_shape)
        if np.prod(new_shape) != self.size:
            target_size = int(np.prod(new_shape))
            raise ValueError(f"Total elements must match: {self.size} ≠ {target_size}")
        reshaped_data = np.reshape(self.data, new_shape)
        result = Tensor(
            reshaped_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            if self.requires_grad and result.grad is not None:
                    self._add_grad(result.grad.reshape(original_shape))

        result._backward = _backward
        return result

    def transpose(self, dim0=None, dim1=None):
        "A linear transformation that rearranges elements"

        axis = None
                
        if dim0 is not None and dim1 is not None:
            # if dim0 < 0 or dim1 < 0:
            #     raise ValueError('Dimensions must be non-negative')
            if dim0 >= len(self.shape) or dim1 >= len(self.shape):
                raise ValueError("Dimensions must be within the tensor shape")
            if dim0 == dim1:
                raise ValueError("Dimensions must be different")

            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            axis = axes
            transposed_data = np.transpose(self.data, axes)
        else:
            # if dim0 is None and dim1 is None:
            if len(self.shape) < 2:  # * 1D or scalar
                transposed_data = self.data.copy()
            else:  # * default case
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                axis = axes
                transposed_data = np.transpose(self.data, axes)
                
        result = Tensor(
            transposed_data,
            requires_grad=self.requires_grad,
            _parents=(self,),
            dtype=np.float32,
            device=self.device,
        )

        def _backward():
            """

            **Key insight:** The gradient of transpose is just transpose the
            gradient.
            """
            if not self.requires_grad:
                return

            # transpose gradient back
            if result.grad is None:
                return
            grad_input = np.transpose(a=result.grad, axes=axis)
            self._add_grad(grad_input)

        result._backward = _backward
        return result

    def detach(self):
        """Return a new Tensor detached from the current computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)

    def backward(self):
        """Compute gradient"""
        # * build an ordered list of all tensors involved(Topological order)
        # * this ensures that when we call backward, we compute gradients in the correct order
        topo = []
        visited = set()

        def build_graph(root):
            if not root in visited:
                visited.add(root)
                # * recursively visit parents first
                for parent in root._parents:
                    build_graph(parent)

                # * add self after all parents are visited
                topo.append(root)

        build_graph(self)

        # * initialize the gradient of the output tensor (self) to 1,
        # * since d(output)/d(output) = 1
        self.grad = np.ones_like(self.data, dtype=np.float32)

        # * go through the topology in reverse order triggering
        # * the backward function of each node
        for node in reversed(topo):
            node._backward()

    def _add_grad(self, grad: NDArray) -> NDArray | None:
        """Accumulate the gradients"""
        if not self.requires_grad:
            return

        if isinstance(grad, Tensor):
            grad = np.asarray(grad.data)

        if self.grad is None:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

        grad = unbroadcast(grad, self.shape)
        self.grad += grad

    def _normalize_axes(self, axis) -> tuple:
        if axis is None:
            return tuple(range(self.data.ndim))
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        return tuple(ax % self.data.ndim for ax in axes)

    @classmethod
    def from_numpy(
        cls, array: NDArray, requires_grad=False, dtype=np.float32, device=None
    ) -> Tensor:
        """Create a Tensor from a NumPy array."""
        return cls(data=array, requires_grad=requires_grad, dtype=dtype, device=device)

    @classmethod
    def zeros(cls, shape, requires_grad=False, dtype=np.float32, device=None) -> Tensor:
        """Create a Tensor filled with zeros."""
        data = np.zeros(shape, dtype=dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)

    @classmethod
    def ones(cls, shape, requires_grad=False, dtype=np.float32, device=None) -> Tensor:
        """Create a Tensor filled with ones."""
        data = np.ones(shape, dtype=dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)

    @classmethod
    def randn(cls, shape, requires_grad=False, dtype=np.float32, device=None) -> Tensor:
        """Create a Tensor filled with random values from a normal distribution."""
        data = np.random.randn(*shape).astype(dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)

    @classmethod
    def rand(
        cls,
        low: float,
        high: float,
        shape: tuple[int, ...],
        requires_grad=False,
        dtype=np.float32,
        device=None,
    ) -> Tensor:
        """Create a Tensor filled with random values from a uniform distribution."""
        data = np.random.uniform(low, high, size=shape).astype(dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)

    @classmethod
    def arange(
        cls, start, end=None, step=1, requires_grad=False, dtype=np.float32, device=None
    ) -> Tensor:
        """Create a Tensor with values from start to end with a given step."""
        if end is None:
            end = start
            start = 0
        data = np.arange(start, end, step, dtype=dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)

    @classmethod
    def constant(
        cls, value, shape=(), requires_grad=False, dtype=np.float32, device=None
    ) -> Tensor:
        """Create a Tensor filled with a constant value."""
        data = np.full(shape, value, dtype=dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)

    @classmethod
    def tril(
        cls,
        input: Tensor,
        diagonal=0,
        requires_grad=False,
        dtype=np.float32,
        device=None,
    ) -> Tensor:
        """Return the lower triangular part of a matrix (2D tensor) or batch of matrices."""
        if input.data.ndim < 2:
            raise ValueError("Input must be at least 2D for tril operation.")
        data = np.tril(input.data, k=diagonal).astype(dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)

    @classmethod
    def masked_fill(
        cls,
        input: Tensor,
        filler=-np.inf,
        requires_grad=False,
        dtype=np.float32,
        device=None,
    ):
        if input.data.ndim <= 2:
            raise ValueError(
                f"Input Tensor must 3-D, you provided a {input.data.ndim}-D Tensor"
            )
        output_data = input.data.copy()
        mask = np.zeros_like(output_data)
        seq_len = output_data.shape[-1]
        rows, cols = np.triu_indices(seq_len, k=1)
        output_data[..., rows, cols] = filler
        mask[..., rows, cols] = True

        result = cls(
            output_data,
            requires_grad=input.requires_grad,
            _parents=(input,),
            dtype=np.float32,
            device=input.device,
        )

        def _backward():
            if result.grad == None:
                return
            grad = result.grad.copy()
            grad[mask] = 0

            input._add_grad(grad)

        return result

    @classmethod
    def stack(cls, input_tensors: list[Tensor], axis=0, requires_grad=False):
        stack = np.stack(
            [tensor.data for tensor in input_tensors], axis=axis, dtype=np.float32
        )
        return Tensor(stack, requires_grad=requires_grad)

    @classmethod
    def randint(
        cls,
        low: int,
        high: int,
        shape: tuple[int, ...],
        requires_grad=False,
        dtype=np.float32,
        device=None,
    ) -> Tensor:
        """Create a Tensor filled with random integers from a uniform distribution."""
        data = np.random.randint(low, high, size=shape).astype(dtype)
        return cls(data=data, requires_grad=requires_grad, dtype=dtype, device=device)
