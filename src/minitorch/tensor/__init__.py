"""
A Tensor class that provides the core data structure for
all Machine Learning operations. It wraps a numpy array and adds
deep learning capabilities (like auto differentiation and gradient tracking).

"""

from .tensor import Tensor

__all__ = ["Tensor"]
