"""
Measuring how wrong the Machine Learning model is wrong.
"""

from .losses import MSE, BCEWithLogits, SoftMaxCrossEntropy, log_softmax, stable_softmax

__all__ = ["MSE", "BCEWithLogits", "SoftMaxCrossEntropy", "log_softmax", "stable_softmax"]