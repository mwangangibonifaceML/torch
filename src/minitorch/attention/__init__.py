"""
Attention - Learning to focus
"""
from .attention import compute_scaled_attention_scores, apply_mask, scaled_dot_product_attention, MultiHeadAttention

__all__ = ["compute_scaled_attention_scores", "apply_mask", "scaled_dot_product_attention", "MultiHeadAttention"]