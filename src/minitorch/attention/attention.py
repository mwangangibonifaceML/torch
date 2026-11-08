from __future__ import annotations

import math
from turtle import forward
import numpy as np 

from minitorch.tensor import Tensor
from minitorch.activations import Softmax
from minitorch.nn import Linear, Module

def compute_scaled_attention_scores(query: Tensor, key: Tensor, head_dim:int=1) -> Tensor:
    """
    Compute the scaled attention scores for a single head.
    
    Args:
        query: A tensor of shape (batch_size, seq_length, head_dim) representing the query vectors.
        key: A tensor of shape (batch_size, seq_length, head_dim) representing the key vectors.
        head_dim: An integer representing the dimension of each head.

    Returns:
        A tensor of shape (batch_size, seq_length, head_dim) representing the attention scores.
    """
    scores = query @ key.transpose(-2,-1)
    return scores / math.sqrt(head_dim)

def apply_mask(scores: Tensor, mask: Tensor) -> Tensor:
    """Apply a mask to the attention scores to prevent attending to future tokens.
    
    Args:
        scores: A tensor of shape (batch_size, seq_length, seq_length) representing the attention scores.

    Returns:
        A tensor of shape (batch_size, seq_length, seq_length) representing the masked attention scores.
    """
    if np.any((mask.data != 0) & (mask.data != 1)):
        raise ValueError("Attention mask must contain only 0 (blocked) or 1 (allowed)")
    allowed = np.broadcast_to(mask.data, scores.shape) == 1
    if np.any(np.sum(allowed, axis=-1) == 0):
        raise ValueError("Each query must have at least one allowed key")
    return scores.masked_fill(~allowed, -1e9)

def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """
    Compute the scaled dot-product attention.
    
    Args:
        query: A tensor of shape (batch_size, seq_length, head_dim) representing the query vectors.
        key: A tensor of shape (batch_size, seq_length, head_dim) representing the key vectors.
        value: A tensor of shape (batch_size, seq_length, head_dim) representing the value vectors.

    Returns:
        A tuple of output, attention_weights where: output is of shape (batch_size, seq_length, head_dim) representing the
        output of the attention mechanism, and attention_weights 
        is of shape (batch_size, seq_length, seq_length) representing the attention weights.
    """
    head_dim = query.shape[-1]
    scores = compute_scaled_attention_scores(query, key, head_dim)
    if mask is not None:
        scores = apply_mask(scores, mask)
    attention_weights = Softmax()(scores)
    output = attention_weights @ value
    return output, attention_weights


class MultiHeadAttention(Module):
    """
    MultiHead Attention mechanism, which run multiple heads
    in parallel, each head learning a different relationship
    from the other heads.
    """
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """
        Initialize multi-head attention.

        Set up linear projections and validate configuration

        Args:
            embed_dim (int): Embedding dimension of the input and output tensors
            n_heads (int): Number of parallel heads to run in the attention mechanism
        """
        super().__init__()
        if not isinstance(embed_dim, (int, np.integer)) or embed_dim <= 0:
                    raise ValueError("embed_dim must be a positive integer")
        if not isinstance(num_heads, (int, np.integer)) or num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Multi-head attention dimension mismatch\n"
                f"  ❌ embed_dim={embed_dim} is not divisible by num_heads={num_heads} (remainder={embed_dim % num_heads})\n"
                f"  💡 Multi-head attention splits embed_dim equally among heads, so embed_dim must be a multiple of num_heads\n"
                f"  🔧 Try: embed_dim={num_heads * (embed_dim // num_heads + 1)} (next valid size) or num_heads=1"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        #* linear projections for Q,K and V
        self.q_proj = Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = Linear(embed_dim, embed_dim, bias=True)

        # Output projection to mix information across heads
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)
        
    def _split_heads(self, X: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        Reshape to separate attention heads for parallel processing.

        TODO: Reshape (batch, seq, embed_dim) to (batch, heads, seq, head_dim)
        """
        #* input X has shape ---> (batch_size, seq_len, embed_dim)
        X = X.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return X.transpose(1,2)
    
    def _merge_heads(self, X: Tensor, batch_size: int, seq_len: int) -> Tensor:
        "Merge attention heads back into single embedding dimension."
        X = X.transpose(1,2)
        return X.reshape(batch_size, seq_len, self.embed_dim)
    
    def forward(self, X: Tensor, mask: Tensor | None= None) -> Tensor:
        "Forward pass through the multi-head attention."
        if len(X.shape) == 2:
            X = X.reshape(1, X.shape[0], X.shape[-1])
            
        batch_size, seq_len, embed_dim = X.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"embed dim mismatch, input has {embed_dim}, MultiHeadAttention expected {self.embed_dim}\n"
                f"The last dimension of input must match embed_dim from initialization of MultiHeadAttention")
            
        #* get the query, key and value matrices
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        
        #* split the heads
        Q = self._split_heads(Q, batch_size, seq_len) #* (batch_size, seq_len, embed_dim) --> (batch_size, heads, seq_len, head_dim)
        K = self._split_heads(K, batch_size, seq_len)
        V = self._split_heads(V, batch_size, seq_len)
        
        #* apply attention
        mask_reshaped = mask
        if mask is not None and len(mask.shape) == 3:
            batch_size_mask, seq_len_mask, embed_dim_mask = mask.shape
            mask_reshaped = mask.reshape(batch_size_mask, 1, seq_len_mask, embed_dim_mask)
        #* attended shape ---> (batch_size, heads, seq_len, head_dim)
        attended,_ = scaled_dot_product_attention(Q, K, V, mask=mask_reshaped)
        
        #* merge heads back together ---> (batch_size, heads, seq_len, head_dim) --> (batch_size, seq_len, embed_dim)
        concated_output = self._merge_heads(attended, batch_size, seq_len)
        output = self.out_proj(concated_output)
        return output