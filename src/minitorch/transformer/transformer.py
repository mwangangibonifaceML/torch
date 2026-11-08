from __future__ import annotations

import numpy as np

from minitorch.attention import MultiHeadAttention
from minitorch.activations import GELU
from minitorch.embeddings import EmbeddingLayer
from minitorch.tensor import Tensor
from minitorch.nn import Linear, Module, LayerNormalization, Dropout, Sequential


rng = np.random.default_rng(seed=42)  # For reproducibility in tests
    
def create_causal_mask(seq_length: int) -> Tensor:
    """
    Create a causal mask for self-attention to prevent attending to future tokens.

    Args:
        seq_length: An integer representing the length of the sequence.
    Returns:
        A tensor of shape (1, seq_length, seq_length) representing the causal mask.
    """
    if not isinstance(seq_length, (int, np.integer)) or seq_length <= 0:
        raise ValueError(f"seq_length must be a positive integer, got {seq_length}")
    
    mask = np.tril(np.ones((seq_length, seq_length), dtype=np.float32))
    return Tensor(mask[np.newaxis, :, :])  # Add batch dimension
    
    
class MLP(Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) for transformer blocks.

    Standard pattern: Linear -> GELU -> Linear with expansion ratio of 4:1.
    This provides the non-linear transformation in each transformer block.
    """
    def __init__(self, embed_dim: int, hidden_dim: int | None = None, dropout: float =0.0) -> None:
        "Initialize MLP with two linear layers."
        super().__init__()
        if not isinstance(embed_dim, (int, np.integer)) or embed_dim <= 0:
            raise ValueError(f"embed_dim must be a positive integer, got {embed_dim}")
        if hidden_dim is not None and (not isinstance(hidden_dim, (int, np.integer)) \
            or hidden_dim <= 0):
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}")
        if not isinstance(dropout, (float, np.floating)) or not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be a float in [0.0, 1.0], got {dropout}")
        
        if hidden_dim is None:
            hidden_dim = embed_dim * 4
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
            
        self.dropout = Dropout(dropout)
        self.linear1 = Linear(embed_dim, hidden_dim, bias=True)
        self.gelu = GELU(approximate="sigmoid")
        self.linear2 = Linear(hidden_dim, embed_dim, bias=True)
        
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through MLP.

        COMPUTATION FLOW:
        x -> Linear -> GELU -> Linear -> output
        """
        hidden = self.linear1(x)
        hidden = self.gelu(hidden)
        output = self.linear2(hidden)
        return output
    

class TransformerBlock(Module):
    """
    Complete Transformer Block with self-attention, MLP, and residual connections.
    Implements Pre-LayerNorm architecture: x = x + Sublayer(LN(x)) for both attention and MLP.
    """
    def __init__(self, embed_dim: int, num_heads: int, 
                dropout: float = 0.0, mlp_ration: float | None=None,
                ff_dim: int | None=None) -> None:
        "Initialize a single transformer block with attention and MLP."
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = Dropout(dropout)
        
        #* Multi-Head Attention layer
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        #* layer normalization before attention and MLP (Pre-LN)
        self.ln1 = LayerNormalization(normalized_shape=embed_dim) #* before attention
        self.ln2 = LayerNormalization(normalized_shape=embed_dim) #* before MLP
        
        #* feed-forward network (MLP) with GELU activation
        if ff_dim is not None and mlp_ration is not None:
            raise ValueError(
                "Give the MLP width once."
                "Specify either ff_dim or mlp_ration, not both.")
            
        if ff_dim is not None:
            hidden_dim = ff_dim
        else:
            hidden_dim = int(embed_dim * (mlp_ration if mlp_ration is not None else 4))
        self.mlp = MLP(embed_dim, hidden_dim=hidden_dim)
        

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass through the transformer block.

        COMPUTATION FLOW:
        1. LayerNorm -> Multi-Head Attention -> Dropout -> Residual Add
        2. LayerNorm -> MLP -> Dropout -> Residual Add
        """
        #* Pre-LayerNorm before attention
        x_norm1 = self.ln1(x)
        attn_output = self.attention(x_norm1, mask=mask)
        attn_output = self.dropout(attn_output)
        x = x + attn_output  # Residual connection
        
        #* Pre-LayerNorm before MLP
        x_norm2 = self.ln2(x)
        mlp_output = self.mlp(x_norm2)
        mlp_output = self.dropout(mlp_output)
        x = x + mlp_output  # Residual connection
        
        return x


class GPT(Module):
    """
    GPT (Generative Pre-trained Transformer) model architecture.

    Combines token embeddings, positional embeddings, stacked transformer blocks,
    final layer normalization, and an un-embedding projection head for autoregressive generation.
    """
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int,
                num_layers: int, max_seq_len: int, dropout: float = 0.0, seed:int=42) -> None:
        "Initialize the GPT model with embeddings and stacked transformer blocks."
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)
        
        #* Token and positional embeddings
        self.token_embedding = EmbeddingLayer(vocab_size, embed_dim, max_seq_len)
        
        #* Stacked transformer blocks
        self.blocks = Sequential(*[TransformerBlock(embed_dim, num_heads, dropout=dropout)
                                for _ in range(num_layers)])
        
        #* Final layer normalization before output projection
        self.ln_f = LayerNormalization(normalized_shape=embed_dim)
        
        #* Un-embedding projection head to map back to vocabulary logits
        self.unembedding_head = Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass through the GPT model.

        COMPUTATION FLOW:
        1. Token Embedding -> Positional Embedding
        2. Stacked Transformer Blocks
        3. Final LayerNorm -> Un-embedding Projection to logits
        """
        _, T = inputs.shape
        if T > self.max_seq_len:
            raise ValueError(f"Input sequence length {T} exceeds max_seq_len {self.max_seq_len}")
        #* Token embeddings
        token_embeds = self.token_embedding(inputs)
        #* Pass through stacked transformer blocks
        if mask is None:
            mask = create_causal_mask(T)
        elif not isinstance(mask, Tensor):
            raise ValueError(f"mask must be a Tensor, got {type(mask)}")
        x = self.blocks(token_embeds, mask=mask)
        
        #* Project to vocabulary logits
        return self.unembedding_head(self.ln_f(x))
    
    def sample_next_token(self, logits: np.ndarray, temperature: float = 0.0, top_k: int | None = None) -> int:
        """
        Sample the next token from the model's output distribution.

        Args:
            logits: Output logits of shape (batch_size, vocab_size).
            temperature: Temperature for sampling.
            top_k: Optional number of top tokens to consider for sampling.
        Returns:
            A tensor of shape (batch_size,) containing the sampled next token IDs.
        """
        if not isinstance(temperature, (float, np.floating)) or temperature < 0:
            raise ValueError(f"temperature must be a positive float, got {temperature}")
        flat_logits = np.asarray(logits.data, dtype=np.float64).reshape(-1)
        if temperature <= 0.0:
            #* Greedy sampling: take the argmax
            return int(np.argmax(flat_logits))
        
        # Sample from the categorical distribution defined by the logits
        scaled_logits = flat_logits / temperature
        if top_k is not None:
            if not isinstance(top_k, (int, np.integer)) or top_k <= 0:
                raise ValueError(f"top_k must be a positive integer, got {top_k}")
            if top_k < len(scaled_logits):
                #* Keep only the top_k logits
                cutoff = np.sort(scaled_logits)[-min(top_k, len(scaled_logits))]
                scaled_logits[scaled_logits <= cutoff] = -np.inf  # Mask out lower logits
        
        centred_logits = scaled_logits - np.max(scaled_logits)  # For numerical stability
        probabilities = np.exp(centred_logits)
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        return int(self.rng.choice(self.vocab_size, p=probabilities))
    
    def generate(self, prompt_tokens: Tensor, max_new_tokens: int, temperature: float =0.0, top_k: int | None = None) -> Tensor:
        "Generate text autoregressively by repeatedly sampling next tokens from the model."
        if len(prompt_tokens.shape) != 2 or prompt_tokens.shape[0] != 1 or prompt_tokens.shape[-1] ==0:
            raise ValueError("generate expects one nonempty prompt with shape (1, sequence)")
        if not isinstance(max_new_tokens, (int, np.integer)) or max_new_tokens <0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if prompt_tokens.shape[-1] + max_new_tokens > self.max_seq_len:
            raise ValueError("Prompt plus generated tokens exceeds max_seq_len. You've run out of context window.")
        
        self.eval(mode=False)
        current_tokens = Tensor(prompt_tokens.data.copy())
        
        for _ in range(max_new_tokens):
            context = current_tokens[:, -self.max_seq_len:]  # Truncate to max_seq_len
            logits = self(context)
            last_logits = logits.data[:, -1, :]
            sampled_token = self.sample_next_token(last_logits, temperature, top_k)
            next_token = np.array([[sampled_token]])
            current_tokens = Tensor(np.concatenate([current_tokens.data, next_token], axis=1))
        return current_tokens