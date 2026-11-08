from __future__ import annotations

import math
import numpy as np

from minitorch.tensor import Tensor
from minitorch.nn import Parameter, Module

rng = np.random.default_rng(seed=42)

class Embedding(Module):
    """
    Learnable embedding layer that maps tokens indices to dense vectors.
    
    It convert discrete tokens into continous representations that neural
    networks can process
    """
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """
        Initialize embedding layer with xavier-uniform weights
        
        Args:
            vocab_size: size of vocabulary. These are the number of unique tokens
            embed_dim: Dimension of the embedding vector
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        #* initialize the embedding weight (the look up table)
        # limit = math.sqrt(6.0 / (vocab_size + embed_dim))
        self.weight = Parameter(rng.normal(0, 0.02, size=(vocab_size, embed_dim)))
        
    def forward(self, indices: Tensor) -> Tensor:
        """
        Forward pass: lookup embeddings for given indices.

        Args:
            indices: Token indices of shape (batch_size, seq_len) or (seq_len,)

        Returns:
            Embedded vectors of shape (*indices.shape, embed_dim)
        """
        ids = indices.data
        if not np.all(np.isfinite(ids)) or np.any(ids != np.floor(ids)):
            raise ValueError("Embedding token IDs must be finite integer.")
        if ids.size and (np.min(ids) < 0 or np.max(ids) >= self.vocab_size):
            raise ValueError(
                f"Embedding index out of range for vocabulary size {self.vocab_size}\n"
                f" Found indices: min={np.min(ids)}, max={np.max(ids)} (valid range: 0 to {self.vocab_size - 1})\n"
                f" Token IDs must be within the vocabulary. IDs >= vocab_size reference non-existent tokens\n"
                f" Check your tokenizer output, or increase vocab_size to at least {np.max(ids) + 1}"
            )
        return self.weight[ids.astype(int)]
    
    def __repr__(self):
        return f"Embedding(vocab_size={self.vocab_size}, embed_dim={self.embed_dim})"
    
    
class PositionalEncoding(Module):
    """
    Learnable positional encoding layer.
    
    Adds trainable position specific vectors to token embeddings allowing the model
    to learn positional patterns specific to the task.
    """
    def __init__(self, max_seq_len: int, embed_dim:int) -> None:
        """
        Initialize learnable positional encoding.
        
        Args:
            max_seq_len: Maximum sequence length to support
            embed_dim: Embedding dimension (must match token embeddings)
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim
        
        # limit = math.sqrt(6.0 / embed_dim)
        self.position_embeddings = Parameter(
            rng.normal(0,0.02, size=(max_seq_len, embed_dim))
        )

    def forward(self, inputs: Tensor, start_pos: int=0) -> Tensor:
        "Add positional encodings to the input embedings."
        if len(inputs.shape) == 2:
            raise ValueError(
                f"Expected 3D input (batch, seq, embed), got 2D: {inputs.shape}\n"
                f" Missing batch dimension\n"
                f" PositionalEncoding expects batched embeddings, not single sequences\n"
                f" Add batch dim: x.reshape(1, {inputs.shape[0]}, {inputs.shape[1]})"
            )
        elif len(inputs.shape) != 3:
            raise ValueError(
                f"Expected 3D input (batch, seq, embed), got {len(inputs.shape)}D: {inputs.shape}\n"
                f" Input must have exactly 3 dimensions\n"
                f" PositionalEncoding expects shape (batch_size, sequence_length, embedding_dim)"
            )
        
        batch_size, seq_len, embed_dim = inputs.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Embedding dimension mismatch: input has {embed_dim}, expected {self.embed_dim}\n"
                f" PositionalEncoding was created with embed_dim={self.embed_dim}, but input has embed_dim={embed_dim}\n"
                f" Token embeddings and positional encodings must have the same dimension to be added together\n"
                f" Ensure your Embedding layer uses embed_dim={self.embed_dim}, or create PositionalEncoding(embed_dim={embed_dim}, ...)"
            )
        if start_pos + seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence runs past the maximum: positions {start_pos}..{start_pos + seq_len - 1} with max_seq_len={self.max_seq_len}\n"
                f" Input has {seq_len} positions starting at {start_pos}, but only {self.max_seq_len} are available\n"
                f" Learned positional encodings have a fixed maximum length set at initialization\n"
                f" Either truncate input to {self.max_seq_len} tokens, or create a new PositionalEncoding(max_seq_len={start_pos + seq_len}, ...)"
            )
        
        pos_embeddings = self.position_embeddings[start_pos: start_pos + seq_len]
        pos_embeddings_batched = pos_embeddings.reshape(1, seq_len, embed_dim)
        return inputs + pos_embeddings_batched
    
    def __repr__(self):
        return f"PositionalEncoding(max_seq_len={self.max_seq_len}, embed_dim={self.embed_dim})"
    
    
def compute_sinusoidal_table(max_len: int, embed_dim: int) -> Tensor:
    """
    Compute the raw sinusoidal positional encoding table as a numpy array then wrap them in a Tensor.

    This helper builds the (max_len, embed_dim) table of sin/cos values
    using the formula from "Attention Is All You Need":
        PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))
        
    These fixed encodings use sine and cosine functions to create unique
    positional patterns that don't require training and can extrapolate
    to longer sequences than seen during training.
    """
    positions = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(
        np.arange(0, embed_dim, 2, dtype=np.float32) *
        -(math.log(10000.0) / embed_dim)
    )
    
    # Initialize the positional encoding matrix
    pe = np.zeros((max_len, embed_dim), dtype=np.float32)

    # Apply sine to even indices (0, 2, 4, ...)
    pe[:, 0::2] = np.sin(positions * div_term)

    # Apply cosine to odd indices (1, 3, 5, ...)
    if embed_dim % 2 == 1:
        # Handle odd embed_dim by only filling available positions
        pe[:, 1::2] = np.cos(positions * div_term[:-1])
    else:
        pe[:, 1::2] = np.cos(positions * div_term)
        
    return Tensor(pe)
    

class EmbeddingLayer(Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_seq_len: int=512, pos_encoding_type: str= "learned") -> None:
        super().__init__()
        for field, value in (("vocab_size", vocab_size), ("embed_dim", embed_dim),
                                    ("max_seq_len", max_seq_len)):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{field} must be a positive integer, got {value!r}")
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pos_encoding_type = pos_encoding_type
        # self.scale_embeddings = scale_embeddings
        
        #* token embedding layer
        self.token_embedding = Embedding(vocab_size, embed_dim)
        
        #* positional enconding
        if pos_encoding_type == "learned":
            self.pos_encoding = PositionalEncoding(max_seq_len, embed_dim)
        elif pos_encoding_type == "sinusoidal":
            self.pos_encoding = compute_sinusoidal_table(max_seq_len, embed_dim)
        elif pos_encoding_type is None:
            self.pos_encoding = None
        else:
            raise ValueError(
                f"Unknown positional encoding type: '{pos_encoding_type}'\n"
                f" pos_encoding must be 'learned', 'sinusoidal', or None\n"
                f" 'learned' = trainable position embeddings (task-specific but fixed max length)\n"
                f" 'sinusoidal' = mathematical sin/cos patterns (no parameters, can extrapolate)\n"
                f" None = no positional encoding (order-agnostic model)\n"
                f" Use: EmbeddingLayer(..., pos_encoding='learned') or pos_encoding='sinusoidal'"
            )
            
    def forward(self, tokens: Tensor, start_pos: int=0) -> Tensor:
        if not isinstance(start_pos, (int, np.integer)) or start_pos < 0:
                raise ValueError("start_pos must be a nonnegative integer")
        if len(tokens.shape) == 1:
            tokens = tokens.reshape(1, -1)
            squezed_batch = True
        else:
            squezed_batch = False
            
        #* get token embeddngs
        token_embed = self.token_embedding(tokens)
        
        #* get positional encondig
        if self.pos_encoding_type == "learned":
            output = self.pos_encoding(token_embed, start_pos=start_pos)
        elif self.pos_encoding_type == "sinusoidal":
            batch_size, seq_len, embed_dim = token_embed.shape
            if start_pos + seq_len > self.max_seq_len:
                raise ValueError("Sequence runs past the sinusoidal position table")
            pos_embeddings = self.pos_encoding[start_pos: start_pos+seq_len]
            pos_embeddings_batched = pos_embeddings.data[np.newaxis,:,:]
            output = token_embed + Tensor(pos_embeddings_batched)
        else:
            output = token_embed
            
        if squezed_batch:
            output = output[0]
        return output
            
    def __repr__(self):
        return (f"EmbeddingLayer(vocab_size={self.vocab_size}, "
                f"embed_dim={self.embed_dim}, "
                f"pos_encoding='{self.pos_encoding_type}')")
    