from __future__ import annotations

import numpy as np

from minitorch.attention import MultiHeadAttention
from minitorch.activations import GELU
from minitorch.dataloader import DataLoader, TensorDataset
from minitorch.embeddings import EmbeddingLayer
from minitorch.tensor import Tensor
from minitorch.tokenizers import CharTokenizer
from minitorch.training import Trainer, CosineScheduler
from minitorch.losses import SoftMaxCrossEntropy
from minitorch.nn import Linear, Module, LayerNormalization, Dropout, Parameter, Sequential
from minitorch.optim import SGD, AdamW


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
        
    def forward(self, inputs: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
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
                scaled_logits[scaled_logits < cutoff] = -np.inf  # Mask out lower logits
        
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
    
    
    
if __name__ == "__main__":
    x = Tensor(np.ones((1, 3, 3)))
    mask = Tensor(np.tril(np.ones((3,3)))[np.newaxis, :, :])
    print(mask)
    out = np.where(mask.data, x.data, -np.inf)
    print(np.exp(out) / np.exp(out).sum())
    
    
    # print(mask)
    # out = Tensor.masked_fill(x, mask, filler=-5.0)
    # print(out.data[0,0,1], out.data[0,1,0])
    # assert out.data[0, 0, 1] == -5.0 and out.data[0, 1, 0] == 1.0
    # text_path = "C:\\Users\\User\\Downloads\\Swahili_Corpus_combined.txt"
    # with open(text_path, 'r', encoding='utf-8') as f:
    #     text = f.read()
        
    # text = text[:1000]
    # #* split into training and val sets
    # n = 0.95
    # text_len = len(text)
    
    # chars = sorted(set(text))
    # max_epochs = 100
    # eval_iters = 10
    # context_length, batch_size, embed_dim = 12,8,64
    # # #* tokenize the text
    # tokenizer = CharTokenizer() #* character level tokenization
    # tokenizer.build_vocab([text])
    # vocab_size = tokenizer.vocab_size
    
    
    # # #* get the train and val loaders
    # data = Tensor(tokenizer.encode(text))
    # train_data = data[:int(n*data.shape[0])]
    # val_data = data[int(n*data.shape[0]):]
    
    # def prepare_data(dataset: Tensor, context_length: int, stride:int =1) -> tuple[Tensor, Tensor]:
    #     start = range(0, dataset.shape[0] - context_length, stride)
    #     x = np.stack([dataset[i:i+context_length].data for i in start])
    #     y = np.stack([dataset[i+1:i+1+context_length].data for i in start])
    #     return Tensor(x), Tensor(y)

    
    # train_features, train_targets = prepare_data(train_data, context_length)
    # val_features, val_targets = prepare_data(val_data, context_length)
    
    # train_loader = DataLoader(TensorDataset([train_features, train_targets]), batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(TensorDataset([val_features, val_targets]), batch_size=batch_size, shuffle=False)
    
    # gpt = GPT(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=8,
    #             num_layers=6, max_seq_len=context_length, dropout=0.2)
    
    # optimizer = AdamW(gpt.parameters(), weight_decay=0.1)
    # scheduler = CosineScheduler(max_lr=1e-3, min_lr=4e-5, total_epochs=max_epochs)
    # loss_fn = SoftMaxCrossEntropy()
    # trainer = Trainer(gpt, optimizer, loss_fn, scheduler=scheduler, grad_clip_norm=1.0)
    
    # # #* ###########################################################
    # #* training loop
    # # print("Epoch          | Training Loss         | Validation Loss | ")
    # # print("-" * 60)

    # # for i in range(max_epochs):        
    # #     training_loss = trainer.train_epoch(train_loader, accumulation_steps=2)
    # #     validation_loss,_ = trainer.evaluate(val_loader)
    
    # #     if (i+1) % eval_iters == 0 or (i+1) == 1:
    # #         print(f'{i+1}             |   {training_loss:.4f}             | {validation_loss:.4f}')
            
            
            
        
    # # Obtain one batch
    # features, targets = next(iter(train_loader))

    # # Disable dropout
    # gpt.eval(mode=False)

    # # Use a fixed learning rate
    # optimizer.lr = 1e-3

    # for step in range(500):

    #     optimizer.zero_grad()

    #     logits = gpt(features)

    #     logits = logits.reshape(
    #         -1,
    #         vocab_size
    #     )

    #     loss = loss_fn(
    #         logits,
    #         targets.reshape(-1)
    #     )

    #     loss.backward()

    #     optimizer.step()

    #     if step % 25 == 0:
    #         print(
    #             f"Step {step:4d} | "
    #             f"Loss {float(loss.data):.6f}"
    #         )
