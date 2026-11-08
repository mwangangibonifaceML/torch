# `minitorch.transformer`

The GPT model itself: `MLP`, `TransformerBlock` (Pre-LayerNorm), and `GPT`,
plus `create_causal_mask`.

```python
create_causal_mask(seq_length) -> Tensor   # (1, T, T) lower-triangular mask
MLP(embed_dim, hidden_dim=None, dropout=0.0)              # Linear -> GELU -> Linear, 4x expansion by default
TransformerBlock(embed_dim, num_heads, dropout=0.0, mlp_ration=None, ff_dim=None)
GPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.0, seed=42)
```

`TransformerBlock` implements the standard Pre-LN residual pattern:
```
x = x + Attention(LN1(x))
x = x + MLP(LN2(x))
```
`GPT.forward` runs token+positional embedding → stacked blocks → final
`LayerNormalization` → `unembedding_head` (untied from the input embedding
— there's no weight sharing between `token_embedding.weight` and
`unembedding_head.weights`). `GPT.generate` autoregressively samples with
optional temperature and top-k via `sample_next_token`.

## Usage
```python
from minitorch.transformer import GPT

gpt = GPT(vocab_size=64, embed_dim=64, num_heads=8, num_layers=6, max_seq_len=12, dropout=0.2)
logits = gpt(input_ids)                          # (batch, seq_len, vocab_size)
generated = gpt.generate(prompt_ids, max_new_tokens=50, temperature=0.8, top_k=20)
```

### `dropout` is only applied *after* attention/MLP output, never as attention-weight dropout
This is a legitimate, common simplification (plenty of GPT implementations
skip attention-dropout), just noting it so it's a documented design choice
rather than something to "discover" later: there's one `Dropout(dropout)`
per block applied to the residual branch outputs, not inside
`MultiHeadAttention` on the softmax weights themselves.


