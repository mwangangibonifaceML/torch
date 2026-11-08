# `minitorch.attention`

Scaled dot-product attention and multi-head attention.

```python
compute_scaled_attention_scores(query, key, head_dim)   # (Q @ Kᵀ) / √head_dim
apply_mask(scores, mask)                                 # masked_fill(-1e9) where mask == 0
scaled_dot_product_attention(query, key, value, mask)    # softmax(scores) @ value
MultiHeadAttention(embed_dim, num_heads)
```

`MultiHeadAttention` projects `X` to `Q`/`K`/`V` with three `Linear` layers,
splits into `num_heads` of size `head_dim = embed_dim // num_heads`, runs
scaled dot-product attention per head, merges the heads back, and applies
an output projection — the standard Transformer attention block.

## Usage
```python
from minitorch.attention import MultiHeadAttention
from minitorch.transformer import create_causal_mask

attn = MultiHeadAttention(embed_dim=64, num_heads=8)
mask = create_causal_mask(seq_length=12)
out = attn(x, mask=mask)   # x: (batch, seq_len, 64)
```

## ⚠️ Bugs / risks

### 1. `apply_mask` raises if a mask ever fully blocks a row
```python
if np.any(np.sum(allowed, axis=-1) == 0):
    raise ValueError("Each query must have at least one allowed key")
```
This is a reasonable guard for a standard causal mask (every position can
always attend to itself), but if a padding mask is ever combined with the
causal mask such that a fully-padded row results, this will raise instead
of degrading gracefully (e.g. by leaving the row's output as zero). Not
currently triggered by the GPT training loop (only a causal mask is used),
but worth keeping in mind if padded-batch training is added later.


## Notes
- `scaled_dot_product_attention` returns `(output, attention_weights)` —
  the weights are discarded by `MultiHeadAttention.forward` today but are
  available if you want to add attention-map visualization/logging later.