# `minitorch.embeddings`

Token embeddings, positional encodings (learned and sinusoidal), and the
combined `EmbeddingLayer` used by `GPT`.

```python
Embedding(vocab_size, embed_dim)              # learnable lookup table
PositionalEncoding(max_seq_len, embed_dim)    # learnable position vectors
compute_sinusoidal_table(max_len, embed_dim)  # fixed sin/cos table (no params)
EmbeddingLayer(vocab_size, embed_dim, max_seq_len, pos_encoding_type="learned")
```

`EmbeddingLayer` is what `GPT` actually instantiates: it combines token
embeddings with either `"learned"`, `"sinusoidal"`, or no (`None`)
positional encoding, and transparently handles both `(seq_len,)` and
`(batch, seq_len)` inputs by squeezing/unsqueezing a batch dimension as
needed.

## Usage
```python
from minitorch.embeddings import EmbeddingLayer

emb = EmbeddingLayer(vocab_size=128, embed_dim=64, max_seq_len=12)
x = emb(token_ids)   # (batch, seq_len) int tensor -> (batch, seq_len, 64)
```

### Sinusoidal branch bypasses the autograd graph
```python
elif self.pos_encoding_type == "sinusoidal":
    ...
    output = token_embed + Tensor(pos_embeddings_batched)
```
`compute_sinusoidal_table` already returns a `Tensor`, but here its `.data`
is re-wrapped in a *new* plain `Tensor(..., requires_grad=False)` (the
default), which is actually correct behavior — sinusoidal encodings have no
learnable parameters, so they shouldn't require grad. Just noting this is
intentional, not a leak, so a future contributor doesn't "fix" it into
`requires_grad=True` and start accumulating pointless gradients on a
constant table.

### `start_pos` validity isn't checked against the *sinusoidal* table's actual length until inside the branch
Minor: the bounds check `start_pos + seq_len > self.max_seq_len` for the
sinusoidal path is correct, but it duplicates (slightly differently
worded) the equivalent check inside `PositionalEncoding.forward` for the
learned path. If you ever change `max_seq_len` handling, keep both error
messages in sync — not a functional bug today, just a maintenance note.

