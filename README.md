# minitorch

A small, from-scratch deep learning framework built on NumPy: a reverse-mode
autograd `Tensor`, a PyTorch-style `Module`/`Parameter`/`Linear`/`Dropout`/
`LayerNorm` layer library, `SGD`/`Adam`/`AdamW` optimizers, multi-head
attention, a GPT-style decoder-only Transformer, two tokenizers
(character-level and BPE), a minimal `DataLoader`, and a `Trainer` with
gradient accumulation, clipping, a cosine LR schedule, and checkpointing.

Built as a learning project to understand every layer of a modern
Transformer training stack — embeddings → attention → optimizer internals
→ gradient flow — by implementing each piece by hand rather than importing
PyTorch.

## Package layout

```
minitorch/
├── tensor/          # Tensor class + autograd engine (the foundation everything else builds on)
├── nn/               # Module, Parameter, Linear, Dropout, Sequential, LayerNorm, BatchNorm, Flatten
├── optim/            # SGD, Adam, AdamW
├── activations/      # Sigmoid, ReLU, Tanh, GELU, Softmax
├── attention/         # Scaled dot-product attention, MultiHeadAttention
├── embeddings/        # Token embeddings, learned & sinusoidal positional encodings
├── losses/            # MSE, SoftMaxCrossEntropy, BCEWithLogits
├── transformer/        # MLP, TransformerBlock, GPT, causal masking, generation
├── dataloader/         # Dataset, TensorDataset, DataLoader
├── tokenizers/          # Tokenizer, CharTokenizer, BPETokenizer
└── training/            # Trainer, CosineScheduler, clip_grad_norm
```

Each folder has its own `README.md` with a detailed breakdown of that
module's API, usage examples, and — importantly — a list of bugs found
during review with suggested fixes. **Read the per-module READMEs before
editing that module**; several contain subtle correctness issues that are
easy to reintroduce if you're not aware of them.

## Dependency order

Roughly bottom-up, each layer builds on the one(s) above it:

```
tensor
  ├── nn (Module, Parameter, Linear, LayerNorm, Dropout, ...)
  │     ├── activations (Sigmoid, ReLU, Tanh, GELU, Softmax)
  │     ├── embeddings (Embedding, PositionalEncoding, EmbeddingLayer)
  │     ├── attention (MultiHeadAttention, built on nn.Linear)
  │     └── transformer (MLP, TransformerBlock, GPT — composes nn + activations + embeddings + attention)
  ├── losses (MSE, SoftMaxCrossEntropy, BCEWithLogits)
  ├── optim (SGD, Adam, AdamW — operates on nn.Parameter)
  ├── dataloader (Dataset, TensorDataset, DataLoader — wraps Tensor)
  └── training (Trainer, CosineScheduler — composes everything above)

tokenizers    # standalone, no Tensor dependency — pure Python text processing
```

## Quick start

```python
from minitorch.tokenizers import CharTokenizer
from minitorch.dataloader import DataLoader, TensorDataset
from minitorch.transformer import GPT
from minitorch.losses import SoftMaxCrossEntropy
from minitorch.optim import AdamW
from minitorch.training import Trainer, CosineScheduler
from minitorch.tensor import Tensor

tokenizer = CharTokenizer()
tokenizer.build_vocab([text])

gpt = GPT(vocab_size=tokenizer.vocab_size, embed_dim=64, num_heads=8,
          num_layers=6, max_seq_len=128, dropout=0.2)

optimizer = AdamW(gpt.parameters(), lr=1e-3, weight_decay=0.1)
scheduler = CosineScheduler(max_lr=1e-3, min_lr=4e-5, total_epochs=100)
trainer = Trainer(gpt, optimizer, SoftMaxCrossEntropy(), scheduler=scheduler, grad_clip_norm=1.0)

for epoch in range(100):
    train_loss = trainer.train_epoch(train_loader, accumulation_steps=2)
    val_loss, _ = trainer.evaluate(val_loader)
    print(f"epoch {epoch+1}: train={train_loss:.4f} val={val_loss:.4f}")
```

## Design notes that apply across the whole codebase

- **No in-place tensor mutation in the autograd graph.** Every op returns a
  new `Tensor` node; gradients are accumulated via `_add_grad`, never by
  mutating a tensor that's already part of a live graph. This is the
  correct, PyTorch-like safety property — preserve it in any new op.
- **`Module.parameters()` returns live references**, not copies — this is
  what makes it safe for an optimizer to mutate `param.data` in place and
  have the next `forward()` see the update. Verified correct during the
  loss-plateau investigation; keep any refactor of `_walk`/`modules()`
  honoring this.
- **Several custom ops (activations, losses) hand-write their backward
  pass** against raw `.data` rather than composing from `Tensor`'s own
  primitive ops. This is a deliberate performance/stability choice (e.g.
  numerically stable softmax/log-softmax), but means those backward
  formulas must be kept in sync by hand if the forward math changes —
  there's no autodiff safety net inside those files the way there is
  everywhere else.
- Several files carry a stray, likely autocomplete-induced import
  (`turtle`, `dis`) — worth a quick `ruff`/`flake8` pass across the whole
  package before publishing, in case there are more.