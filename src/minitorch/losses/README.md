# `minitorch.losses`

Loss functions: `MSE`, `SoftMaxCrossEntropy` (used for the character-level
GPT), `BCEWithLogits`. Each is a callable `Loss` subclass that returns a
scalar `Tensor` whose `_backward` computes the gradient w.r.t. its inputs
analytically (bypassing `Tensor`'s own op-level autograd, similar to
`minitorch.activations`).

```python
Loss(reduction="mean")   # "mean" | "sum" | "none"
  .forward(predictions, targets) -> Tensor
```

`SoftMaxCrossEntropy` is what `train_transformer.ipynb` / `train.py` use
for language-model training: it takes raw logits `(..., vocab_size)` and
integer class indices, reshapes to `(N, vocab_size)`/`(N,)`, computes a
numerically-stable log-softmax, gathers the target log-probability per row,
and averages the negative log-likelihood — the standard PyTorch
`CrossEntropyLoss(logits, targets)` contract (no one-hot targets needed).

## Usage
```python
from minitorch.losses import SoftMaxCrossEntropy

loss_fn = SoftMaxCrossEntropy()
loss = loss_fn(logits.reshape(-1, vocab_size), targets.reshape(-1))
loss.backward()
```




