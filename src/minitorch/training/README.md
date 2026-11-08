# `minitorch.training`

The training orchestration layer: gradient clipping, a cosine LR schedule,
and the `Trainer` class that ties model + optimizer + loss + scheduler
together into `train_epoch()`/`evaluate()`, plus checkpointing.

```python
clip_grad_norm(parameters, max_norm=1.0) -> float   # global-norm clipping, returns pre-clip norm
CosineScheduler(min_lr, max_lr, total_epochs)
  .get_lr(epoch) -> float
Trainer(model, optimizer, loss_fn, scheduler=None, grad_clip_norm=None)
  .train_epoch(dataloader, accumulation_steps=2) -> float
  .evaluate(dataloader) -> (avg_loss, accuracy)
  .save_checkpoint(path) / .load_checkpoint(path)
```

## `clip_grad_norm`
Computes the L2 norm across *all* parameter gradients jointly (not
per-parameter), and if it exceeds `max_norm`, scales every gradient down by
the same factor `max_norm / total_norm` — standard global-norm clipping,
matching `torch.nn.utils.clip_grad_norm_`.

## `CosineScheduler`
```
lr(epoch) = min_lr + (max_lr - min_lr) · (1 + cos(π·epoch/total_epochs)) / 2
```
Correctly decays from `max_lr` at `epoch=0` to `min_lr` at
`epoch=total_epochs`, and clamps to `min_lr` for any epoch beyond that.

## `Trainer`
`train_epoch` iterates the dataloader, calls `_process_batch` (forward +
backward) per batch, and calls `_optimizer_update` every
`accumulation_steps` batches (plus once more for a trailing partial group).
`_optimizer_update` divides accumulated gradients by the number of batches
accumulated, applies `clip_grad_norm` if configured, then `optimizer.step()`
+ `optimizer.zero_grad()`.

## Usage
```python
from minitorch.training import Trainer, CosineScheduler
from minitorch.optim import AdamW
from minitorch.losses import SoftMaxCrossEntropy

optimizer = AdamW(gpt.parameters(), weight_decay=0.1)
scheduler = CosineScheduler(max_lr=1e-3, min_lr=4e-5, total_epochs=100)
trainer = Trainer(gpt, optimizer, SoftMaxCrossEntropy(), scheduler=scheduler, grad_clip_norm=1.0)

for epoch in range(100):
    train_loss = trainer.train_epoch(train_loader, accumulation_steps=2)
    val_loss, _ = trainer.evaluate(val_loader)
```