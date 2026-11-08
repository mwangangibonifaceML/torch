# `minitorch.dataloader`

Minimal `Dataset`/`DataLoader` pair, modeled on PyTorch's.

```python
Dataset               # ABC: __len__, __getitem__
TensorDataset(tensors: list[Tensor])      # zips N parallel tensors by index
DataLoader(dataset, batch_size, shuffle)  # batches + shuffles each epoch
```

`DataLoader.__iter__` shuffles indices (if `shuffle=True`), slices them into
`batch_size` chunks, pulls each sample via `dataset[idx]`, then re-groups
"list of tuples" into "tuple of batched Tensors" by stacking along axis 0.

## Usage
```python
from minitorch.dataloader import DataLoader, TensorDataset

loader = DataLoader(TensorDataset([features, targets]), batch_size=32, shuffle=True)
for batch_features, batch_targets in loader:
    ...
```
