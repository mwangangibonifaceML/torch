from __future__ import annotations

import random
import numpy as np

from abc import ABC, abstractmethod
from minitorch.tensor import Tensor


class Dataset(ABC):
    "Abstract base class for all datasets."
    
    @abstractmethod
    def __len__(self):
        "Return the total number of samples in the dataset."
        pass
    
    def __getitem__(self, key):
        "Return the sample at the given index."
        pass
    

class TensorDataset(Dataset):
    def __init__(self, tensors: list[Tensor]) -> None:
        super().__init__()
        if len(tensors) == 0:
            raise ValueError("TensorDataset needs atleast one Tensor")
        
        #* check if the tensors have the number of samples
        #* in the zero dimension
        n_samples = len(tensors[0].data)
        for tensor_idx, tensor in enumerate(tensors):
            if len(tensor.data) != n_samples:
                raise ValueError(
                    f"Tensor mismatch in TensorDataset\n"
                    f"Tensor 0 has {n_samples} samples but Tensor{tensor_idx} has {len(tensor.data)} samples\n"
                    f"All tensors must have the same size in their first dimension (the sample dimension)\n"
                    f"Check your data: features.shape[0] should equal labels.shape[0]"
                )
        self.tensors = tensors
        
    def __len__(self) -> int:
        "Return the number of samples (size of the first dimension)."
        return len(self.tensors[0].data)
    
    def __getitem__(self, key: int):
        "Return a tuple of tensor slices at a given key."
        if key <0:
            key += len(self)
            
        if key < 0 or key >= len(self):
            raise ValueError(f"Index {key} our of range for dataset of size {len(self)}")
        return tuple(tensor[key] for tensor in self.tensors)
    
    
class DataLoader(Dataset):
    "Dataloader with batching and shuffling support."
    
    def __init__(self, dataset: TensorDataset, batch_size: int, shuffle: bool) -> None:
        "Create a DataLoader for batched iteration."
        super().__init__()
        if batch_size < 1 or not isinstance(batch_size, int):
            raise ValueError("batch size must be greater than zero.")
        if not isinstance(shuffle, bool):
            raise ValueError(f"shuffle must be a boolean, you provided {type(shuffle)}")
    
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __len__(self) -> int:
        "Return the number of batches in a dataloader per epoch."
        return (len(self.dataset) + (self.batch_size -1)) // self.batch_size
    
    def __iter__(self):
        "Iterator through the batches"
        
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
            
        #* yield batched
        for index in range(0, len(indices), self.batch_size):
            batch_indices = indices[index:index + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            
            #* convert a list of tuple to a tuple of Tensors
            batched_tensor = []
            num_tensors = len(batch[0])
            
            if len(batch) ==0:
                return 
            
            #* group tensors by position
            for tensor_index in range(num_tensors):
                tensor_list = [sample[tensor_index].data for sample in batch]
                
                #* stack into batch tensor
                batched_data = np.stack(tensor_list, axis=0)
                batched_tensor.append(Tensor(batched_data))

            yield batched_tensor
        
        
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X = Tensor(rng.standard_normal((101,10)))
    y = Tensor(np.arange(101))
    
    tensor_dataset = TensorDataset((X,y))
    loader = DataLoader(tensor_dataset, 10, True)
    
    print(f"Dataset: {len(tensor_dataset)} samples")
    print("Batch size: 10")
    print(f"Number of batches: {len(loader)}")

    print("\nBatches:")
    for i, (batch_x, batch_y) in enumerate(loader):
        print(f"  Batch {i+1}: {batch_x.shape[0]} samples, shape {batch_x.shape}")

        if i+1 == 11:
            print(batch_x, batch_y)
    print("\nDataLoader organizes data for efficient training!")