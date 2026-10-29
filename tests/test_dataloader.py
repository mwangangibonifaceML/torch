import numpy as np
import pytest

from minitorch.dataloader import Dataset, TensorDataset, DataLoader
from minitorch.tensor import Tensor

def test_unit_dataset():
    """🧪 Test Dataset abstract base class."""
    print("🧪 Unit Test: Dataset Abstract Base Class...")

    # Test that Dataset is properly abstract
    try:
        dataset = Dataset()
        assert False, "Should not be able to instantiate abstract Dataset"
    except TypeError:
        print("✅ Dataset is properly abstract")

    # Test concrete implementation
    class TestDataset(Dataset):
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return f"item_{idx}"

    dataset = TestDataset(10)
    assert len(dataset) == 10
    assert dataset[0] == "item_0"
    assert dataset[9] == "item_9"



def test_unit_tensordataset():
    # Test basic functionality
    features = Tensor([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
    labels = Tensor([0, 1, 0])                   # 3 labels

    dataset = TensorDataset((features, labels))

    # Test length
    assert len(dataset) == 3, f"Expected length 3, got {len(dataset)}"

    # Test indexing
    sample = dataset[0]
    assert len(sample) == 2, "Should return tuple with 2 tensors"
    assert np.array_equal(sample[0].data, [1, 2]), f"Wrong features: {sample[0].data}"
    assert sample[1].data == 0, f"Wrong label: {sample[1].data}"

    sample = dataset[1]
    assert np.array_equal(sample[1].data, 1), f"Wrong label at index 1: {sample[1].data}"

    # Test error handling
    with pytest.raises(ValueError):
        dataset[10]


    # Test mismatched tensor sizes
    bad_features = Tensor([[1, 2], [3, 4]])  # Only 2 samples
    bad_labels = Tensor([0, 1, 0])           # 3 labels - mismatch!
    with pytest.raises(ValueError):
        TensorDataset((bad_features, bad_labels))