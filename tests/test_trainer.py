
import os
import pickle
import numpy as np
from pathlib import Path

from minitorch.activations import ReLU
from minitorch.optim import Adam, Optimizer, SGD, AdamW
from minitorch.tensor import Tensor
from minitorch.nn import Parameter, Module, Linear
from minitorch.dataloader import DataLoader, TensorDataset
from minitorch.losses import SoftMaxCrossEntropy, BCEWithLogits, MSE, Loss
from minitorch.training import Trainer, clip_grad_norm, CosineScheduler

def test_unit_cosine_schedule():
    """🧪 Test CosineScheduler implementation."""
    print("🧪 Unit Test: CosineScheduler...")

    # Test basic schedule
    schedule = CosineScheduler(max_lr=0.1, min_lr=0.01, total_epochs=100)

    # Test start, middle, and end
    lr_start = schedule.get_lr(0)
    lr_middle = schedule.get_lr(50)
    lr_end = schedule.get_lr(100)

    print(f"Learning rate at epoch 0: {lr_start:.4f}")
    print(f"Learning rate at epoch 50: {lr_middle:.4f}")
    print(f"Learning rate at epoch 100: {lr_end:.4f}")

    # Validate behavior
    assert abs(lr_start - 0.1) < 1e-6, f"Expected 0.1 at start, got {lr_start}"
    assert abs(lr_end - 0.01) < 1e-6, f"Expected 0.01 at end, got {lr_end}"
    assert 0.01 < lr_middle < 0.1, f"Middle LR should be between min and max, got {lr_middle}"

    # Test monotonic decrease in first half
    lr_quarter = schedule.get_lr(25)
    assert lr_quarter > lr_middle, "LR should decrease monotonically in first half"
    print("✅ CosineScheduler works correctly!")
    
def test_unit_clip_grad_norm():
    """🧪 Test clip_grad_norm implementation."""
    print("🧪 Unit Test: Gradient Clipping...")

    # Test case 1: Large gradients that need clipping
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param1.grad = np.array([3.0, 4.0])  # norm = 5.0

    param2 = Tensor([3.0, 4.0], requires_grad=True)
    param2.grad = np.array([6.0, 8.0])  # norm = 10.0

    params = [param1, param2]
    # Total norm = sqrt(5² + 10²) = sqrt(125) ≈ 11.18

    original_norm = clip_grad_norm(params, max_norm=1.0)

    # Check original norm was large
    assert original_norm > 1.0, f"Original norm should be > 1.0, got {original_norm}"

    # Check gradients were clipped
    new_norm = 0.0
    for param in params:
        if isinstance(param.grad, np.ndarray):
            grad_data = param.grad
        else:
            # grad set by hand as a Tensor; take its array
            grad_data = param.grad.data
        new_norm += np.sum(grad_data ** 2)
    new_norm = np.sqrt(new_norm)

    print(f"Original norm: {original_norm:.2f}")
    print(f"Clipped norm: {new_norm:.2f}")

    assert abs(new_norm - 1.0) < 1e-6, f"Clipped norm should be 1.0, got {new_norm}"

    # Test case 2: Small gradients that don't need clipping
    small_param = Tensor([1.0, 2.0], requires_grad=True)
    small_param.grad = np.array([0.1, 0.2])
    small_params = [small_param]
    original_small = clip_grad_norm(small_params, max_norm=1.0)

    assert original_small < 1.0, "Small gradients shouldn't be clipped"

    print("✅ Gradient clipping works correctly!")   
    
def test_unit_trainer_init():
    """🧪 Test Trainer.__init__ implementation."""
    print("🧪 Unit Test: Trainer.__init__...")

    class DummyModel:
        def __init__(self):
            self.training = True
        def forward(self, x):
            return x
        def parameters(self):
            return []

    model = DummyModel()
    optimizer = SGD([], lr=0.01)
    loss_fn = MSE()
    scheduler = CosineScheduler(max_lr=0.1, min_lr=0.01, total_epochs=10)

    trainer = Trainer(model, optimizer, loss_fn, scheduler, grad_clip_norm=1.0)

    # Verify components stored
    assert trainer.model is model, "Model not stored"
    assert trainer.optimizer is optimizer, "Optimizer not stored"
    assert trainer.loss_fn is loss_fn, "Loss function not stored"
    assert trainer.scheduler is scheduler, "Scheduler not stored"
    assert trainer.grad_clip_norm == 1.0, "Grad clip norm not stored"

    # Verify state initialization
    assert trainer.epoch == 0, f"Expected epoch=0, got {trainer.epoch}"
    assert trainer.step == 0, f"Expected step=0, got {trainer.step}"
    assert trainer.training_mode is True, "Expected training_mode=True"

    # Verify history
    assert 'train_loss' in trainer.history, "Missing train_loss in history"
    assert 'eval_loss' in trainer.history, "Missing eval_loss in history"
    assert 'learning_rates' in trainer.history, "Missing learning_rates in history"
    assert len(trainer.history['train_loss']) == 0, "train_loss should be empty"

    # Test without optional args
    trainer2 = Trainer(model, optimizer, loss_fn)
    assert trainer2.scheduler is None, "Scheduler should default to None"
    assert trainer2.grad_clip_norm is None, "Grad clip should default to None"

    print("✅ Trainer.__init__ works correctly!")
    

def test_unit_trainer_process_batch():
    """🧪 Test Trainer._process_batch implementation."""
    print("🧪 Unit Test: Trainer._process_batch...")

    class SimpleModel(Module):
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = MSE()
    trainer = Trainer(model, optimizer, loss_fn)

    inputs = Tensor([[1.0, 0.5]])
    targets = Tensor([[2.0]])

    scaled_loss = trainer._process_batch(inputs, targets)

    # Should return a float
    assert isinstance(scaled_loss, float), f"Expected float, got {type(scaled_loss)}"
    # Loss should be non-negative
    assert scaled_loss >= 0, f"Expected non-negative loss, got {scaled_loss}"

    print("✅ Trainer._process_batch works correctly!")
    
    
def test_unit_trainer_optimizer_update():
    """🧪 Test Trainer._optimizer_update implementation."""
    print("🧪 Unit Test: Trainer._optimizer_update...")

    class SimpleModel(Module):
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = MSE()
    trainer = Trainer(model, optimizer, loss_fn)

    # Do a forward-backward to create gradients
    inputs = Tensor([[1.0, 0.5]])
    targets = Tensor([[2.0]])
    trainer._process_batch(inputs, targets)

    # Record params before update
    params_before = [p.data.copy() for p in model.parameters()]

    # Perform update
    trainer._optimizer_update(inputs.shape[0])

    # Parameters should have changed
    params_after = [p.data for p in model.parameters()]
    changed = any(not np.allclose(b, a) for b, a in zip(params_before, params_after))
    assert changed, "Parameters should change after optimizer update"

    print("✅ Trainer._optimizer_update works correctly!")
    
    
def test_unit_trainer_train_epoch():
    """🧪 Test Trainer.train_epoch implementation."""
    print("🧪 Unit Test: Trainer.train_epoch...")

    class SimpleModel(Module):
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = SimpleModel()
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = MSE()

    trainer = Trainer(model, optimizer, loss_fn)

    dataloader = [
        (Tensor([[1.0, 0.5]]), Tensor([[2.0]])),
        (Tensor([[0.5, 1.0]]), Tensor([[1.5]]))
    ]

    # Train one epoch
    loss = trainer.train_epoch(dataloader)

    # Verify return type
    assert isinstance(loss, (float, np.floating)), f"Expected float loss, got {type(loss)}"

    # Verify epoch incremented
    assert trainer.epoch == 1, f"Expected epoch=1, got {trainer.epoch}"

    # Verify history recorded
    assert len(trainer.history['train_loss']) == 1, "Should have 1 loss recorded"

    # Verify model was in training mode
    assert trainer.training_mode is True, "Should be in training mode"

    # Train another epoch - loss should still be a valid number
    loss2 = trainer.train_epoch(dataloader)
    assert isinstance(loss2, (float, np.floating)), f"Second epoch loss should be float"
    assert trainer.epoch == 2, f"Expected epoch=2, got {trainer.epoch}"
    assert len(trainer.history['train_loss']) == 2, "Should have 2 losses recorded"

    # A one-sample tail must have the same weight as a one-sample batch.
    tail_model = Linear(1, 1, bias=True)
    tail_model.weights.data[:] = 1.0
    tail_model.bias.data[:] = 0.0
    tail_trainer = Trainer(tail_model, SGD(tail_model.parameters(), lr=0.1), MSE())
    tail_loss = tail_trainer.train_epoch(
        [(Tensor([[1.0]]), Tensor([[0.0]]))], accumulation_steps=4
    )
    assert np.allclose(tail_model.weights.data, [[0.8]]), "Normalize by the actual tail size"
    assert np.isclose(tail_loss, 1.0), "Accumulation must not scale the reported loss"

    # Test with scheduler
    model2 = SimpleModel()
    optimizer2 = SGD(model2.parameters(), lr=0.1)
    scheduler = CosineScheduler(max_lr=0.1, min_lr=0.01, total_epochs=10)
    trainer2 = Trainer(model2, optimizer2, loss_fn, scheduler=scheduler)

    trainer2.train_epoch(dataloader)
    assert len(trainer2.history['learning_rates']) == 1, "Should record LR with scheduler"

    # Test with gradient clipping
    model3 = SimpleModel()
    optimizer3 = SGD(model3.parameters(), lr=0.01)
    trainer3 = Trainer(model3, optimizer3, loss_fn, grad_clip_norm=0.5)

    loss3 = trainer3.train_epoch(dataloader)
    assert isinstance(loss3, (float, np.floating)), "Training with grad clip should work"

    print(f"  Epoch 1 loss: {loss:.4f}")
    print(f"  Epoch 2 loss: {loss2:.4f}")
    print("✅ Trainer.train_epoch works correctly!")
    
def test_unit_trainer_evaluate():
    """🧪 Test Trainer.evaluate implementation."""
    print("🧪 Unit Test: Trainer.evaluate...")

    # --- Regression case: MSE has no classification accuracy ---
    class RegressionModel(Module):
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    reg_model = RegressionModel()
    reg_trainer = Trainer(reg_model, SGD(reg_model.parameters(), lr=0.01), MSE())

    reg_dataloader = [
        (Tensor([[1.0, 0.5]]), Tensor([[2.0]])),
        (Tensor([[0.5, 1.0]]), Tensor([[1.5]]))
    ]

    eval_loss, accuracy = reg_trainer.evaluate(reg_dataloader)

    assert isinstance(eval_loss, (float, np.floating)), f"Expected float eval_loss, got {type(eval_loss)}"
    assert isinstance(accuracy, (float, np.floating)), f"Expected float accuracy, got {type(accuracy)}"
    # Regression: accuracy must be 0.0 (no classification branch entered)
    assert accuracy == 0.0, (
        f"Regression evaluate() returned accuracy={accuracy:.4f} instead of 0.0. "
        "MSE outputs must not enter the argmax classification branch."
    )

    assert reg_trainer.training_mode is False, "Should be in eval mode after evaluate()"
    assert reg_model.training is False, "Model should be in eval mode"
    assert len(reg_trainer.history['eval_loss']) == 1, "Should have 1 eval loss recorded"
    assert np.isfinite(eval_loss), f"Eval loss should be finite, got {eval_loss}"

    # --- Classification case: cross-entropy computes argmax accuracy ---
    class ClassificationModel(Module):
        def __init__(self):
            self.layer = Linear(2, 3)  # 3-class output
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    cls_model = ClassificationModel()
    cls_trainer = Trainer(cls_model, SGD(cls_model.parameters(), lr=0.01), SoftMaxCrossEntropy())

    cls_dataloader = [
        (Tensor([[1.0, 0.5]]), Tensor([0])),   # integer class label
        (Tensor([[0.5, 1.0]]), Tensor([2]))
    ]

    _, cls_accuracy = cls_trainer.evaluate(cls_dataloader)
    assert isinstance(cls_accuracy, (float, np.floating)), "Classification accuracy must be a float"
    # accuracy is 0.0 or 1.0 here -- just verify it was actually computed (total > 0)
    assert 0.0 <= cls_accuracy <= 1.0, f"Classification accuracy out of range: {cls_accuracy}"

    print(f"  Regression  - loss: {eval_loss:.4f}, accuracy: {accuracy:.4f} (expected 0.0)")
    print(f"  Classification - accuracy: {cls_accuracy:.4f} (range check passed)")
    print("✅ Trainer.evaluate works correctly!")
    
    
def test_unit_trainer_save_checkpoint():
    """🧪 Test Trainer.save_checkpoint implementation."""
    print("🧪 Unit Test: Trainer.save_checkpoint...")

    class DummyModel(Module):
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = DummyModel()
    optimizer = SGD(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer, MSE())

    # Set some state to verify it persists
    trainer.epoch = 5
    trainer.step = 100
    trainer.history['train_loss'].append(0.5)

    checkpoint_path = "/tmp/test_save_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    # Verify file exists
    assert os.path.exists(checkpoint_path), "Checkpoint file should exist"

    # Verify contents
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    assert checkpoint['epoch'] == 5, f"Expected epoch=5, got {checkpoint['epoch']}"
    assert checkpoint['step'] == 100, f"Expected step=100, got {checkpoint['step']}"
    assert 'model_state' in checkpoint, "Missing model_state"
    assert 'optimizer_state' in checkpoint, "Missing optimizer_state"
    assert 'history' in checkpoint, "Missing history"
    assert len(checkpoint['history']['train_loss']) == 1, "History should have 1 entry"

    # Clean up
    os.remove(checkpoint_path)

    print("✅ Trainer.save_checkpoint works correctly!")
    
    
def test_unit_trainer_load_checkpoint():
    """🧪 Test Trainer.load_checkpoint implementation."""
    print("🧪 Unit Test: Trainer.load_checkpoint...")

    class DummyModel(Module):
        def __init__(self):
            self.layer = Linear(2, 1)
            self.training = True
        def forward(self, x):
            return self.layer.forward(x)
        def parameters(self):
            return self.layer.parameters()

    model = DummyModel()
    optimizer = SGD(model.parameters(), lr=0.05)
    trainer = Trainer(model, optimizer, MSE())

    # Set distinctive state
    trainer.epoch = 7
    trainer.step = 200
    trainer.history['train_loss'].extend([0.9, 0.7, 0.5])
    trainer.training_mode = False

    # Save original model weights for comparison
    original_weights = model.parameters()[0].data.copy()

    checkpoint_path = "/tmp/test_load_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    # Corrupt state
    trainer.epoch = 999
    trainer.step = 0
    trainer.history = {'train_loss': [], 'eval_loss': [], 'learning_rates': []}
    trainer.training_mode = True

    # Restore
    trainer.load_checkpoint(checkpoint_path)

    # Verify restoration
    assert trainer.epoch == 7, f"Expected epoch=7, got {trainer.epoch}"
    assert trainer.step == 200, f"Expected step=200, got {trainer.step}"
    assert trainer.training_mode is False, "training_mode should be restored to False"
    assert len(trainer.history['train_loss']) == 3, "History should have 3 entries"
    assert trainer.history['train_loss'] == [0.9, 0.7, 0.5], "History values should match"

    # Verify model weights restored
    restored_weights = model.parameters()[0].data
    assert np.allclose(restored_weights, original_weights), "Model weights should be restored"

    # Clean up
    os.remove(checkpoint_path)

    print("✅ Trainer.load_checkpoint works correctly!")
    
def demonstrate_complete_training_pipeline():
    """
    Complete end-to-end training example using all components.

    This demonstrates how Trainer, scheduler, gradient clipping,
    and checkpointing work together in a real training scenario.
    """
    print("🏗️ Building Complete Training Pipeline...")
    print("=" * 60)

    # Step 1: Create model using REAL Linear layer
    class SimpleNN(Module):
        def __init__(self):
            self.layer1 = Linear(3, 5)
            self.relu = ReLU()
            self.layer2 = Linear(5, 2)
            self.training = True

        def forward(self, x):
            x = self.layer1.forward(x)
            x = self.relu.forward(x)   # through the operation object, so gradients reach layer1
            x = self.layer2.forward(x)
            return x

        def parameters(self):
            return self.layer1.parameters() + self.layer2.parameters()

    print("✓ Model created: 3 → 5 → 2 network")

    # Step 2: Create optimizer
    model = SimpleNN()
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    print("✓ Optimizer: SGD with momentum")

    # Step 3: Create loss function
    loss_fn = MSE()
    print("✓ Loss function: MSE")

    # Step 4: Create scheduler
    scheduler = CosineScheduler(max_lr=0.1, min_lr=0.001, total_epochs=5)
    print("✓ Scheduler: Cosine annealing (0.1 → 0.001)")

    # Step 5: Create trainer with gradient clipping
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=1.0
    )
    print("✓ Trainer initialized with gradient clipping")

    # Step 6: Create synthetic training data
    rng = np.random.default_rng(seed=42)
    train_data = [
        (Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((4, 2)))),
        (Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((4, 2)))),
        (Tensor(rng.standard_normal((4, 3))), Tensor(rng.standard_normal((4, 2))))
    ]
    print("✓ Training data: 3 batches of 4 samples")

    # Step 7: Train for multiple epochs
    print("\n🚀 Starting Training...")
    print("-" * 60)
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Learning Rate':<15}")
    print("-" * 60)

    for epoch in range(3):
        loss = trainer.train_epoch(train_data)
        lr = scheduler.get_lr(epoch)
        print(f"{epoch:<8} {loss:<12.6f} {lr:<15.6f}")

    # Step 8: Save checkpoint
    checkpoint_path = "/tmp/training_example_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\n✓ Checkpoint saved: {checkpoint_path}")

    # Step 9: Evaluate
    eval_loss, _ = trainer.evaluate(train_data)
    print(f"✓ Regression evaluation - MSE loss: {eval_loss:.6f}")

    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n" + "=" * 60)
    print("✅ Complete training pipeline executed successfully!")
    print("\n💡 This pipeline demonstrates:")
    print("   • Model → Optimizer → Loss → Scheduler → Trainer integration")
    print("   • Training loop with scheduling and gradient clipping")
    print("   • Checkpointing for training persistence")
    print("   • Evaluation mode for model assessment")
    
if __name__ == "__main__":
    test_unit_cosine_schedule()
    test_unit_clip_grad_norm()
    test_unit_trainer_init()
    test_unit_trainer_process_batch()
    test_unit_trainer_optimizer_update()
    test_unit_trainer_train_epoch()
    test_unit_trainer_evaluate()
    test_unit_trainer_save_checkpoint()
    test_unit_trainer_load_checkpoint()
    demonstrate_complete_training_pipeline()    
    