from __future__ import annotations
import random
from typing import Any

import os
import pickle
import numpy as np
from pathlib import Path
from sympy import N

from minitorch.optim import Adam, Optimizer, SGD, AdamW
from minitorch.tensor import Tensor
from minitorch.nn import Parameter, Module, Linear
from minitorch.dataloader import DataLoader, TensorDataset
from minitorch.losses import SoftMaxCrossEntropy, BCEWithLogits, MSE, Loss

def clip_grad_clip_norm(paramters: list[Parameter], max_norm: float= 1.0) -> float:
    """
    clip gradients by global norm to prevent exploding gradients.
    
    Instead of clipping each gradient individually, calculate the global
    norm across all paramters and scale unformly if needed.
    """
    if not paramters:
        return 0.0
    if not np.isfinite(max_norm) or max_norm < 0.0:
        raise ValueError("max norm must be finite and nonnegative.")
    
    #* collect all the gradients
    total_norm = 0.0
    for param in paramters:
        if param.grad is not None:
            grad_data = param.grad
            total_norm += np.sum(grad_data.astype(np.float32) ** 2)
            
    total_norm = np.sqrt(total_norm)
    if total_norm > max_norm:
        clip_coef = float(max_norm / total_norm)
        for param in paramters:
            if param.grad is not None:
                param.grad *= clip_coef
                
    return float(total_norm)
            

class CosineScheduler(object):
    """
    Cosine annealling learning rate schedule.
    
    Starts at max_lr decreases following a cosine curve to min_lr
    over T epochs. This provies aggresive learning initially, then 
    fine-tuning at the end

    """
    def __init__(self, min_lr: float, max_lr: float, total_epochs: int) -> None:
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        
    def get_lr(self, epoch: int) -> float:
        "Get learning rate the current epoch"
        if epoch >= self.total_epochs:
            return self.min_lr
        
        cosine_annelling = (1 + np.cos(np.pi * epoch/ self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_annelling
    


class Trainer(object):
    """
    Complete training orchestrator for neural networks.

    Handles the full training lifecycle: forward pass, loss computation,
    backward pass, optimization, scheduling, checkpointing, and evaluation.

    This is the central class that brings together all the components
    you've built in previous modules.

    """
    def __init__(self, 
                model: Module,
                optimizer: Optimizer, 
                loss_fn: Loss,
                scheduler: CosineScheduler | None = None, 
                grad_clip_norm:float | None = None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.loss_fn = loss_fn
        
        
        self.training_mode = True
        self.step = 0
        self.epoch = 0
        
        #* history
        self.history = {
            "train_loss": [],
            "eval_loss":  [],
            "learning_rates": []
        }
    
    def _process_batch(self, features: Tensor, targets: Tensor) -> float:
        "call the forward pass on the model and the backward pass."
        batch_size = features.shape[0]
        if batch_size ==0:
            raise ValueError("Training batches must contain at least one sample")
        
        outputs = self.model(features)
        
        if outputs.data.ndim == 3 and targets.data.ndim == 2:
            vocab_size = outputs.shape[-1]
            output_reshaped = outputs.reshape(-1, vocab_size)
            loss = self.loss_fn(output_reshaped, targets.reshape(-1))
        else:
            loss = self.loss_fn(outputs, targets)
        
        loss.backward()
        return float(loss.data)
        
    def train_epoch(self, dataloader: DataLoader, accumulation_steps: int=2) -> float:
        if not isinstance(accumulation_steps, (int, np.integer)) or accumulation_steps < 1:
            raise ValueError("accumulation steps must be nonnegative integer")
        
        self.model.train(mode=True)
        self.training_mode = True
        
        #* calculate learning rate for the epoch
        if self.scheduler is not None:
            lr = self.scheduler.get_lr(self.epoch)
            self.optimizer.lr = lr
            self.history["learning_rates"].append(float(lr))
        
        #* track stats for the epoch
        total_loss = 0.0
        total_samples = 0
        accumulated_loss = 0.0
        batches_in_accumulation = 0
        samples_in_accumulation = 0
        
        for i, (features, targets) in enumerate(dataloader):
            batch_loss = self._process_batch(features, targets)            
            batch_size = features.shape[0]
            total_samples += batch_size
            total_loss += batch_loss * batch_size
            
            batches_in_accumulation += 1
            samples_in_accumulation += batch_size
            
            #* update parameters every accumulation steps
            if batches_in_accumulation == accumulation_steps:
                self._optimizer_update(samples_in_accumulation)
                accumulated_loss += total_loss
                self.step += 1
                
                batches_in_accumulation = 0
                samples_in_accumulation = 0
            
        #* for batch count not divisible by accumulation steps
        #* makes sure that there is no gradient lost
        if batches_in_accumulation > 0:
            self._optimizer_update(samples_in_accumulation)
            self.step += 1
            
        average_loss = total_loss / max(total_samples, 1)
        self.history["train_loss"].append(average_loss)
        
        self.epoch += 1
        return average_loss
    
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval(mode=False)
        self.training_mode = False
        
        total_loss = 0.0
        correct = 0
        total = 0
        total_samples = 0

        for inputs, targets in dataloader:
            #* Forward pass only
            outputs = self.model(inputs)
            if outputs.data.ndim == 3 and targets.data.ndim == 2:
                vocab_size = outputs.shape[-1]
                output_reshaped = outputs.reshape(-1, vocab_size)
                loss = self.loss_fn(output_reshaped, targets.reshape(-1))
            else:
                loss = self.loss_fn(outputs, targets)

            batch_size = inputs.shape[0]
            total_loss += float(loss.data) * batch_size
            total_samples += batch_size

            # The loss defines the task
            if isinstance(self.loss_fn, SoftMaxCrossEntropy):
                predictions = np.argmax(outputs.data, axis=-1)
                correct += np.sum(predictions == targets.data)
                total += predictions.size
            elif isinstance(self.loss_fn, BCEWithLogits):
                predictions = outputs.data >= 0.0
                correct += np.sum(predictions == (targets.data >= 0.5))
                total += predictions.size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = float(correct / total) if total > 0 else 0.0

        self.history['eval_loss'].append(avg_loss)

        return avg_loss, accuracy
    
    def _get_optimizer_state(self) -> dict[str, any] | None:
        if self.optimizer is None:
            return
        states = {}
        optim_attrs = ('lr', 'step_count', 'momentum', 'beta1', 'beta2', 'eps', 'weight_decay')
        for attr in optim_attrs:
            if hasattr(self.optimizer, attr):
                states[attr] = getattr(self.optimizer, attr)
        if hasattr(self.optimizer, "has_momentum") and self.optimizer.has_momentum():
            momentum = self.optimizer.get_momentum_state()
            if momentum is not None:
                states["momentum_buffers"] = momentum
        return states
    
    def _set_optimizer_state(self, state: dict[str, any] | None) -> None:
        if state is None or self.optimizer is None:
            return 
        optim_attrs = ('lr', 'step_count', 'momentum', 'beta1', 'beta2', 'eps', 'weight_decay')
        for attr in optim_attrs:
            if attr in state and hasattr(self.optimizer, attr):
                setattr(self.optimizer, attr, state[attr])
        if "momentum_buffers" in state:
            if hasattr(self.optimizer, "has_momentum") and self.optimizer.has_momentum():
                self.optimizer.set_momentum_state(state["momentum_buffers"])
            
    def _optimizer_update(self, sample_count: int) -> None:
        params = list(self.model.parameters())
        for param in params:
            if param.grad is not None:
                param.grad /= sample_count
        if self.grad_clip_norm is not None:
            clip_grad_clip_norm(params, self.grad_clip_norm)
            
        self.optimizer.step()
        self.optimizer.zero_grad()
            
    def _get_scheduler_state(self) -> dict[str, any] | None:
        if self.scheduler is None:
            return
        return {
            "min_lr": getattr(self.scheduler, "min_lr", None),
            "max_lr": getattr(self.scheduler, "max_lr", None),
            "total_epochs": getattr(self.scheduler, "total_epochs", None)
        }
        
    def _set_scheduler_state(self, state: dict[str, any] | None) -> None:
        if state is None or self.scheduler is None:
            return
        for key, value in state.items():
            if hasattr(self.scheduler, key):
                setattr(self.scheduler, key, value)
    
    def _get_model_state(self) -> dict[int, np.ndarray]:
        "Extract model parameters for checkpointing."
        return {i: param.data.copy() for i,param in enumerate(self.model.parameters())}
    
    def _set_model_state(self, state: dict[int, np.ndarray] | None) -> None:
        "Restore model parameters"
        if state is None:
            return 
        
        parameters = self.model.parameters()
        if set(state) != set(range(len(parameters))):
            raise ValueError(
                "mismatch, checkpointing parameter count does not match model"
                )
            
        #* validate all shapes before modifyinh any parameters
        for i,param in enumerate(self.model.parameters()):
            if state[i].shape != param.shape: 
                raise ValueError(
                    f"Parameter {i} shape mismatch: "
                    f"checkpoint={state[i].shape}, "
                    f"model={param.shape}"
                    )
                
        #* restore parameters
        for i, param in enumerate(parameters):
            param.data = state[i].copy()
    
    def save_checkpoint(self, path: str) -> None:
        "save parameters, optimizer, scheduler, and epoch history for resumption."
        checkpoint = {
            "epoch": self.epoch,
            "step": self.step,
            "model_state": self._get_model_state(),
            "optimizer_state": self._get_optimizer_state(),
            "scheduler_state": self._get_scheduler_state(),
            "history": self.history,
            "training_mode": self.training_mode,
            "grad_clip_norm": self.grad_clip_norm
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        
    def load_checkpoint(self, path: str) -> None:
        "Load training state from checkpoint: restores exact training state."
        with open(path, "rb") as file:
            checkpoint = pickle.load(file)
            
        self.epoch = checkpoint["epoch"]
        self.step = checkpoint["step"]
        self.training_mode = checkpoint["training_mode"]
        self.grad_clip_norm = checkpoint["grad_clip_norm"]
        self.model.training = self.training_mode
        self.history = checkpoint["history"]
        
        #* restore state: model, optimizer and scheduler.
        self._set_model_state(checkpoint.get("model_state", None))
        self._set_optimizer_state(checkpoint.get("optimizer_state", None))
        self._set_scheduler_state(checkpoint.get("scheduler_state", None))
        