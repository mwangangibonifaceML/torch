
import pytest
import numpy as np

from minitorch.tensor import Tensor
from minitorch.nn import Parameter
from minitorch.optim import AdamW, Adam, SGD, Optimizer

def test_unit_adam_update_moments():
    """🧪 Test Adam _update_moments computes correct EMA and bias correction."""
    print("🧪 Unit Test: Adam Moment Updates...")
    
    param = Parameter(np.array([1.0,2.0]))
    optimizer = Adam([param], lr=0.01, betas=[0.9,0.999], eps=1e-8)
    grad = np.array([2.0,1.0])
    
    #* auto calculation
    optimizer.step_count = 1
    m_hat, v_hat = optimizer._update_moments(0, grad)
    
    #* manual calculation
    m = 0.9 * 0 + (1-0.9) * grad
    v = 0.999 * 0 + (1-0.999) * grad ** 2
    m_bias_corr = 1 - 0.9 ** 1
    v_bias_corr = 1 - 0.999 ** 1
    m_hat_man = m / m_bias_corr
    v_hat_man = v / v_bias_corr
    
    assert np.allclose(m_hat, m_hat_man), f'first moment should equal grad at step 1, got {m_hat}'
    assert np.allclose(v_hat, v_hat_man), f'second moment should equal the square of the grad at step 1, got {v_hat}'
    
    #* step 2
    optimizer.step_count = 2
    m_hat2, v_hat2 = optimizer._update_moments(0, grad)
    
    assert np.allclose(m_hat, m_hat_man), f'first moment should equal grad at step 1, got {m_hat}'
    assert np.allclose(v_hat, v_hat_man), f'second moment should equal the square of the grad at step 1, got {v_hat}'
        
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None
    
    
def test_unit_adam_optimizer():
    """🧪 Test Adam optimizer implementation."""
    print("🧪 Unit Test: Adam Optimizer...")
    
    param = Parameter([5.2,6.5])
    optimizer = Adam([param], lr=0.001, betas=[0.9,0.999], eps=1e-8)
    param.grad = np.array([1.0,2.0])
    
    #* auto calculation
    optimizer.step()
    
    #* manual caculation
    #? moments calculation
    m = 0.9 * 0 + (1-0.9) * param.grad
    v = 0.999 * 0 + (1-0.999) * param.grad ** 2
    m_bias_corr = 1 - 0.9 ** 1
    v_bias_corr = 1 - 0.999 ** 1
    m_hat_man = m / m_bias_corr
    v_hat_man = v / v_bias_corr
    
    #* param update
    original_data = param.data.copy()
    expected = original_data - 0.001 * m_hat_man / (np.sqrt(v_hat_man) + 1e-8)
    
    assert np.allclose(param.data, expected, rtol=1e-3), f"Expected auto parameter update to be close to manual update, expected {expected}, got {param}"
    
    
def test_unit_adamw_optimizer():
    """🧪 Test AdamW optimizer implementation."""
    print("🧪 Unit Test: AdamW Optimizer...")

    # Test AdamW vs Adam difference in weight decay
    # Create identical parameters for comparison
    param_adam = Parameter([1.0, 2.0])
    param_adamw = Parameter([1.0, 2.0])

    # Create optimizers with same settings
    adam = Adam([param_adam], lr=0.01, weight_decay=0.01)
    adamw = AdamW([param_adamw], lr=0.01, weight_decay=0.01)

    # Set gradients AFTER creating optimizers (optimizer.__init__ resets grad to None)
    param_adam.grad = np.array([0.1, 0.2])
    param_adamw.grad = np.array([0.1, 0.2])

    # Take one step
    adam.step()
    adamw.step()

    # Results should be different due to weight decay implementation
    assert not np.allclose(param_adam.data, param_adamw.data, rtol=1e-6)

    # Test AdamW basic functionality
    param = Parameter([1.0, 2.0])
    optimizer = AdamW([param], lr=0.01, weight_decay=0.01)
    
    # Set gradient AFTER creating optimizer
    param.grad = np.array([0.1, 0.2])
    original_data = param.data.copy()

    optimizer.step()

    # Parameter should have changed
    assert not np.array_equal(param.data, original_data)
    assert optimizer.step_count == 1

    # Test that moment buffers are created
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None

    # Test zero weight decay behaves like Adam
    param1 = Parameter([1.0, 2.0])
    param2 = Parameter([1.0, 2.0])

    adam_no_wd = Adam([param1], lr=0.01, weight_decay=0.0)
    adamw_no_wd = AdamW([param2], lr=0.01, weight_decay=0.0)

    # Set gradients AFTER creating optimizers
    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([0.1, 0.2])

    adam_no_wd.step()
    adamw_no_wd.step()

    # Should be very similar (within numerical precision)
    assert np.allclose(param1.data, param2.data, rtol=1e-10)

if __name__ == '__main__':
    test_unit_adam_update_moments()
    test_unit_adam_optimizer()
    test_unit_adamw_optimizer()