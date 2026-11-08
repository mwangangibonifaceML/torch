from __future__ import annotations
import numpy as np
import math

from sklearn.datasets import make_regression
from minitorch.tensor import Tensor
from minitorch.nn import Parameter, Module
from minitorch.losses import MSE
from minitorch.optim import SGD
from minitorch.training import Trainer
from minitorch.dataloader import DataLoader, TensorDataset

class Regularization:
    def __init__(self, alpha: float) -> None:
        if alpha < 0:
            raise ValueError("alpha must be a non-negative float.")
        self.alpha = alpha

class L1_regularization(Regularization):
    """L1 regularization class.
    
    Adds a penalty equal to the absolute value of the
    magnitude of coefficients."""
    def __init__(self, alpha: float=0.5) -> None:
        super().__init__(alpha)
        
    def __call__(self, weights: Tensor) -> Tensor:
        n = weights.shape[0]
        weights = Tensor._ensure_tensor(weights)
        return self.alpha * weights.abs().sum()

    
class L2_regularization(Regularization):
    """L2 regularization class.
    
    Adds a penalty equal to the square of the magnitude of coefficients."""
    def __init__(self, alpha: float=0.5) -> None:
        super().__init__(alpha)
        self.alpha = alpha
        
    def __call__(self, weights: Tensor) -> Tensor:
        n = weights.shape[0]
        weights = Tensor._ensure_tensor(weights)
        return self.alpha * (weights ** 2).sum()

class Regression(Module):
    """Base regression model. Models the relationship between
    a scalar dependent variable y and the independent variables X."""
    def __init__(self, degree: int | None, n_iterations: int, learning_rate: float,
                regularization: Regularization | None, seed: int | None =42, *args, **kwds) -> None:
        super().__init__()
        if degree is not None and degree < 1:
            raise ValueError("Degree must be a positive integer.")
        if n_iterations < 1:
            raise ValueError("n_iteration must be a positive integer.")
        if learning_rate < 0 or learning_rate > 1.0:
            raise ValueError("learning rate must lie in the range(0,1)")
        if alpha is not None:
            if alpha < 0.0:
                raise ValueError("alpha must be a nonnegative integer.")
        self.degree = degree
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.regularization = regularization if regularization is not None else lambda x: 0
        self.loss_fn = MSE()
        self.rng = np.random.default_rng(seed=seed) if seed is not None else np.random.default_rng()
        self.training_losses: list[float] = []
        self._mean = None
        self._std = None  
        
    def _design_matrix(self, X: Tensor, fit: bool=False) -> Tensor:
        """Design matrix for Lasso regression."""
        X = Tensor._ensure_tensor(X).data.astype(np.float32)
        
        if X.data.ndim == 1:
            X.data.reshape(-1,1)
        if X.data.ndim != 2:
            raise ValueError("Input data must be 1D or 2D")
        if self.degree is not None and self.degree >1:
            if X.data.shape[-1] != 1:
                raise ValueError("degree > 1 requires a single input feature.")
            X = np.vander(X[:, 0], N=self.degree+1, increasing=True)[:, 1:]

        if fit:
            self._mean = X.mean(axis=0)
            std = X.var(axis=0) ** 0.5
            std[std == 0] = 1.0
            self._std = std
        else:     
            if self._mean is None:
                raise RuntimeError("model is not fitted yet")
        X = (X - self._mean) / self._std
        return Tensor(np.hstack([np.ones((X.shape[0], 1)), X.data]))
    
    @staticmethod
    def _prepare_y(y: Tensor, n_samples: int) -> Tensor:
        """Prepare the target variable for Lasso regression."""
        y = Tensor._ensure_tensor(y)
        if y.data.ndim != 1:
            raise ValueError("Target variable must be 1D")
        if y.data.shape[0] != n_samples:
            raise ValueError("Number of samples in X and y must match. X has {} samples, but y has {} samples.".format(n_samples, y.data.shape[0]))
        return y
    
    def initialize_weights(self, n_features) -> None:
        #* initialize the model weights
        limit = 1 / (math.sqrt(n_features))
        self.weights = Parameter(
            self.rng.uniform(-limit, limit, size=(n_features,)))
        self.optimizer = SGD(self.parameters(), self.learning_rate)
        
    def fit(self, X: Tensor, y: Tensor) -> None:
        X = self._design_matrix(X, fit=True)
        y = self._prepare_y(y, X.shape[0])
        self.initialize_weights(X.shape[-1])
        self._fit_gradient_descent(X, y)
        
    def _fit_gradient_descent(self, X:Tensor, y: Tensor) -> None:
        self.training_losses = []
        for _ in range(self.n_iterations):
            # #* clear the previous gradients
            self.optimizer.zero_grad()
            
            #* forward pass + backward pass
            y_pred = X @ self.weights
            loss = self.loss_fn(y_pred, y) + self.regularization(self.weights)
            loss.backward()
            self.training_losses.append(float(loss.data))
            
            # #* update the weights
            self.optimizer.step()
            
    def predict(self, X: Tensor) -> Tensor:
        X = self._design_matrix(X)
        return X @ self.weights
    
    
class LinearRegression(Regression):
    """
    Linear model.
    
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    gradient_descent: boolean
        True or false depending if gradient descent should be used when training. If 
        false then we use batch optimization by least squares.
    """
    def __init__(self, n_iterations: int=5000, regularization=None, seed=42, learning_rate=0.001, gradient_descent:bool=True) -> None:
        super().__init__(None, n_iterations, learning_rate, regularization, seed)
        self.gradient_descent = gradient_descent
        
    def fit(self, X: Tensor, y: Tensor) -> None:
        "train the linear model using gradient descent or least squares."
        X = Tensor._ensure_tensor(X)
        y = Tensor._ensure_tensor(y)
        X = Tensor(np.insert(X.data, 0, 1, axis=1))
        self.initialize_weights(X.shape[-1])
        
        #* perform least squares
        if not self.gradient_descent:
            X = self._design_matrix(X, fit=True)
            y = self._prepare_y(y, X.shape[0])
            self.initialize_weights(X.shape[1])
            self.weights.data = np.linalg.lstsq(X.data, y.data, rcond=None)[0]
            resid = X @ self.weights.data - y
            self.training_losses = [float(np.mean(resid.data ** 2))]
        else: #* perform gradient descent
            return super().fit(X,y)
        
    def predict(self, X: Tensor) -> Tensor:
        X = Tensor(np.insert(X.data, 0, 1, axis=1))
        return super().predict(X)

class LassoRegression(Regression):
    """
    Least squares with an L1 penalty (alpha * ||w||_1, bias excluded), 
    which encourages sparsity in the model weights.
    
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    alpha: float
        The regularization strength. Must be a positive float. Larger values specify 
        stronger regularization.
    """
    def __init__(self, degree:int, n_iterations: int=5000, learning_rate=0.001, seed=42, alpha=0.5) -> None:
        super().__init__(degree, n_iterations, learning_rate, regularization=L1_regularization(alpha), seed=seed)
            
        
class RidgeRegression(Regression):
    """
    Least squares with an L2 penalty (alpha * ||w||^2, bias excluded)..
    
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    alpha: float
        The regularization strength. Must be a positive float. Larger values specify 
        stronger regularization.
    """
    def __init__(self, degree:int, n_iterations: int=5000, learning_rate=0.001, seed=42, alpha=0.5) -> None:
        super().__init__(degree, n_iterations, learning_rate, regularization=L2_regularization(alpha), seed=seed)
                
        
            
if __name__ == "__main__":
    from sklearn.linear_model import LinearRegression as sklinear
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    
    inputs, targets = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    inputs = Tensor(inputs)
    targets = Tensor(targets)
    
    n,s = len(inputs.data), inputs.data.std()
    sxx = (inputs.data - inputs.data.mean() ** 2).sum()
    alpha = 0.5
    
    regression = LinearRegression()
    ridge = RidgeRegression(degree=1, alpha=0.022)
    lasso = LassoRegression(degree=1, alpha=alpha)
    
    regression.fit(inputs, targets)
    ridge.fit(inputs, targets)
    lasso.fit(inputs, targets)
    print(regression.weights, ridge.weights, lasso.weights)
    
    sk_regression = sklinear()
    sk_ridge = Ridge(alpha=0.5)
    sk_lasso = Lasso(alpha=0.5)
    
    sk_regression.fit(inputs.data, targets.data)
    sk_lasso.fit(inputs.data, targets.data)
    sk_ridge.fit(inputs.data, targets.data)
    
    print(f"manual linear regression results:\n {regression.predict(inputs[:5]), targets[:5]}")
    print(f"manual ridge regression results:\n {ridge.predict(inputs[:5]), targets[:5]}")
    print(f"manual lasso regression results:\n {lasso.predict(inputs[:5]), targets[:5]}")
    
    print(f"sklearn linear regression results:\n {sk_regression.predict(inputs[:5].data), targets[:5].data}")
    print(f"sklearn ridge regression results:\n {sk_ridge.predict(inputs[:5].data), targets[:5].data}")
    print(f"sklearn lasso regression results:\n {sk_lasso.predict(inputs[:5].data), targets[:5].data}")
    # print(sk_ridge., sk_lasso.get_params(), sk_regression.get_params())
    # print()
    