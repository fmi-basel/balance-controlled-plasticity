# Parameterized activation functions
# Julian Rossbroich
# 2025

import jax 
import jax.numpy as jnp

from abc import ABC, abstractmethod

EPSILON = 1e-8

class ActivationFunction(ABC):
    """
    Abstract base class for parametrized activation functions.
    """
    
    def __init__(self):
        self._params = []
        self._param_names = []
        
    def add_param(self, value: float, name: str = None):
        self._params.append(value)
        self._param_names.append(str(name))
        return value
    
    @property
    def params(self):
        return dict(zip(self._param_names, self._params))
    
    @abstractmethod
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
    @abstractmethod
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Derivative of the activation function
        """
        raise NotImplementedError
    
    @abstractmethod
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Inverse of the activation function
        """
        raise NotImplementedError
    
    def inv_lin_taylor(self, y: jnp.ndarray, y0: jnp.ndarray) -> jnp.ndarray:
        """
        Linear approximation of the inverse function using taylor expansion around
        the point y0.
        
        :param y: The output of the function.
        :param y0: The output of the function at the point of linearization.
        :return: Linear approximation of the inverse function.
        """
        intercept, slope = self._get_taylorexp_model(y0)
        return intercept + slope * y
    
    def inv_linreg(self, y: float, x_start: float = None, x_end: float = None, num_points: int = None):
        """
        Compute the approximate inverse using the linear regression model.
        
        :param y: The output of the function.
        :param y0: The output of the function at the point of linearization.
        :param x_start: Starting point of the input range for the linear regression model (optional).
        :param x_end: Ending point of the input range for the linear regression model (optional).
        :param num_points: Number of data points to be used in the linear regression model (optional).
        :return: Linear regression approximation of the inverse function.
        """
        # Calculate the linear regression model if it has not been established yet
        intercept, slope = self._get_linreg_model(x_start, x_end, num_points)

        # Use the established linear regression model to approximate the inverse
        return intercept + slope * y
 
    def _get_linreg_model(self, x_start: float, x_end: float, num_points: int):
        """
        Calculate the linear regression model for the inverse function.
        
        :param x_start: Starting point of the input range.
        :param x_end: Ending point of the input range.
        :param num_points: Number of data points to be used in the linear regression.
        :return: A tuple (intercept, slope) representing the linear regression model of the inverse function.
        """
        x_values = jnp.linspace(x_start, x_end, num_points)
        y_values = self(x_values)
        slope, intercept = jnp.polyfit(y_values, x_values, 1)
        
        return intercept, slope
    
    def _get_taylorexp_model(self, y0):
        """
        Calculate a linear model approximating the inverse function using taylor expansion.
        
        :param y0: The output of the function at the point of linearization.
        :return: A tuple (intercept, slope) representing the linear model of the inverse function.
        """
        inv_y0 = self.inv(y0)
        deriv_y0 = self.deriv(inv_y0)
        intercept = inv_y0 - y0 / deriv_y0
        slope = 1 / deriv_y0
        
        return intercept, slope
    
class Linear(ActivationFunction):
    """
    Linear activation function.
    """
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(x)
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return y
    

class Tanh(ActivationFunction):

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.tanh(x)
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return 1 - jax.nn.tanh(x)**2
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.arctanh(y)


class ReLU(ActivationFunction):
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.relu(x)
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(x > 0, 1, 0)
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(y > 0, y, 0)
    
    
class Exp(ActivationFunction):
    def __init__(self, exp: float):
        super().__init__()
        
        self.exp = self.add_param(exp, 'exp')
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x ** self.exp
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.exp * x ** (self.exp - 1)
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return y ** (1 / self.exp)


class Sigmoid(ActivationFunction):
    
    def __init__(self):
        super().__init__()
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(x)
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.sigmoid(x) * (1 - jax.nn.sigmoid(x))
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(y / (1 - y))


class ParameterizedLogisticFunction(ActivationFunction):
    """
    Parameterized logistic function
    """
    
    def __init__(self, x0: float, max: float, slope: float):
        super().__init__()
        self.x0 = self.add_param(x0, 'x0')
        self.max = self.add_param(max, 'max')
        self.slope = self.add_param(slope, 'slope')
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.max / (1 + jnp.exp(-self.slope * (x - self.x0)))
    
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.slope * self.max * jnp.exp(-self.slope * (x - self.x0)) / ((1 + jnp.exp(-self.slope * (x - self.x0)))**2)
    
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        return self.x0 - jnp.log(self.max / (y+EPSILON) - 1) / self.slope
    
    
class SoftReLu(ActivationFunction):
    
    def __init__(self, scale: float, sharpness: float, shift: float):
        """
        Soft ReLU function with tunable parameters.
        
        :param scale: Controls the overall scaling of the function.
        :param sharpness: Controls the sharpness or smoothness of the function.
        :param shift: Controls the horizontal shift of the function.
        """
        super().__init__()
        self.scale = self.add_param(scale, 'scale')
        self.sharpness = self.add_param(sharpness, 'sharpness')
        self.shift = self.add_param(shift, 'shift')
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the value of the soft ReLU function for the given input.
        
        :param x: Input to the soft ReLU function.
        :return: Output of the soft ReLU function.
        """
        return self.scale * jnp.logaddexp(self.sharpness * (x - self.shift), 0)
        
    def deriv(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the derivative of the soft ReLU function for the given input.
        
        :param x: Input to the soft ReLU function.
        :return: Derivative of the soft ReLU function.
        """
        gradunfc = lambda x: jnp.sum(self.__call__(x))
        return jax.grad(gradunfc)(x)
        
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the inverse of the soft ReLU function for the given output.
        
        :param y: Output of the soft ReLU function.
        :return: Input that produced the given output.
        """
        
        return (self.shift * self.sharpness + jnp.log(jnp.exp(y / self.scale) - 1)) / self.sharpness
        
    def inv(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the inverse of the soft ReLU function for the given output.
        
        :param y: Output of the soft ReLU function.
        :return: Input that produced the given output.
        """
        large_y = y / self.scale > 50   # Threshold to switch to linear approximation
                                        # Necessary for numerical stability
        stable_inv = jnp.log(jnp.expm1(y / self.scale))
        return jnp.where(large_y, y / self.scale, self.shift + (1 / self.sharpness) * stable_inv)
