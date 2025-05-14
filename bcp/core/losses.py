# Loss functions
# Julian Rossbroich
# 2025

    
import jax.numpy as jnp
from jax.nn import softmax, log_softmax

import flax.linen as nn

from abc import abstractmethod

# Custom implementation of loss functions.
# Loss functions are implemented as modules, so that they can be passed to the controller
# The __call__ function of the loss is designed to take in a single sample, and return the loss for that sample.
# To calculate the loss across a batch, the trainer class v-maps the loss function over the batch dimension.

class Loss(nn.Module):
    """
    Abstract class for loss functions
    """
    
    name: str = 'loss'
    
    @abstractmethod
    def __call__(self, y_pred, y_true):
        raise NotImplementedError

    @abstractmethod
    def get_error(self, y_pred, y_target):
        """ 
        returns the error vector for the controller.
        Hard-coded version of jax.grad(__call__, argnums=0)(y_pred, y_true)
        """
        raise NotImplementedError

    def get_nudge_targets(self, y_pred, y_target, nudge_factor):
        """
        Returns the nudged target vector for the controller.
        """
        raise NotImplementedError

class MSE(Loss):
    """ 
    Mean squared error loss 
    """
    name: str = 'mse'
    
    def __call__(self, y_pred, y_true):
        return 1/2 * jnp.sum(jnp.square(y_true - y_pred))
    
    def get_error(self, y_pred, y_target):
        """ 
        returns the error vector for the controller.
        Hard-coded version of - jax.grad(loss, argnums=0)(y_pred, y_true)
        """
        return y_target - y_pred
    
    def get_nudge_targets(self, y_pred, y_target, nudge_factor):
        """
        Returns the nudged target vector for the controller.
        """
        return y_pred + nudge_factor * self.get_error(y_pred, y_target)

class SoftmaxCrossEntropy(Loss):
    """ 
    Softmax cross entropy loss 
    """
    name: str = 'cross_entropy'

    def __call__(self, y_pred, y_true):
        return - jnp.sum(y_true * log_softmax(y_pred))
    
    def get_error(self, y_pred, y_target):
        """ 
        returns the error vector for the controller.
        Hard-coded version of - jax.grad(loss, argnums=0)(y_pred, y_true)
        """
        return y_target - softmax(y_pred)
    
    def get_nudge_targets(self, y_pred, y_target, nudge_factor):
        """
        Returns the nudged target vector for the controller.
        """
        return softmax(y_pred + nudge_factor * self.get_error(y_pred, y_target))


class SigmoidCrossEntropy(Loss):
    """
    Sigmoid cross entropy loss with optional shifting/steepness.
    """
    name: str = 'sigmoid_cross_entropy'
    shift: float = 0.0      # shifts the input to the sigmoid
    steepness: float = 1.0  # scales the input to the sigmoid

    def sigmoid(self, x):
        """
        Shifted & steepened sigmoid:
            σ(z) = 1 / (1 + exp(steepness * (-z + shift)))
        """
        return 1 / (1 + jnp.exp(self.steepness * (-x + self.shift)))

    def __call__(self, y_pred, y_true):
        """
        y_pred: output of a linear layer
        y_true: targets in {0,1}, same shape as y_pred

        returns: sum of binary cross-entropy losses
                 = -Σ [y_true*log(p) + (1-y_true)*log(1-p)]
        """
        eps = 1e-7
        p = jnp.clip(self.sigmoid(y_pred), eps, 1 - eps)
        return -jnp.sum(y_true * jnp.log(p)
                        + (1 - y_true) * jnp.log(1 - p))

    def get_error(self, y_pred, y_true):
        """
        returns the error vector for the controller,
        i.e. hard-coded -∂L/∂z = y_true - σ(z)
        """
        return y_true - self.sigmoid(y_pred)

    def get_nudge_targets(self, y_pred, y_true, nudge_factor):
        raise NotImplementedError