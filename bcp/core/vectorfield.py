# VectorField base class
# Julian Rossbroich
# 2025

import jax
import jax.numpy as jnp

import flax.linen as nn

from typing import Tuple
from abc import abstractmethod

import logging

# SET UP LOGGER
logger = logging.getLogger(__name__)

class VectorField(nn.Module):

    dim_output: int
    flatten_input: bool 
    dtype: jnp.dtype

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, state, x, y, fb_weights, closedloop):
        raise NotImplementedError
    
    @abstractmethod
    def get_initial_state(self, x) -> jnp.ndarray:
        """ Get initial vf state """
        raise NotImplementedError
    
    def get_initial_state_batchexp(self, x):
        """ Get initial vdf state expanded along the batch dimension """
        batchsize = x.shape[0]
        state0 = self.get_initial_state(x)
        return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0).repeat(batchsize, axis=0), state0)

    @abstractmethod
    def out(self, sol: Tuple):
        """
        Given a diffrax solution (sol.ys), return the output of the vector field
        """
        raise NotImplementedError

    def get_in_axes(self, closedloop: bool):
        """
        in_axes for vmapping across batch dimension of __call__ method
        """
        if closedloop:
            return (None, 0, 0, 0, 0, None)
        else:
            return (None, 0, 0, None, None, None)
        
    @abstractmethod
    def get_init_kwargs(self, *args):
        """ Get kwargs for the `init` method"""
        return {}
    
    @abstractmethod
    def calculate_jacobian(self, vf_state):
        """ Calculates the Jacobian at the current VF state """
        raise NotImplementedError
    
    @abstractmethod
    def calculate_gradients(self, x, vf_state, errors):
        """ given an input and VF state, returns dictionary of gradients """
        raise NotImplementedError
    

class ForwardVectorField(VectorField):
    """ 
    VectorField that has a standard forward pass 
    (i.e. consists of a forward-only operation without
    the use of an ODE solver).
    """

    @abstractmethod
    def forward(self, x):
        """ Standard forward pass """

