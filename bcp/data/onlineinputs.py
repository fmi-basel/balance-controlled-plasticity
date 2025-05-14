# Inputs for online learning tasks
# # # # # # # # # # # # # # # # # # # # # 
#
# J. Rossbroich
# 2025

import jax
import jax.numpy as jnp
from jax import lax
import time

from diffrax import LinearInterpolation
from abc import ABC, abstractmethod


import numpy as np
from functools import partial

import jax
import jax.numpy as jnp


class OnlineLearningInputs(ABC):
    
    def __init__(self,
                 dt: float,
                 N: int,
                 key: jax.random.PRNGKey = None):
        
        self.dt = dt
        self.N = N
        
        # RNG key
        if key is None:
            key = jax.random.PRNGKey(int(time.time()))
            
        self.key = key
        
        # State
        self.state = None

    def _interpolate(self, values, T):
        
        timesteps = int(T / self.dt)
        t = jnp.linspace(0, T, timesteps)
        
        return LinearInterpolation(ts=t, ys=values)
    
    @abstractmethod
    def reset(self):
        """
        Reset state
        """
        raise NotImplementedError
    
    @abstractmethod
    def simulate(self, T: float):
        """
        Simulate inputs for a given time T
        """
        raise NotImplementedError
    
    
class SinusoidalInputs(OnlineLearningInputs):
    def __init__(self,
                 dt: float,
                 N: int,
                 freq_min: float,
                 amp_min: float,
                 amp_max: float,
                 key: jax.random.PRNGKey = None,
                 linear_combination: bool = False
                 ):
        super().__init__(dt, N, key)

        # Get frequencies, phases, and amplitudes
        subkeys = jax.random.split(self.key, 3)

        # Frequencies are integer multiples of freq_min
        self.frequencies = jnp.arange(1, N+1) * freq_min
        
        # No phase shifts
        self.phases = jnp.zeros(N)
        
        # Random amplitudes
        self.amplitudes = jax.random.uniform(subkeys[0], (N,), minval=amp_min, maxval=amp_max)        

        # Linear combination of sinusoids
        self.linear_combination = linear_combination
        if linear_combination:
            # random weights
            self.W = jax.random.normal(subkeys[1], (N, N))
            
        # Initialize current time
        self.reset()

    def reset(self):
        """
        Reset the internal state.
        """
        self.current_time = 0.0  # Start time at zero

    def simulate(self, T: float):
        """
        Simulate inputs for a given time T.
        Returns the generated inputs over the time interval [current_time, current_time + T].
        """
        timesteps = int(T / self.dt)
        t_start = self.current_time
        t_end = self.current_time + T
        t = jnp.linspace(t_start, t_end, timesteps)  # Time vector of shape [timesteps]

        # Broadcast t to shape [timesteps, 1] for element-wise operations with frequencies, phases, and amplitudes
        t = t[:, None]

        # Vectorized computation of sine wave inputs
        inputs = self.amplitudes * jnp.sin(2 * jnp.pi * self.frequencies * t + self.phases)

        if self.linear_combination:
            inputs = jnp.dot(inputs, self.W)
        
        # Update current time
        self.current_time += T

        return inputs