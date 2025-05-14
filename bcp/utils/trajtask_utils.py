import jax
import jax.numpy as jnp

from functools import partial
from jax import jit
from diffrax import LinearInterpolation

# LOCAL
from ..data.onlineinputs import SinusoidalInputs
from ..data.shapetrajectory import ShapeTrajectory

# Logger
import logging
logger = logging.getLogger(__name__)

class ShapeTrajectoryTask:
    
    def __init__(self,
                 inputs: SinusoidalInputs,
                 target: ShapeTrajectory):
        
        self.inputs = inputs
        self.target = target

    def simulate(self):
        logger.debug("Generating inputs and target trajectory...")
        x = self.inputs.simulate(self.target.T)
        y = self.target.get_Y()
        
        x_ts = jnp.arange(0, self.target.T, self.inputs.dt)
        x_interp = LinearInterpolation(x_ts, x)
        y_interp = LinearInterpolation(self.target.ts, y)
        logger.debug("Input generation complete.")
        
        return x, y, x_interp, y_interp

# CONTROL NETWORK WITHOUT HIDDEN LAYER:
# # # # # # # # # # # # # # # # # # # # #

class Net_NoHidden:

    "Linear network with no hidden layer"
        
    def __init__(self, nb_inputs, nb_output, tau):
        self.nb_inputs = nb_inputs
        self.nb_output = nb_output
        self.tau = tau
        
    @partial(jit, static_argnums=(0,))
    def __call__(self, state, params, t, inputs):
                
        x = inputs.evaluate(t)
        _, u = state
        W = params

        du = 1 / self.tau * (-u + jnp.dot(x, W))
                    
        return (0, du)
    
    def get_initial_state(self):
        
        u = jnp.zeros(self.nb_output)
        
        return (0, u)
    
    def get_initial_params(self, key):
        
        W = jax.random.normal(key, (self.nb_inputs, self.nb_output)) * 1 / jnp.sqrt(self.nb_inputs)
        
        return W