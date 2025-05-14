# Controller 
# Julian Rossbroich
# 2025

import logging

import jax.numpy as jnp
import flax.linen as nn

# SET UP LOGGER
logger = logging.getLogger("Model")


class Controller(nn.Module):
    
    loss: nn.Module
    dim_output: int

    def __call__(self, y_pred, y_target, state):
        """
        A call to the controller takes the predicted output of the vector field,
        the target output, and the current state of the controller.
        
        It returns the control signal and the updated controller state.
        """
        pass
    
    def get_initial_state(self):
        pass
        
    def get_initial_state_onlineVF(self):
        pass
                


class ProportionalController(Controller):

    k_p: float
    
    def __call__(self, y_pred, y_target, state):
        
        error = self.loss.get_error(y_pred, y_target)
        c = self.k_p * error
        
        return c, {}
    
    @staticmethod
    def get_initial_state():
        return {}
    
    @staticmethod
    def get_initial_state_onlineVF():
        return {}
    
class LeakyPIController(Controller):
    
    k_p: float
    k_i: float
    leak: float
    tau: float
    
    def __call__(self, y_pred, y_target, state):
        
        error = self.loss.get_error(y_pred, y_target)

        # unpack state
        c_int = state['c_int']
        c = self.k_i * c_int + self.k_p * error

        # update 
        delta_c_int = 1 / self.tau * (error - self.leak * c_int)
        
        # pack state
        delta_state = {'c_int': delta_c_int}
        
        return c, delta_state
    
    def get_initial_state(self):
        return {'c_int': jnp.zeros(self.dim_output)}
    
    def get_initial_state_onlineVF(self):
        return {'c_int': jnp.zeros((1, self.dim_output))}
    
class PIDController(Controller):
    
    k_p: float
    k_i: float
    k_d: float
    leak: float
    tau: float
    
    def __call__(self, y_pred, y_target, state):
        
        error = self.loss.get_error(y_pred, y_target)
        de = error - state['e_prev']
        
        # unpack state
        c_int = state['c_int']
        c = self.k_i * c_int + self.k_p * error + self.k_d * de

        # update 
        delta_c_int = 1 / self.tau * (error - self.leak * c_int)
        
        # pack state
        delta_state = {'c_int': delta_c_int,
                       'e_prev': error}
        
        return c, delta_state
    
    def get_initial_state(self):
        return {'c_int': jnp.zeros(self.dim_output),
                'e_prev': jnp.zeros(self.dim_output)}
    
    def get_initial_state_onlineVF(self):
        return {'c_int': jnp.zeros((1, self.dim_output)),
                'e_prev': jnp.zeros((1, self.dim_output))}
