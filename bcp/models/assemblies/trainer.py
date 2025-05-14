import logging

import jax
import jax.numpy as jnp

from typing import Iterable

# local
from ...core import FeedbackControlTrainer

# Logger
import logging
logger = logging.getLogger(__name__)

EPSILON = 1e-6
    
    
# DEVIATION FROM BALANCE @ STEADY-STATE
# # # # # # # # # # # # # # # # # # # #

class BalanceControlled(FeedbackControlTrainer):
    
    activity_reg_strength: float = 0.0
    gate_by_derivative: bool = False
    use_fr_error: bool = True
    
    class ExtTrainState(FeedbackControlTrainer.ExtTrainState):
        mean_activity: Iterable[jnp.ndarray]

    def init_trainstate_params(self, params):
        """ Initializes the extra parameters of the train state. """
        
        ensemble_sizes, sizes_exc, sizes_inh = self.model.vf._get_hidden_sizes()
            
        mean_activity = [jnp.zeros((s,), dtype=self.model.dtype) for s in sizes_exc]
        
        return {'mean_activity': mean_activity}
    
    def update_trainstate_params(self, trainstate, ol_sol, x):
        """ Updates the extra parameters of the train state. """
        
        mean_activity = trainstate.mean_activity
        
        actE = self.model.vf.actE
        actI = self.model.vf.actI
        
        #ff_inputs = jax.vmap(lambda x, s: self.model.vf.compute_ff_inputs(trainstate.params, x, s))(x, ol_sol.ys['vf'])
        
        # unpack vf_sol
        for l in range(self.model.vf.nb_hidden):
            
            u_exc = ol_sol.ys['vf'][l]['exc']
            
            # Update mean activity
            r_exc = actE(u_exc)
            mean_activity[l] = r_exc.mean(0)
            
        return {'mean_activity': mean_activity}
        
    def _get_gradients(self, train_state, x, y, OL_y_pred, CL_y_pred, OL_state, CL_state):
    
        actE = self.model.vf.actE
        actI = self.model.vf.actI
        nb_hidden_layers = len(CL_state['vf']) - 1
        ff_inputs = self.model.vf.compute_ff_inputs(train_state.params , x, CL_state['vf'])
        
        errors = []

        # HIDDEN LAYERS
        # # # # # # # # # # #
        errors = []
        for l in range(nb_hidden_layers):
            
            # Unpack vectorfield state
            u_exc = CL_state['vf'][l]['exc']
            r_exc = actE(u_exc)
            u_inh = CL_state['vf'][l]['inh']
            r_inh = actI(u_inh)
            
            # Get EI / IE weights
            W_IE, W_EI = self.model.vf.apply(
                train_state.params,
                method=self.model.vf.get_recurrent_weight,
                layer_idx=l)
            
            # Calculate inh current and expected current
            I_inh = jnp.dot(r_inh, W_IE)
            
            # Equivalent error at steady state in terms of firing rate
            if self.use_fr_error:
                beta = - self.model.vf.alpha / (self.model.vf.alpha - 1)
                I_inh_expected = beta * r_exc
                
            else:
                I_inh_expected = jax.nn.relu(self.model.vf.alpha * ff_inputs[l]['exc'])

            
            error = (I_inh - I_inh_expected)
            
            # Activity regularization: Penalize mean activity of smaller than 0.01
            error -= (train_state.mean_activity[l] < 0.01) * self.activity_reg_strength
            
            # Gate by derivative
            if self.gate_by_derivative:
                error *= actE.deriv(u_exc)
            
            errors.append(error)
                        
        # READOUT LAYER        
        if self.model.vf.fb_to_readout:
            error = ff_inputs[-1] - CL_state['vf'][-1]
        else:
            error = jax.grad(self.loss, argnums=0)(OL_y_pred, y)
        errors.append(error)
        
        grads = self.model.vf.calculate_gradients(train_state.params, 
                                                  x,
                                                  CL_state['vf'],
                                                  errors)
        
        # set grads for constants to zero
        grads['constants'] = jax.tree_map(lambda x: jnp.zeros_like(x), grads['constants'])
        
        return grads    
    
