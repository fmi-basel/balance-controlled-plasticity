
import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from functools import partial

import flax.linen as nn
from jax.nn.initializers import Initializer as Initializer
from jax.nn.initializers import constant, orthogonal, normal, uniform
from flax.linen.initializers import lecun_normal, glorot_normal
from flax import struct

from typing import Iterable, Any, List
import dataclasses

# local
from bcp.core.vectorfield import VectorField
from bcp.core.controller import Controller
from bcp.core.activation import ActivationFunction
from bcp.models.assemblies.utils import (
    make_membership_matrices,
    get_gIE_analytic,
    get_W_IE_optimized,
    get_W_from_g_and_M,
)

def _get_hidden_sizes(sizes_hidden, nb_hidden, nb_exc_per_ensemble, EI_ratio):
    # Check sizes of hidden layers. 
    # If only one size is given, repeat it nb_hidden times
    if len(sizes_hidden) == 1:
        ensemble_sizes = [sizes_hidden[0]] * nb_hidden
    else:
        assert len(sizes_hidden) == nb_hidden, "Number of hidden layers does not match number of hidden sizes"
        ensemble_sizes = sizes_hidden   

    sizes_exc = [int(s * nb_exc_per_ensemble) for s in ensemble_sizes]
    sizes_inh = [int(s / EI_ratio) for s in sizes_exc]

    return ensemble_sizes, sizes_exc, sizes_inh


def construct_membership_and_recurrence(RNG_Key,
                                 nb_hidden: int,
                                 sizes_hidden: Iterable[int],
                                 nb_exc_per_ensemble: int,
                                 EI_ratio: float,
                                 alpha: float,
                                 overlap: float,
                                 g_EI: float,
                                 g_XI: float = 1.0,
                                 g_EE: float = 0.0,
                                 g_II: float = 0.0):
    """
    Construct membership matrices and recurrent weights for the assembly model.
    Uses analytic W_IE for non-overlapping assemblies, optimization for overlapping.
    """

    # if RNG key is an integer, convert it to a PRNGKey
    if isinstance(RNG_Key, int):
        RNG_Key = jax.random.PRNGKey(RNG_Key)

    # Get hidden sizes
    ensemble_sizes, sizes_exc, sizes_inh = _get_hidden_sizes(sizes_hidden, nb_hidden, nb_exc_per_ensemble, EI_ratio)

    # 1. Make Membership matrices
    M_E = []
    M_I = []

    for l in range(nb_hidden):
        RNG_l, RNG_Key = jax.random.split(RNG_Key)

        M_E_l, M_I_l = make_membership_matrices(RNG_l,
                                                ensemble_sizes[l],
                                                sizes_exc[l],
                                                sizes_inh[l],
                                                overlap=overlap)
        M_E.append(M_E_l)
        M_I.append(M_I_l)

    membership_matrices = {'M_E': M_E,
                           'M_I': M_I}

    # 2. Compute W_EI weights from Membership matrices
    W_EI = [get_W_from_g_and_M(g_EI, M_E[l], M_I[l]) for l in range(nb_hidden)]

    # 3. Compute W_IE weights: analytic if no overlap, optimized otherwise
    g_IE_analytic = get_gIE_analytic(alpha, g_II, g_EI, g_EE, g_XI)

    W_IE = []
    for l in range(nb_hidden):
        W_IE_analytic = get_W_from_g_and_M(g_IE_analytic, M_I[l], M_E[l])

        if overlap == 0.0:
            W_IE.append(W_IE_analytic)
        else:
            W_IE_opt, _ = get_W_IE_optimized(
                W_EI[l].T, M_E[l], M_I[l], g_XI, alpha,
                initial_W_IE=W_IE_analytic.T,
            )
            W_IE.append(W_IE_opt)

    recurrent_weights = {'W_IE': W_IE,
                         'W_EI': W_EI}

    return membership_matrices, recurrent_weights


class ExcInhAssemblyVectorField(VectorField):
    """
    Vector field with low-rank structure in each hidden layer,
    dictated by membership matrices M_E and M_I.
    
    Each layer has:
        - a linear transformation W and bias b (if use_bias=True)
        - membership matrices M_E and M_I (fixed during training)    
    """
    
    #RNG
    RNG_Key: jax.random.PRNGKey
    
    # Architecture
    nb_hidden: int                  # number of hidden layers
    sizes_hidden: Iterable[int]     # size of each hidden layer (nb of ensembles)
    use_bias: bool
    nb_exc_per_ensemble: int        # number of excitatory neurons per ensemble
    EI_ratio: float                 # ratio of inhibitory to excitatory neurons
    overlap: float                  # overlap between ensembles (probability)
    g_EI: Any                       # E-to-I gain ('default' = 1/nb_exc_per_ensemble, or float)
    g_XI: float                     # feedforward-to-I gain
    g_EE: float                     # E-to-E gain
    g_II: float                     # I-to-I gain
    
    # Dynamics
    actE: ActivationFunction
    actI: ActivationFunction
    controller: Controller
    tauE: float
    tauI: float
    tauOut: float
    fb_to_readout: bool
    alpha: float
    
    # Initialization
    FF_init: str  
    
    # Jacobian
    inh_deriv_in_jac: bool
    
    def _get_hidden_sizes(self):
        return _get_hidden_sizes(self.sizes_hidden, self.nb_hidden, 
                                 self.nb_exc_per_ensemble, self.EI_ratio)
    
    def _get_FF_init_params(self):
        if self.FF_init == 'orthogonal':
            ff_initializer = orthogonal()
        elif self.FF_init == 'normal':
            ff_initializer = normal()
        elif self.FF_init == 'default':
            ff_initializer = lecun_normal()
        else:
            raise ValueError("Unknown FF initializer")
        
        return ff_initializer

    def setup(self):
        ensemble_sizes, sizes_exc, sizes_inh = self._get_hidden_sizes()
        ff_initializer = self._get_FF_init_params()

        # Only compute membership/recurrence during init, not during traced apply() calls
        if not self.has_variable("constants", "memberships"):
            g_EI = 1.0 / self.nb_exc_per_ensemble if self.g_EI == 'default' else float(self.g_EI)

            membership_matrices, recurrent_weights = construct_membership_and_recurrence(self.RNG_Key,
                                                                      self.nb_hidden,
                                                                      self.sizes_hidden,
                                                                      self.nb_exc_per_ensemble,
                                                                      self.EI_ratio,
                                                                      self.alpha,
                                                                      self.overlap,
                                                                      g_EI=g_EI,
                                                                      g_XI=self.g_XI,
                                                                      g_EE=self.g_EE,
                                                                      g_II=self.g_II)
        else:
            membership_matrices = None
            recurrent_weights = None

        # Membership matrices & Recurrent weights as constant variables
        self.membership_matrices = self.variable(
            "constants",  # the collection name
            "memberships",
            lambda: membership_matrices
        )

        self.recurrent_weights = self.variable(
            "constants",  # the collection name
            "recurrent_weights",
            lambda: recurrent_weights
        )
        
        self.hidden = [nn.Dense(ensemble_sizes[i], use_bias=self.use_bias,
                                kernel_init=ff_initializer,
                                bias_init=constant(0.0),
                                dtype=self.dtype,
                                )
                       for i in range(self.nb_hidden)]
        
        self.readout = nn.Dense(self.dim_output, use_bias=False, 
                                dtype=self.dtype)
        
        
    # Quick access for the membership matrices and recurrent weights
    @property
    def M_E(self):
        return self.membership_matrices.value['M_E']

    @property
    def M_I(self):
        return self.membership_matrices.value['M_I']
    
    @property
    def W_IE(self):
        return self.recurrent_weights.value['W_IE']
    
    @property
    def W_EI(self):
        return self.recurrent_weights.value['W_EI']
    
    # Quick access for recurrent weights from outside the class
    def get_recurrent_weight(self, layer_idx):
        # Now we can safely access self.recurrent_weights inside a method
        return self.recurrent_weights.value["W_IE"][layer_idx], \
               self.recurrent_weights.value["W_EI"][layer_idx]
    
    def _compute_ff_inputs(self, x, state_vf):
        """ Get feedforward inputs to the vector field """
        
        ff_inputs = []
        h = x
        for l in range(self.nb_hidden):
            input_to_ensembles = self.hidden[l](h)
            I_XE = jnp.dot(self.M_E[l], input_to_ensembles)
            I_XI = jnp.dot(self.M_I[l], input_to_ensembles)
            ff_inputs.append({'exc': I_XE, 'inh': I_XI})
            h = jnp.dot(self.actE(state_vf[l]['exc']), self.M_E[l])
            
        ff_inputs.append(self.readout(h))
        
        return ff_inputs
    
    
    def compute_ff_inputs(self, params, x, state_vf):
        return self.apply(params, x, state_vf, method=self._compute_ff_inputs)


    def __call__(self, state, x, y, fb_weights, closedloop):
        """ ODE step for closed-loop dynamics """
        
        state_vf = state['vf']
        state_ctrl = state['ctrl']

        # Compute FF Inputs
        ff_inputs = self._compute_ff_inputs(x, state_vf)
        
        # Compute Feedback control
        if closedloop:
            y_pred = self.out(state)
            ctrl, delta_state_ctrl = self.controller(y_pred, y, state_ctrl)
        else:
            ctrl = jnp.zeros(self.dim_output)
            delta_state_ctrl = self.controller.get_initial_state()

        # Update hidden dynamics     
        delta_state_vf = []
        for l in range(self.nb_hidden):
            
            u_exc = state_vf[l]['exc']
            r_exc = self.actE(u_exc)
            u_inh = state_vf[l]['inh']
            r_inh = self.actI(u_inh)
            
            if closedloop:
                fb_input = jnp.dot(ctrl, fb_weights[l])
            else:
                fb_input = jnp.zeros_like(u_inh)
                
            # Excitatory neurons
            I_XE = ff_inputs[l]['exc']
            I_IE = jnp.dot(r_inh, self.W_IE[l])
            delta_exc = 1 / self.tauE * (- u_exc + I_XE - I_IE)
            
            # Inhibitory neurons
            I_XI = ff_inputs[l]['inh']
            I_EI = jnp.dot(r_exc, self.W_EI[l])
            delta_inh = 1 / self.tauI * (- u_inh + I_XI + I_EI - fb_input)
            
            delta_state_vf.append({'exc': delta_exc,
                                   'inh': delta_inh})
            
        # Update readout layer
        # Update readout dynamics
        if self.fb_to_readout and closedloop:
            delta_readout = 1 / self.tauE * (- state_vf[-1] + ff_inputs[-1] + ctrl)
        else:
            delta_readout = 1 / self.tauE * (- state_vf[-1] + ff_inputs[-1])

        delta_state_vf.append(delta_readout)

        return {'vf': delta_state_vf,
                'ctrl': delta_state_ctrl}
        

    def get_initial_state(self, x):
        """ Get initial vf state """
        
        # Hidden layers
        ensemble_sizes, sizes_exc, sizes_inh = self._get_hidden_sizes()
        
        state_vf = []
        for i in range(self.nb_hidden):
            state_vf.append({'exc': jnp.zeros(sizes_exc[i], dtype=self.dtype),
                             'inh': jnp.zeros(sizes_inh[i], dtype=self.dtype)})
        
        # Readout
        state_vf.append(jnp.zeros(self.dim_output, dtype=self.dtype))

        # Controller
        state_ctrl = self.controller.get_initial_state()

        return {'vf': state_vf,
                'ctrl': state_ctrl}
        
    def out(self, state):
        return state['vf'][-1]
    
    def calculate_jacobian(self, vf_state):
        """ Calculates the Jacobian at the current VF state """

        hidden_layer_state = vf_state[:-1]
        exc_layer_state = [l['exc'] for l in hidden_layer_state]
        inh_layer_state = [l['inh'] for l in hidden_layer_state]
        exc_derivs = jax.tree_map(self.actE.deriv, exc_layer_state)
        inh_derivs = jax.tree_map(self.actI.deriv, inh_layer_state) 

        # Per-ensemble output of each layer  
        output_per_layer = [jnp.dot(self.actE(exc), self.M_E[l]) for l, exc in enumerate(exc_layer_state)]
        
        # Derivative factor of each layer
        def calc_deriv_factor(M, deriv_exc):
            return (M.T * deriv_exc) @ M
        
        deriv_factors = [calc_deriv_factor(self.M_E[l], exc_derivs[l]) for l in range(self.nb_hidden)]
        
        def surrogate_func(ensemble_outputs, deriv_factors):
            y = 0

            # hidden layers
            for l in range(1, self.nb_hidden):
                y = jnp.dot(self.hidden[l](y + ensemble_outputs[l-1]), deriv_factors[l])
                
            # readout layer
            y = self.readout(y + ensemble_outputs[-1])
            
            return y
        
        jac = jax.jacrev(surrogate_func)(output_per_layer, deriv_factors)

        for l in range(self.nb_hidden):

            # Project per-ensemble feedback through M_I onto the inhibitory population
            if self.inh_deriv_in_jac:
                jac[l] = jnp.dot(jac[l], self.M_I[l].T) * jnp.expand_dims(inh_derivs[l], axis=0)
            else:
                jac[l] = jnp.dot(jac[l], self.M_I[l].T)

        return jac

    def init_random_fb(self, key):
        """ Fixed random per-assembly feedback, projected through M_I.
        """
        ensemble_sizes, sizes_exc, sizes_inh = self._get_hidden_sizes()

        fb = []
        for l in range(self.nb_hidden):
            key, sub = jax.random.split(key)
            raw = jax.random.normal(sub, (self.dim_output, ensemble_sizes[l]),
                                    dtype=self.dtype)
            fb.append(jnp.dot(raw, self.M_I[l].T))

        return fb
    
    
    def calculate_gradients(self, params, x, vf_state, errors):
        """ 
        Given the parameter dict `params`,
        an input x, a VF state (sol.ys['vf']) 
        and a list of neuron-specific error signals,
        calculate the gradients wrt the weights
        """
        
        surrfunc = lambda p: self.apply(p, x, vf_state, errors, method=self._grads_surrfunc)
        return jax.grad(surrfunc)(params)
    
    
    def _grads_surrfunc(self, x, vf_state, errors):
        """ 
        Surrogate function for calculating gradients wrt the weights
        """
        
        y = 0
        h = x
        
        # Hidden
        for l in range(self.nb_hidden):
            y += jnp.sum(self.hidden[l](h) * jnp.dot(errors[l], self.M_E[l]))
            h = jnp.dot(self.actE(vf_state[l]['exc']), self.M_E[l])
        
        # Readout
        y += jnp.sum(self.readout(h) * errors[-1])    

        return y