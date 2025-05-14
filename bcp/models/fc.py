# Standard Fully Connected Vector Field

import jax
import jax.numpy as jnp
from jax import jit

import flax
import flax.linen as nn
#local
from ..core.activation import ActivationFunction
from ..core.vectorfield import ForwardVectorField
from ..core.controller import Controller
from typing import Iterable


class NoHLVectorField(ForwardVectorField):
    """
    Vector Field without any hidden layer, used as control
    """
    use_bias: bool
    
    def setup(self):
        self.w = nn.Dense(self.dim_output, use_bias=self.use_bias, 
                          dtype=self.dtype)

    def forward(self, x):
        return self.w(x), self.w(x)

    def __call__(self, u, x, y, fb_weights, closedloop):        
        du = 1 / self.tau * (- u + self.w(x)) 
        return du
    
    def get_initial_state(self, x):
        """ Get initial vf state """
        out_state = jnp.zeros(self.dim_output)
        return out_state
    
    def out(self, state):
        return state
    
    def get_in_axes(self):
        return (None, 0, 0, 0, None)


class FullyConnectedVectorField(ForwardVectorField):
    
    # Architecture
    nb_hidden: int
    sizes_hidden: Iterable[int]
    use_bias: bool

    # Dynamics
    activation: ActivationFunction
    controller: Controller
    tau: float
    fb_to_readout: bool


    def _get_hidden_sizes(self):
        # Check sizes of hidden layers. 
        # If only one size is given, repeat it nb_hidden times
        if len(self.sizes_hidden) == 1:
            sizes = [self.sizes_hidden[0]] * self.nb_hidden
        else:
            assert len(self.sizes_hidden) == self.nb_hidden, "Number of hidden layers does not match number of hidden sizes"
            sizes = self.sizes_hidden    

        return sizes        
    
    def setup(self):
        sizes = self._get_hidden_sizes()
        self.hidden = [nn.Dense(sizes[i], use_bias=self.use_bias,
                                dtype=self.dtype) 
                       for i in range(self.nb_hidden)]
        self.readout = nn.Dense(self.dim_output, use_bias=False,
                                dtype=self.dtype)
        
    def forward(self, x):
        """ 
        Classic forward pass without top-down control
        Implemented for a single input x.
        V-mapping is done in the forward pass of the model.
        """

        state_vf = []

        h = x
        for l in range(self.nb_hidden):
            preact = self.hidden[l](h)
            state_vf.append(preact)
            h = self.activation(preact)
        
        h = self.readout(h)
        state_vf.append(h)

        state_ctrl = self.controller.get_initial_state()

        state = {'vf': state_vf,
                 'ctrl': state_ctrl}

        return h, state
    
    def _compute_ff_inputs(self, x, state_vf):
        """ Get feedforward inputs to the vector field """
        
        ff_inputs = []
        h = x
        for l in range(self.nb_hidden):
            ff_inputs.append(self.hidden[l](h))
            h = self.activation(state_vf[l])
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
            delta_layer = 1 / self.tau * (- state_vf[l] + ff_inputs[l] + jnp.dot(ctrl, fb_weights[l]))
            delta_state_vf.append(delta_layer)
        
        # Update readout dynamics
        if self.fb_to_readout:
            delta_readout = 1 / self.tau * (- state_vf[-1] + ff_inputs[-1] + ctrl)
        else:
            delta_readout = 1 / self.tau * (- state_vf[-1] + ff_inputs[-1])

        delta_state_vf.append(delta_readout)

        return {'vf': delta_state_vf,
                'ctrl': delta_state_ctrl}
        
    def get_initial_state(self, x):
        """ Get initial vf state """
                
        # Hidden layers
        sizes = self._get_hidden_sizes()
        state_vf = []
        for i in range(self.nb_hidden):
            state_vf.append(jnp.zeros(sizes[i], dtype=self.dtype))
        
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
        hidden_layer_derivs = jax.tree_map(self.activation.deriv, hidden_layer_state)
        
        def surrogate_func(state, derivs):
            y = 0

            # hidden layers
            for l in range(1, self.nb_hidden):
                y = self.hidden[l]((y + state[l-1]) * derivs[l-1])
            
            # readout layer
            y = self.readout((y + state[-1]) * derivs[-1])

            return y
        
        jac = jax.jacrev(surrogate_func)(hidden_layer_state, hidden_layer_derivs)

        return jac

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
            y += jnp.sum(self.hidden[l](h) * errors[l])
            h = self.activation(vf_state[l])
        
        # Readout
        y += jnp.sum(self.readout(h) * errors[-1])    
        
        return y