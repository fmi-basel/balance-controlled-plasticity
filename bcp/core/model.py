# Model base class
# Julian Rossbroich
# 2025

import jax
import jax.numpy as jnp
from jax import vmap

import flax

import diffrax
from diffrax import PIDController, ConstantStepSize, SteadyStateEvent, ImplicitAdjoint
from diffrax import diffeqsolve, ODETerm, SaveAt

from typing import Callable, Tuple,Union
import logging
from functools import partial

from ..core import VectorField, ForwardVectorField

# SET UP LOGGER
logger = logging.getLogger(__name__)


class Model(flax.struct.PyTreeNode):
    """
    Model base class
    
    This is a wrapper around a Flax module implementing the vector field (or forward pass)
    and a solver for the vector field dynamics (with diffrax).
    
    IMPORTANT:
     -  Any attributes of the model object need to remain static during training 
        because the model object is passed to the (jax-compiled) training step function.
    """
    
    vf: VectorField
    stateful: bool
    dtype: jnp.dtype
    
    # Solver parameters
    dt: float 
    T: float 
    rtol: float 
    atol: float
    solver: diffrax.AbstractSolver 
    adaptive_stepsize: bool 
    early_termination: bool 
    throw_on_error: bool

    # INITIALIZATION
    # # # # # # # # # # # # # # # #

    def init(self, key: jax.random.PRNGKey, x: jnp.ndarray, y: jnp.ndarray, *args):
        """ 
        Flax model initialization from mock data (x, y) 
        (x, y) should NOT have a batch dimension, as the 
        vector field functions are v-mapped over the batch dimension
        in the model forward pass.
        """
        
        if isinstance(self.vf, ForwardVectorField):
            params = self.vf.init(key, x, method=self.vf.forward)

        else:
            state0 = self.vf.get_initial_state(x)
            params = self.vf.init(key, state0, x, None, None, False)
        
        return params

    # EASY-ACCESS OL / CL FORWARD PASSES
    # # # # # # # # # # # # # # # #

    def openloop(self, params, state0, x):
        """
        Open-loop forward pass of the model (without top-down control)
        """
        if isinstance(self.vf, ForwardVectorField):
            func = lambda a: self.vf.apply(params, a, method=self.vf.forward)
            y_pred, final_state = vmap(func)(x)
            sol = None    
        else:
            y_pred, final_state, sol = self.forward(params, state0, x, y=None, fb_weights=None, closedloop=False)    
        
        return y_pred, final_state, sol
    
    def closedloop(self, params, state0, x, y, fb_weights):
        """
        Closed-loop forward pass of the model (with top-down control)
        """
        
        return self.forward(params, state0, x, y, fb_weights, closedloop=True)    

    # VECTOR FIELD / FORWARD PASS
    # # # # # # # # # # # # # # # #
    
    def forward(self, params, state0, x, y, fb_weights, closedloop):
        """ V-mapped Forward pass of the model """

        in_axes = self.vf.get_in_axes(closedloop)
        return vmap(self._forward, in_axes=in_axes)(params, state0, x, y, fb_weights, closedloop)
        
    def _forward(self, params, state0, x, y, fb_weights, closedloop):
            """ 
            Forward pass of the model
            
            :param params: model parameters
            :param u0: initial state
            :param x: input data
            :param y: target data
            :param T: time horizon
            """
        
            def f(t, state, etc):
                return self.vf.apply(params, state, x, y, fb_weights, closedloop)

            if self.adaptive_stepsize:
                stepsize_ctrl = PIDController(rtol=self.rtol, 
                                              atol=self.atol,
                                              pcoeff=0.8, 
                                              icoeff=0.2, 
                                              dcoeff=0.1,
                                              dtmin=self.dt / 10,
                                              dtmax=100 * self.dt,
                                              factormin=0.1,
                                              factormax=5)
            else:
                stepsize_ctrl = ConstantStepSize()
                
            if self.early_termination:
                event = SteadyStateEvent(rtol=self.rtol, atol=self.atol)
            else:
                event = None
                
            term = ODETerm(f)
            sol = diffeqsolve(term, 
                            self.solver,
                            t0=0, 
                            t1=self.T,
                            dt0=self.dt, 
                            y0=state0,
                            stepsize_controller=stepsize_ctrl,
                            discrete_terminating_event = event,
                            #adjoint=ImplicitAdjoint(),
                            throw=self.throw_on_error)

            # Remove time dimension from solution
            # Hacky way to replace sol.ys because sol is immutable 
            object.__setattr__(sol, 'ys', jax.tree_map(lambda x: x.squeeze(axis=0), sol.ys))            
            
            state = sol.ys

            # Get prediction from final state
            y_pred = self.vf.out(state)
            
            return y_pred, state, sol


    # EASY-ACCESS OL / CL / COMBINED TRAJECTORIES 
    # # # # # # # # # # # # # # # #
    
    @partial(jax.jit, static_argnums=(0))
    def traj_openloop(self, params, x, state0=None):
        """ 
        Open-loop trajectory of the model (without top-down control)
        """
        if state0 is None:
            state0 = self.vf.get_initial_state_batchexp(x)
        traj, sol = self.trajectory(params, state0, x, None, None, False)    
        
        return traj, sol
    
    @partial(jax.jit, static_argnums=(0))    
    def traj_closedloop(self, params, x, y, fb_weights, state0=None):
        """ 
        Closed-loop trajectory of the model (with top-down control)
        """
        if state0 is None:
            state0 = self.vf.get_initial_state_batchexp(x)
        traj, sol = self.trajectory(params, state0, x, y, fb_weights, True)    
        
        return traj, sol
    
    def traj_combined_calcFB(self, params, x, y, trainer):
        """
        Open-loop followed by closed-loop trajectory to illustrate
        the change in steady-state. Concatenates the two `traj` objects.
        """
        traj_OL, sol_OL = self.traj_openloop(params, x)    

        # Extract final state
        final_state_OL = jax.tree_map(lambda x: x[:,-1,:], traj_OL)
        
        # Calculate feedback weights
        fb_weights = self.get_fb_weights(params, 
                                         final_state_OL)
        fb_weights = trainer.modify_fb_weights(fb_weights, (x, y))
        
        # Run closed-loop trajectory
        traj_CL, sol_CL = self.traj_closedloop(params, x, y, fb_weights, final_state_OL)    

        # Concatenate OL and CL solutions
        traj = jax.tree_map(lambda x, y: jnp.concatenate((x, y), axis=1), traj_OL, traj_CL)

        return traj
    
    # VECTOR FIELD TRAJECTORY FUNCTIONS
    # # # # # # # # # # # # # # # #
    
    def trajectory(self, params, state0, x, y, fb_weights, closedloop):
        """ V-mapped forward pass with trajectory output """
        
        in_axes = self.vf.get_in_axes(closedloop)
        return vmap(self._trajectory, in_axes=in_axes)(params, state0, x, y, fb_weights, closedloop)
    
    def _trajectory(self, params, state0, x, y, fb_weights, closedloop):
        
        def f(t, state, etc):
            return self.vf.apply(params, state, x, y, fb_weights, closedloop)
        
        stepsize_ctrl = ConstantStepSize()
        saveat = SaveAt(ts=jnp.arange(0, self.T, self.dt))
        
        term = ODETerm(f)
        sol = diffeqsolve(term, 
                          self.solver,
                          t0=0, 
                          t1=self.T, 
                          dt0=self.dt, 
                          y0=state0,
                          stepsize_controller = stepsize_ctrl,
                          saveat = saveat)
        
        traj = sol.ys
        
        return traj, sol
    
    @property
    def apply_fun(self) -> Callable:
        return NotImplemented
    
    
    # JACOBIAN CALCULATION / FB WEIGHTS
    # # # # # # # # # # # # # # # #
    
    def get_fb_weights(self, params, state):
        """ V-mapped Jacobian calculation given parameters and VF state """
        
        def _calc_jacobian(vf_state):
            return self.vf.apply(params, vf_state, method=self.vf.calculate_jacobian)
        
        return vmap(_calc_jacobian)(state['vf'])
