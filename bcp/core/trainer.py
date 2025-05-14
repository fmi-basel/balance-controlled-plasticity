# Trainer class
# Julian Rossbroich
# 2025

import jax
import jax.numpy as jnp
from jax import vmap

import itertools

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.core.frozen_dict import FrozenDict, unfreeze, freeze    
from optax import GradientTransformation

from typing import Callable, Tuple, Union, Any
from functools import partial
from abc import ABC, abstractmethod

from tqdm import tqdm

import logging

from bcp.core import Model
from bcp.core.losses import Loss

# SET UP LOGGER
logger = logging.getLogger(__name__)


def normalize_gradients(grads, norm=1.0):
    """
    Normalize gradients in a pytree structure.
    Each leaf is normalized independently
    so that the total frob. norm of the gradients 
    AT EACH LEAF is equal to `norm`.
    
    Args:
        grads: PyTree of gradients
        norm: Gradient norm
        
    Returns:
        Normalized gradient PyTree
    """
    
    def normalize_leaf(grad):
        grad_norm = jnp.linalg.norm(grad)
        scale = jnp.where(grad_norm > 0, norm / grad_norm, 1.0)
        return grad * scale
    
    normalized_grads = jax.tree_map(normalize_leaf, grads)
    return normalized_grads
    

def clip_nn_params(params_frozendict, param_min, param_max):
    """
    Clip leaves only under 'params' key.
    
    `params_frozendict` is the entire train_state.params, 
    which might have keys like:
        {'params', 'batch_stats', ...}
    We'll only clip the 'params' subtree.
    """
    # 1) Convert from FrozenDict -> mutable dict
    params_dict = unfreeze(params_frozendict)
    
    # 2) Modify the "params" subtree in place
    params_dict["params"] = jax.tree_map(
        lambda x: jnp.clip(x, param_min, param_max),
        params_dict["params"]
    )
    
    # 3) Convert back to FrozenDict
    return params_dict

class Trainer(flax.struct.PyTreeNode):
    
    model: Model
    optimizer: GradientTransformation
    loss: Loss

    # TRAIN STATE EXTENSION
    # # # # # # # # # # # # # # # #
    # This class and these two methods should be overridden for models with additional parameters
    # that are being tracked in the TrainState.
    
    class ExtTrainState(TrainState):
        """Extended TrainState class that contains additional parameters."""
        pass
    
    def init_trainstate_params(self, params):
        """ Initializes the extra parameters of the train state. """
        return {}
    
    def update_trainstate_params(self, trainstate, vf_sol, x):
        """ Updates the extra parameters of the train state. """
        return {}
    
    
    # TRAIN STATE INITIALIZATION
    # # # # # # # # # # # # # # # #
        
    def load_train_state_from_file(self, path):
        """
        Load train state from file.
        
        Checks whether train_state has the expected structure and
        all additional parameters are present.
        """
        raise NotImplementedError
    
    def get_initial_train_state(self, dataset, rng):
        """
        Returns a TrainState object that contains the model parameters 
        (i.e. weights and biases), the optimizer state, 
        """
        
        # Split RNG
        rng_data, rng_model = jax.random.split(rng)
        
        # We can initialize with a single example because the dynamics are
        # v-mapped over the batch dimension.
        batchsize = None
        
        # Get mock data
        x, y = dataset.get_mock_data(batchsize=batchsize,
                                     rng=rng_data,
                                     flatten=self.model.vf.flatten_input)
        
        # Initialize model parameters
        params = self.model.init(rng_model, x, y)
                
        # Extended train state parameters
        additional_fields = {}
        additional_fields.update(self.init_trainstate_params(params))
        
        # Initialize train state
        train_state = self.ExtTrainState.create(apply_fn=self.model.apply_fun,
                                                params = params,
                                                tx = self.optimizer,
                                                **additional_fields)
    
        return train_state
        
    # TRAINING
    # # # # # # # # # # # # # # # #
    
    def train_epoch(self, train_state, train_data, batchsize, monitor=None, **kwargs):
        """
        Training epoch of the model.
        """
        
        # Create iterator over training data
        total_batches = len(train_data)
        train_data = iter(train_data)
        
        # TODO: Make all of this part of the monitor class.
        # For online metric tracking, set save_to_disk=False and logging=True.
        # In the end-of-batch recording, start appending to a list.
        # In the end-of-epoch recording, compute the mean (or sum) of the list and output.
        
        # Create dictionary of online metrics
        metrics = self._get_empty_metrics_dict()
    
        # get one example batch from data and reset the iterator
        batch = next(train_data)
        u0 = self.model.vf.get_initial_state_batchexp(batch[0])
        
        # Iterate over batches
        for batch in tqdm(itertools.chain([batch], train_data), total=total_batches):
            train_state, vf_sol, batch_metrics = self.train_step(train_state, batch, u0, **kwargs)
            
            # TODO: Batch recording for monitors
            if monitor is not None:
                monitor.record_batch(train_state, vf_sol, batch_metrics)

            # Record batch metrics
            for k, v in batch_metrics.items():
                metrics[k].append(v)
            
        # End-of-epoch Monitor recording
        # Saving to disk and logging should be handled by the monitor.
        if monitor is not None:
            monitor.record_epoch()

        logger.debug("Appending metrics to list.")
        # compute mean of metrics across each batch in epoch.
        metrics = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
        
        return train_state, metrics
        
    @abstractmethod
    @partial(jax.jit, static_argnums=(0))
    def train_step(self, train_state, batch, u0, *args, **kwargs):
        """
        Training step of the model.
        Performs a forward pass and computes the loss.
        
        RETURNS updated `train_state`, diffrax output `vf_sol`, `loss`
        """
        raise NotImplementedError
        
    # EVALUATION
    # # # # # # # # # # # # # # # #
    
    def eval(self, train_state, test_data, batchsize, monitor=None, **kwargs):
        """
        Evaluation epoch of the model.
        """
        
        # Create iterator over training data
        total_batches = len(test_data)
        test_data = iter(test_data)
        
        # Create list of model outputs
        y_pred_list = []
        metrics = self._get_empty_metrics_dict()
        
        # get one example batch from data and reset the iterator
        batch = next(test_data)
        u0 = self.model.vf.get_initial_state_batchexp(batch[0])

        # Iterate over batches
        for batch in tqdm(itertools.chain([batch], test_data), total=total_batches):

            y_pred, vf_sol, batch_metrics = self.eval_step(train_state, batch, u0, **kwargs)
            
            # Record y_pred
            y_pred_list.append(y_pred)
            
            # Record batch metrics
            for k, v in batch_metrics.items():
                metrics[k].append(v)
                
            #TODO: Monitor recording (eval mode)

        # Append y_pred from each batch in epoch to a single array
        y_pred_list = jnp.concatenate(y_pred_list, axis=0)

        # compute mean of metrics across each batch in epoch.
        metrics = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
        
        # TODO: End-of-epoch Monitor recording (eval mode)

        return y_pred_list, metrics
    
    @partial(jax.jit, static_argnums=(0))
    def eval_step(self, train_state, batch, u0, *args):
        """
        Test step of the model.
        Performs a forward pass and computes the loss.
        RETURNS y_pred, vf_sol, loss
        """

        y_pred, final_state, vf_sol = self.model.openloop(train_state.params, u0, batch[0])        
        logger.debug('calculating metrics')
        metrics = self.calc_metrics(y_pred, batch[1], train_state)

        return y_pred, vf_sol, metrics
    
    # MISC
    # # # # # # # # # # # # # # # #
    
    # TODO: Remove this and put into monitor class
    # METRICS
    # # # # # # # # # # # # # # # #
    
    def _compute_loss(self, y_pred, y_true):
        """
        Computes the loss across the batch.
        """
        
        return jnp.mean(vmap(self.loss)(y_pred, y_true))
    
    def _compute_accuracy(self, y_pred, y_true):
        """
        Computes the mean accuracy across the batch.
        Assumes one-hot encoding of y_true.
        """
        return jnp.mean(jnp.argmax(y_pred, axis=-1) == jnp.argmax(y_true, axis=-1)) * 100
    
    def _get_empty_metrics_dict(self):
        
        metrics = {'loss': [], 'avg_solver_steps': [], 'weight_norm': []}
        
        if self.loss.name == 'cross_entropy':
            metrics['accuracy'] = []
        
        return metrics

    def calc_metrics(self, y_pred, y_true, trainstate, sol=None):
        """
        Evaluation metrics
        """
        metrics = self._get_empty_metrics_dict()
        metrics['loss'] = self._compute_loss(y_pred, y_true)
        
        if self.loss.name == 'cross_entropy':
            metrics['accuracy'] = self._compute_accuracy(y_pred, y_true)
            
        if sol is not None:
            norms = jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), trainstate.params)
            # concatenate norms to a 1-D array
            norms = jnp.concatenate([jnp.reshape(v, (-1,)) for v in jax.tree_util.tree_leaves(norms)])
            metrics['avg_solver_steps'] = jnp.mean(sol.stats['num_steps'])
            metrics['weight_norm'] = jnp.mean(norms)
            
        else:
            metrics['avg_solver_steps'] = 0
            metrics['weight_norm'] = 0

            
        return metrics
    

class FeedbackControlTrainer(Trainer):
    """
    Abstract class for feedback control trainers.
    
    This class should be inherited by all Training methods that involve
    - a forward pass with top-down control that reaches steady-state
    - manual gradient computation given the train_state and the steady-state solution
    """
    
    # FB weight modifications
    average_fb_weights: bool = False
    clip_fb_weights: bool = False
    clip_val: float = 1.0
    norm_fb_weights: bool = False
    norm_val: float = 1.0
    clip_params: bool = False
    clip_val_params: float = 1.0
    
    # Gradient modifications
    norm_grads: bool = False
    clip_grads: bool = False
    clip_val_grads: float = 1.0
    
    # FB strength
    strong_fb: bool = True
    target_nudge: float = 0.1
    
    def modify_fb_weights(self, fb_weights, batch):
        """
        Modify the feedback weights before they are applied.
        """

        if self.average_fb_weights:
            # take mean over batch dimension
            fb_weights = [jnp.mean(w, axis=0) for w in fb_weights]
            # broadcast to original shape
            fb_weights = [jnp.broadcast_to(w, (batch[0].shape[0], *w.shape)) for w in fb_weights]

        if self.clip_fb_weights:
            fb_weights = [jnp.clip(w, -self.clip_val, self.clip_val) for w in fb_weights]
        
        if self.norm_fb_weights:
            
            def normalize_weights(weights, norm_val):
                
                # Fully connected nets
                if len(weights.shape) == 3:  # [batch, pre, post]

                    def _norm(w):
                        """
                        Normalizes a single FB weight matrix of shape [pre, post]
                        """
                        return w / jnp.linalg.norm(w, keepdims=True) * norm_val
                    
                    return vmap(_norm)(weights)
                
                # For conv nets
                elif len(weights.shape) == 5:  # [batch, pre, x, y, post]
                    # TODO: POTENTIALLY ADJUST THIS!
                    
                    norm = jnp.sqrt(jnp.sum(jnp.square(weights), axis=(2, 3), keepdims=True))
                    normalized_weights = weights / norm
                    return normalized_weights * norm_val
                    
                else:
                    raise ValueError("Unsupported tensor shape")

            fb_weights = [normalize_weights(w, self.norm_val) for w in fb_weights]            
            
        return fb_weights        
    
    def calculate_targets(self, OL_y_pred, y_true):
        """
        Calculates the target activations for the closed-loop pass.
        Can either be the true targets (if strong_fb == True) 
        or a nudge towards the true targets (if strong_fb == False).
        """
        if self.strong_fb:
            return y_true
        else:
            func = lambda x, y: self.loss.get_nudge_targets(x, y, self.target_nudge)
            return vmap(func)(OL_y_pred, y_true)

    @partial(jax.jit, static_argnums=(0))
    def train_step(self, train_state, batch, u0):
          
        # FORWARD PASS
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                
        # OL forward pass
        logger.debug('running OL forward pass')
        OL_y_pred, OL_state, OL_vf_sol = self.model.openloop(train_state.params, 
                                                             u0, 
                                                             batch[0], 
                                                             )

        # Calc feedback weights
        logger.debug('calculating feedback weights')
        fb_weights = self.model.get_fb_weights(train_state.params, 
                                               OL_state
                                               )
        fb_weights = self.modify_fb_weights(fb_weights, batch)

        # Calc targets
        y_targets = self.calculate_targets(OL_y_pred, batch[1])
                
        # CL forward pass
        logger.debug('running CL forward pass')
        CL_y_pred, CL_state, CL_vf_sol = self.model.closedloop(train_state.params, 
                                                               OL_state,
                                                               batch[0], 
                                                               y_targets, 
                                                               fb_weights
                                                               )
        
        # METRICS 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        logger.debug('computing metrics')
        metrics = self.calc_metrics(CL_y_pred, batch[1], train_state, CL_vf_sol)

        # TRAIN STATE & PARAMETER UPDATES 
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # Train state updates
        logger.debug('updating trainstate params')
        trainstate_param_updates = self.update_trainstate_params(train_state, OL_vf_sol, batch[0])
        train_state = train_state.replace(**trainstate_param_updates)

        # Parameter updates
        logger.debug('calculating gradients')
        grads = self.get_gradients(train_state, batch[0], batch[1], OL_y_pred, CL_y_pred, OL_state, CL_state)
        
        if self.norm_grads:
            grads = normalize_gradients(grads, 1.0)
            
        if self.clip_grads:
            grads = jax.tree_map(lambda x: jnp.clip(x, -self.clip_val_grads, self.clip_val_grads), grads)
        
        logger.debug('applying gradients')
        train_state = train_state.apply_gradients(grads=grads)
        
        # -- Clip the parameters (only in the "params" subtree)
        if self.clip_params:
            new_params = clip_nn_params(train_state.params, -self.clip_val_params, self.clip_val_params)
            train_state = train_state.replace(params=new_params)

        return train_state, CL_vf_sol, metrics
    
    
    def get_gradients(self, train_state, x, y, OL_y_pred, CL_y_pred, OL_state, CL_state):
        # vmap over batch (first dimension)
        in_axes = (None, 0, 0, 0, 0, 0, 0)
        grads = vmap(self._get_gradients, in_axes=in_axes)(train_state, x, y, 
                                                           OL_y_pred, CL_y_pred, OL_state, CL_state)
        
        # take average update across batch dimension
        grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), grads)
        
        return grads
        
    @abstractmethod
    def _get_gradients(self, train_state, x, y, OL_y_pred, CL_y_pred, OL_state, CL_state):
        """
        Computes gradients.
        """
        pass
        

class BPTrainer(Trainer):
    """
    Backpropagation trainer.
    
    - Optimizes the given loss function with backpropagation.
    - No additional parameters in the train_state.
    """
    
    @partial(jax.jit, static_argnums=(0))
    def train_step(self, train_state, batch, vf_state0):
        """
        Training step with backpropagation.
        """
        
        # Define function that returns loss, y_pred and vf_sol
        def loss_fn(params):
            y_pred, state, vf_sol = self.model.openloop(params, vf_state0, batch[0])
            loss = self._compute_loss(y_pred, batch[1])
            return loss, (y_pred, vf_sol)

        # GET PARAMETER UPDATES
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (y_pred, vf_sol)), grads = grad_fn(train_state.params)

        trainstate_param_updates = self.update_trainstate_params(train_state, vf_sol, batch[0])
        
        # apply parameter updates
        train_state = train_state.apply_gradients(grads=grads, **trainstate_param_updates)

        # Compute metrics
        metrics = self.calc_metrics(y_pred, batch[1], train_state, vf_sol)
        
        return train_state, vf_sol, metrics
