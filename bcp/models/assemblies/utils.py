# Utilities for E/I assembly model
# J. Rossbroich
# 2025

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import time
import optax

from sklearn.metrics import r2_score


# GENERAL UTILS
# # # # # # # # # # # # # # # #

def MSELoss(y, y_pred):
    return jnp.mean(jnp.sum((y - y_pred) ** 2, axis=1))


def R2score(y, y_pred):
    return r2_score(np.array(y), np.array(y_pred))


# MEMEBERSHIP MATRIX GENERATION
# # # # # # # # # # # # # # # #


def weight_init_hard_tiling(Npre, Npost):
    """
    generates Npre * Npost weight matrix that tiles the space with hard boundaries
    (i.e. the weight matrix is block diagonal) consisting of a total of
    min(Npre, Npost) blocks of size min(Npre, Npost)/max(Npre, Npost)
    """
    W = np.zeros((Npre, Npost))
    for i in range(min(Npre, Npost)):
        W[i :: min(Npre, Npost), i :: min(Npre, Npost)] = 1

    # re-order the rows to make the blocks contiguous
    if Npre > Npost:
        W = W[np.argsort(np.arange(Npre) % min(Npre, Npost)), :]
    else:
        W = W[:, np.argsort(np.arange(Npost) % min(Npre, Npost))]

    return jnp.array(W)


def make_membership_matrices(RNG_Key, 
                            nb_ensembles,
                            nb_exc,
                            nb_inh,
                            probability_overlap,
                            binary=False,
                            sigma_lognorm=0.5,
                            scale=True,
                            normalize=True):
    
    # Tiling component
    M_E = weight_init_hard_tiling(nb_exc, nb_ensembles)  # [nb_exc, nb_ensembles]
    M_I = weight_init_hard_tiling(nb_inh, nb_ensembles)  # [nb_inh, nb_ensembles]

    # Random component
    rngE, rngI, rng = jax.random.split(RNG_Key, 3)
    OL_ME = jax.random.bernoulli(
        rngE, p=probability_overlap, shape=(nb_exc, nb_ensembles)
    )  # [nb_exc, nb_ensembles]
    OL_MI = jax.random.bernoulli(
        rngI, p=probability_overlap, shape=(nb_inh, nb_ensembles)
    )  # [nb_inh, nb_ensembles]
    
    # Combine tiling and random components
    # Clip to ensure that memberships are binary
    M_E = (M_E + OL_ME).clip(0, 1)
    M_I = (M_I + OL_MI).clip(0, 1)
    
    # Scale membership matrices by the number of memberships per neuron
    # i.e. divide by the sum of memberships per neuron
    if scale:
        M_E = M_E / jnp.sum(M_E, axis=1, keepdims=True)
        M_I = M_I / jnp.sum(M_I, axis=1, keepdims=True)
        
    if not binary:
        rngE, rngI, rng = jax.random.split(RNG_Key, 3)
        norm_factor = jnp.exp(sigma_lognorm**2 / 2)
        
        M_E = (
            M_E
            * jax.random.lognormal(rngE, shape=(nb_exc, nb_ensembles), sigma=sigma_lognorm)
            / norm_factor
        )
        M_I = (
            M_I
            * jax.random.lognormal(rngI, shape=(nb_inh, nb_ensembles), sigma=sigma_lognorm)
            / norm_factor
        )
        
    if normalize:
        # normalize each column to a unit L2 norm
        M_E = M_E / jnp.linalg.norm(M_E, axis=0)
        M_I = M_I / jnp.linalg.norm(M_I, axis=0)

    return M_E, M_I


# OPTIMIZATION OF I-E WEIGHTS
# # # # # # # # # # # # # #

def make_optimizer(lr, steps):
    """
    Create an Adam optimizer with a cosine decay schedule.
    """
    scheduler = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=int(steps)
    )
    return optax.adam(scheduler)


def param_to_W_IE(params):
    """
    Map unconstrained parameters to strictly positive W_IE.
    Here we use ReLU to enforce nonnegativity.
    """
    return jax.nn.relu(params)

def compute_A(W_IE, M_E, M_I, W_EI, delta=1e-6):
    """
    A(W_IE) = M_I + W_EI * [I + delta*I + W_IE W_EI]^{-1} * [M_E - W_IE M_I]
    """
    N, M = W_IE.shape
    I_N = jnp.eye(N, dtype=W_IE.dtype)
    mat = I_N + delta * I_N + (W_IE @ W_EI)  # NxN
    rhs = M_E - W_IE @ M_I                  # NxK

    sol = jnp.linalg.solve(mat, rhs)        # NxK
    return M_I + W_EI @ sol


def compute_residual(W_IE, alpha, M_E, M_I, W_EI):
    """
    R = alpha * M_E - W_IE @ A(W_IE).
    R will be (N, K).
    """
    A_mat = compute_A(W_IE, M_E, M_I, W_EI)     # (M, K)
    # W_IE @ A_mat => (N, M) * (M, K) = (N, K)
    return alpha * M_E - W_IE @ A_mat


def loss_fn(params, alpha, M_E, M_I, W_EI):
    """
    Frobenius norm^2 of the residual matrix:  || R ||_F^2.
    We'll sum up the squared entries (which is the Frobenius norm squared).
    """
    W_IE = param_to_W_IE(params)
    R = compute_residual(W_IE, alpha, M_E, M_I, W_EI)  # (N, K)
    return jnp.sqrt(jnp.mean(jnp.square(R)))

def init_W_IE(W_EI, M_E, M_I, alpha):
    """
    Initialize W_IE from W_EI, M_E, M_I, and alpha.
    """
    return alpha * (1 / W_EI.T.sum(0)).mean() * W_EI.T


def compute_W_IE(
    W_EI,
    M_E,
    M_I,
    alpha,
    num_steps=10000,
    lr=1e-3,
):
    """
    Main optimization routine to find W_IE that minimizes || alpha M_E - W_IE A(W_IE) ||_F^2.
    - W_EI: (M, N)
    - M_E:  (N, K)
    - M_I:  (M, K)
    - alpha: scalar
    - num_steps, lr: optimization hyperparameters
    - key: JAX PRNGKey for reproducibility

    Returns:
      final_params, losses over iterations
    """
    N, K = M_E.shape
    M_ = M_I.shape[0]  # shape of M_I is (M, K), so M_ = M

    init_params = init_W_IE(W_EI, M_E, M_I, alpha)

    # Build the optimizer (Adam with cosine decay)
    optimizer = make_optimizer(lr, num_steps)
    opt_state = optimizer.init(init_params)

    def scan_step(carry, _):
        """One optimization step."""
        params, opt_state = carry
        loss, grads = jax.value_and_grad(loss_fn)(params, alpha, M_E, M_I, W_EI)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state), loss

    (final_params, _), loss_history = jax.lax.scan(
        scan_step,
        (init_params, opt_state),  # initial carry
        xs=None,
        length=num_steps 
    )
    final_W_IE = param_to_W_IE(final_params)

    return final_W_IE.T, loss_history