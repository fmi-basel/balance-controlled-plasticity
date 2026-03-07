# Utilities for E/I assembly model
# J. Rossbroich
# 2025

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import logging

import time
import optax

from sklearn.metrics import r2_score


logger = logging.getLogger(__name__)


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
                             overlap,
                             sigma_lognorm=0.5,
                             return_binary=False):
    # Tiling component (block-diagonal binary matrices)
    M_E = weight_init_hard_tiling(nb_exc, nb_ensembles)
    M_I = weight_init_hard_tiling(nb_inh, nb_ensembles)


    rng1, rng2, rng3, rng4 = jax.random.split(RNG_Key, 4)

    def _add_overlap(rng, B, N, A):
        """Add random memberships in the B == 0 part."""
        off_diag_mask = (B == 0)
        k = jnp.sum(B)
        n_off = N * A - k
        # Solve for p such that overlap = p*n_off / (k + p*n_off)
        p = overlap * k / (n_off * (1.0 - overlap))
        p = jnp.clip(p, 0.0, 1.0)
        OL = jax.random.bernoulli(rng, p=p, shape=(N, A))
        OL = OL * off_diag_mask  # only add where B is 0
        return (B + OL).clip(0, 1)

    B_E = _add_overlap(rng1, M_E, nb_exc, nb_ensembles)
    B_I = _add_overlap(rng2, M_I, nb_inh, nb_ensembles)

    # Row-normalize so total membership weight per neuron = 1
    B_E = B_E / jnp.sum(B_E, axis=1, keepdims=True)
    B_I = B_I / jnp.sum(B_I, axis=1, keepdims=True)

    # Lognormal weight scaling
    norm_factor = jnp.exp(sigma_lognorm**2 / 2)
    M_E = (
        B_E
        * jax.random.lognormal(rng3, shape=(nb_exc, nb_ensembles), sigma=sigma_lognorm)
        / norm_factor
    )
    M_I = (
        B_I
        * jax.random.lognormal(rng4, shape=(nb_inh, nb_ensembles), sigma=sigma_lognorm)
        / norm_factor
    )

    # Column-normalize to unit L2 norm
    M_E = M_E / jnp.linalg.norm(M_E, axis=0)
    M_I = M_I / jnp.linalg.norm(M_I, axis=0)
    
    if return_binary:
        return B_E, B_I
    
    else:
        return M_E, M_I

# Upper bound for gEE to guarantee stability
# # # # # # # # # # # # # #


def max_gEE(gII, gIE, gEI, tauE, tauI):
    """upper bound for gEE"""
    B = (gEI * gIE) / (1 + gII)
    first = 1 + B
    second = 1 + (tauE / tauI) * (1 + gII)
    return min(first, second)


# Calculating I-E Weights directly
# # # # # # # # # # # # # #
def get_gIE_analytic(alpha, gII, gEI, gEE, gXI):
    """Calculate gIE to guarantee alpha"""
    num = alpha * (1 + gII)
    denom = (1 - alpha) * gEI + gXI * (1 - (1 - alpha) * gEE)

    return jax.nn.relu(num / jnp.maximum(denom, 1e-6))


# OPTIMIZATION OF I-E WEIGHTS
# # # # # # # # # # # # # #

def make_optimizer(lr, steps):
    """
    Create an Adam optimizer with a cosine decay schedule.
    """
    scheduler = optax.warmup_cosine_decay_schedule(
    init_value=lr * 0.01,
    peak_value=lr,
    warmup_steps=int(steps * 0.05),
    decay_steps=steps,
    end_value=lr * 0.001,
    )
    #optimizer = optax.adam(scheduler)
    
    scheduler = optax.sgdr_schedule(
cosine_kwargs=[
    {"init_value": 0.01 * lr,  "peak_value": lr,        "decay_steps": steps // 3, "end_value": lr * 1e-3, "warmup_steps": int(steps * 0.05)},
    {"init_value": 0.005 * lr, "peak_value": lr * 0.1,  "decay_steps": steps // 3, "end_value": lr * 1e-5, "warmup_steps": int(steps * 0.05)},
    {"init_value": 0.001 * lr, "peak_value": lr * 0.01, "decay_steps": steps // 3, "end_value": lr * 1e-7, "warmup_steps": int(steps * 0.05)},
]
    )
    optimizer = optax.adam(scheduler)
    return optimizer


def param_to_W_IE(params):
    """
    Map unconstrained parameters to strictly positive W_IE.
    Here we use ReLU to enforce nonnegativity.
    """
    return params

@jax.jit
def compute_A(W_IE, M_E, M_I, W_EI, delta=1e-6):
    N = W_IE.shape[0]
    mat = jnp.eye(N) * (1 + delta) + W_IE @ W_EI  # symmetric positive definite
    rhs = M_E - W_IE @ M_I
    cho = jax.scipy.linalg.cho_factor(mat)
    sol = jax.scipy.linalg.cho_solve(cho, rhs)
    return M_I + W_EI @ sol

@jax.jit
def compute_residual(W_IE, alpha, M_E, M_I, W_EI):
    """
    R = alpha * M_E - W_IE @ A(W_IE).
    R will be (N, K).
    """
    A_mat = compute_A(W_IE, M_E, M_I, W_EI)     # (M, K)
    # W_IE @ A_mat => (N, M) * (M, K) = (N, K)
    return alpha * M_E - W_IE @ A_mat

@jax.jit
def loss_fn(params, alpha, M_E, M_I, W_EI):
    """
    Frobenius norm^2 of the residual matrix:  || R ||_F^2.
    We'll sum up the squared entries (which is the Frobenius norm squared).
    """
    W_IE = param_to_W_IE(params)
    R = compute_residual(W_IE, alpha, M_E, M_I, W_EI)  # (N, K)
    return jnp.mean(jnp.square(R) + jax.nn.relu(R))

def init_W_IE(W_EI, M_E, M_I, alpha):
    """
    Initialize W_IE from W_EI, M_E, M_I, and alpha.
    """
    return alpha * (1 / W_EI.T.sum(0)).mean() * W_EI.T


def get_W_IE_optimized(
    W_EI,
    M_E,
    M_I,
    gXI,
    alpha,
    num_steps=30000,
    lr=1e-3,
    progress_every=1000,
    initial_W_IE = None,
):
    """
    Main optimization routine to find W_IE that minimizes || alpha M_E - W_IE A(W_IE) ||_F^2.
    - W_EI: (M, N)
    - M_E:  (N, K)
    - M_I:  (M, K)
    - alpha: scalar
    - num_steps, lr: optimization hyperparameters
    - progress_every: report optimization progress every N steps.
    - key: JAX PRNGKey for reproducibility

    Returns:
      final_params, losses over iterations
    """
    N, K = M_E.shape
    M_ = M_I.shape[0]  # shape of M_I is (M, K), so M_ = M

    # Scale M_I by gXI to match the effective ff inh
    M_I *= gXI

    if initial_W_IE is None:
        initial_W_IE = init_W_IE(W_EI, M_E, M_I, alpha)
        
    init_params = initial_W_IE

    # Build the optimizer (Adam with cosine decay)
    optimizer = make_optimizer(lr, num_steps)
    opt_state = optimizer.init(init_params)

    logger.info(
        "Optimizing W_IE with %d steps (logging every %d steps)",
        num_steps,
        progress_every,
    )

    @jax.jit
    def train_step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, alpha, M_E, M_I, W_EI)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        new_params = jnp.maximum(new_params, 0.0)  # project onto nonneg orthant
        return new_params, new_opt_state, loss

    params = init_params
    losses = []
    for step in range(1, num_steps + 1):
        params, opt_state, loss = train_step(params, opt_state)
        losses.append(loss)

        if step == 1 or step % progress_every == 0 or step == num_steps:
            logger.info(
                "W_IE optimization step %d/%d | loss=%.6e",
                step,
                num_steps,
                float(loss),
            )

    final_params = params
    loss_history = jnp.stack(losses)

    final_W_IE = param_to_W_IE(final_params)

    return final_W_IE.T, loss_history


# Get Neuron-level params from Ensemble-level params
# # # # # # # # # # # # # #

def get_W_from_g_and_M(g, M_source, M_target):
    """
    Map from assembly-level parameters to neuron-level weights

    g: global connection strength parameter (scalar)
    M_source: [nb_source, nb_assemblies] source population memberships
    M_target: [nb_target, nb_assemblies] target population memberships

    returns W of shape [nb_source, nb_target]
    """

    W = g * M_source @ M_target.T  # [N_source, N_target]
    return W