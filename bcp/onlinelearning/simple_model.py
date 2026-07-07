import jax.numpy as jnp

import time

from jax import random

from ..models.assemblies.utils import (
    R2score,
    MSELoss,
)


# ---------------------------------------------------------------------------
# CONTROL VF: No hidden layer
# ---------------------------------------------------------------------------


class SimplePopModel_NoHidden:
    # Bit of a hacky and inefficient way of implementing fully linear no-hidden layer
    # network, but it allows to just swap this class in place of SimplePopModel

    def __init__(
        self,
        data_dim,
        nb_outputs,
        tau,
        eta,
        rng_key=None,
        clip_weights=False,
        clip_val=1.0,
    ):
        if rng_key is None:
            seed = int(1000 * time.time())
            rng_key = random.PRNGKey(seed)

        self.rng_key = rng_key

        self.data_dim = data_dim
        self.nb_outputs = nb_outputs

        # Network parameters
        self.tau = tau

        # Learning
        self.eta = eta
        self.clip_weights = clip_weights
        self.clip_val = clip_val

    def _clip_weights(self, *weights):
        if not self.clip_weights:
            return weights
        return tuple(weight.clip(-self.clip_val, self.clip_val) for weight in weights)

    def membership_matrices(self):
        return None, None

    def __call__(
        self,
        state,
        t,
        data,
        target=None,
        mode=None,
        closedloop=False,
        update_wFF=False,
        update_wOUT=False,
        update_wEE=False,
        update_wIE=False,
        phase_iter=0,
        W_IE_override=None,
    ):
        if mode is not None:
            update_wOUT = mode.update_wOUT

        # Evaluate data
        x = data.evaluate(t)

        # Evaluate target and compute output error
        if target is not None:
            y = target.evaluate(t)
            y_pred = self.out(state)
            out_error = y - y_pred
        else:
            y = jnp.zeros(self.nb_outputs)
            y_pred = jnp.zeros(self.nb_outputs)
            out_error = jnp.zeros(self.nb_outputs)

        # Unpack state
        u = state["u"]

        # Unpack parameters
        W = state["W"]

        (W,) = self._clip_weights(W)

        # New state
        delta_state = {}

        delta_state["u"] = 1 / self.tau * (-u + jnp.dot(x, W))

        # presynaptic eligibility traces
        delta_state["eligR"] = 1 / self.tau * (-state["eligR"] + x)

        # Update params
        if update_wOUT:
            delta_state["W"] = self.eta * jnp.outer(state["eligR"], out_error)

        else:
            delta_state["W"] = jnp.zeros_like(W)

        return delta_state

    def out(self, state):
        return state["u"]

    def get_initial_state(self):
        state = {}

        # Weights
        key1, key2 = random.split(self.rng_key)

        # Scaling factors
        w_scale = 2 / self.data_dim

        state["W"] = (
            random.normal(key1, shape=(self.data_dim, self.nb_outputs)) * w_scale
        )

        # dynamics
        state["u"] = jnp.zeros(shape=(1, self.nb_outputs))

        # learning traces
        state["eligR"] = jnp.zeros(shape=(1, self.data_dim))

        return state

    def project_state(self, state, phase_iter=0):
        return state

    def analyze_run(self, inputs, sol, dt, rec_dt, targets=None, closedloop=False):
        out_dict = sol.ys.copy()
        out_dict = {key: val.squeeze() for key, val in out_dict.items()}

        # Align inputs and targets with recording times
        if rec_dt != dt:
            assert rec_dt > dt, "rec_dt must be larger than dt"
            diff = int(rec_dt / dt)
            inputs = inputs[::diff]
            if targets is not None:
                targets = targets[::diff]

        # Calculate output error
        if targets is not None:
            y = targets
            y_pred = self.out(out_dict)
            out_dict["output_error"] = targets - y_pred
        else:
            y = jnp.zeros((inputs.shape[0], self.nb_outputs))
            y_pred = jnp.zeros((inputs.shape[0], self.nb_outputs))
            out_dict["output_error"] = jnp.zeros((inputs.shape[0], self.nb_outputs))

        out_dict["Loss"] = float(MSELoss(y, y_pred))
        out_dict["R2"] = R2score(y, y_pred)

        return out_dict
