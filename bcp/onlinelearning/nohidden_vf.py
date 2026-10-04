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


class NoHiddenOnlineLearningVF:
    """Control network without a hidden layer
    """

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

        # Readout and input-trace time constant
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
        # closedloop, update_wFF/wEE/wIE, phase_iter and W_IE_override have no
        # counterpart without a hidden layer; they keep the shared signature
        if mode is not None:
            update_wOUT = mode.update_wOUT

        x = data.evaluate(t)

        if target is not None:
            out_error = target.evaluate(t) - self.out(state)
        else:
            out_error = jnp.zeros(self.nb_outputs)

        (W,) = self._clip_weights(state["W"])

        delta_state = {
            "u": 1 / self.tau * (-state["u"] + jnp.dot(x, W)),
            "eligX": 1 / self.tau * (-state["eligX"] + x),
        }

        if update_wOUT:
            delta_state["W"] = self.eta * jnp.outer(state["eligX"], out_error)
        else:
            delta_state["W"] = jnp.zeros_like(W)

        return delta_state

    def out(self, state):
        return state["u"]

    def get_initial_state(self, rng_key=None):
        if rng_key is None:
            rng_key = self.rng_key

        key_W, _ = random.split(rng_key)
        w_scale = 2 / self.data_dim

        return {
            "W": random.normal(key_W, shape=(self.data_dim, self.nb_outputs)) * w_scale,
            "u": jnp.zeros(shape=(1, self.nb_outputs)),
            "eligX": jnp.zeros(shape=(1, self.data_dim)),
        }

    def project_state(self, state, phase_iter=0):
        return state

    def analyze_run(
        self,
        inputs,
        sol,
        dt,
        rec_dt,
        targets=None,
        closedloop=False,
        phase_iter=0,
        W_IE_override=None,
    ):
        out_dict = {key: val.squeeze() for key, val in sol.ys.items()}

        # Align inputs and targets with recording times
        if rec_dt != dt:
            assert rec_dt > dt, "rec_dt must be larger than dt"
            diff = int(rec_dt / dt)
            inputs = inputs[::diff]
            if targets is not None:
                targets = targets[::diff]

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
