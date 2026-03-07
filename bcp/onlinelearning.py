import jax
import jax.numpy as jnp

import time

from jax import vmap
from jax import random
from functools import partial

from .models.assemblies.utils import (
    make_membership_matrices,
    get_W_from_g_and_M,
    R2score,
    MSELoss,
    get_gIE_analytic,
    get_W_IE_optimized,
)


class ExcInhAssemblyOnlineLearningVF:
    def __init__(
        self,
        data_dim,
        nb_ensembles,
        nb_exc,
        nb_inh,
        nb_outputs,
        actE,
        actI,
        tauE,
        tauI,
        tauOut,
        tauSlow,
        tauPre,
        eta_OUT,
        eta_FF,
        alpha: float,
        g_EI="default",  # 'auto' means set to 1 / nb_exc_per_ens or float
        g_EE=0.0,
        g_II=0.0,
        g_XI=1.0,
        use_bias=True,
        rng_key=None,
        overlap: float = 0.0,
        controller=None,
        global_fb=False,
        random_fb_per_ensemble=False,
        random_fb_per_neuron=False,
        only_disinhibitory_feedback=False,
        clip_weights=False,
        clip_val=1.0,
        weight_decay=0.000,
        w_ie_log_every: int = 1000,
    ):
        if rng_key is None:
            seed = int(1000 * time.time())
            rng_key = random.PRNGKey(seed)

        self.rng_key = rng_key

        self.data_dim = data_dim
        self.nb_ensembles = nb_ensembles
        self.nb_exc = nb_exc
        self.nb_inh = nb_inh
        self.nb_outputs = nb_outputs

        self.nb_exc_per_ens = nb_exc / nb_ensembles
        self.nb_inh_per_ens = nb_inh / nb_ensembles
        self.overlap = overlap

        # Network parameters
        self.actE = actE
        self.actI = actI

        self.tauE = tauE
        self.tauI = tauI
        self.tauOut = tauOut
        self.tauSlow = tauSlow  # Not used anymore
        self.tauPre = tauPre

        # weight scale parameters
        self.g_EI = g_EI if g_EI != "default" else 1 / self.nb_exc_per_ens
        self.g_EE = g_EE
        self.g_II = g_II
        self.g_XI = g_XI

        # Controller
        self.controller = controller
        self.global_fb = global_fb
        self.random_fb_per_ensemble = random_fb_per_ensemble
        self.random_fb_per_neuron = random_fb_per_neuron
        self.only_disinhibitory_feedback = only_disinhibitory_feedback

        # Learning
        self.use_bias = use_bias
        self.alpha = alpha
        self.beta = -alpha / (alpha - 1)
        self.eta_FF = eta_FF
        self.eta_OUT = eta_OUT
        self.clip_weights = clip_weights
        self.clip_val = clip_val
        self.weight_decay = weight_decay
        self.w_ie_log_every = w_ie_log_every

        # Generate ensemble membership matrices
        self.M_E, self.M_I = make_membership_matrices(
            rng_key,
            nb_ensembles,
            nb_exc,
            nb_inh,
            overlap=overlap,
        )

        # Override tau_bar with min(tauE, tauI) to match the fastest network mode under slow inputs.
        self.tau_bar = min(self.tauE, self.tauI)

        # Make recurrent weight matrices
        # # # # # # # # # # # # # # # # #

        self.W_EI = get_W_from_g_and_M(self.g_EI, self.M_E, self.M_I)
        self.W_EE = get_W_from_g_and_M(self.g_EE, self.M_E, self.M_E)
        self.W_II = get_W_from_g_and_M(self.g_II, self.M_I, self.M_I)

        if self.overlap == 0.0:
            self.g_IE = get_gIE_analytic(
                self.alpha, self.g_II, self.g_EI, self.g_EE, self.g_XI
            )
            self.W_IE = get_W_from_g_and_M(self.g_IE, self.M_I, self.M_E)

        else:
            g_IE = get_gIE_analytic(
                self.alpha, self.g_II, self.g_EI, self.g_EE, self.g_XI
            )
            W_IE_analytic = get_W_from_g_and_M(g_IE, self.M_I, self.M_E)

            self.g_IE = None
            self.W_IE, _ = get_W_IE_optimized(
                self.W_EI.T,
                self.M_E,
                self.M_I,
                self.g_XI,
                alpha,
                progress_every=self.w_ie_log_every,
                initial_W_IE=W_IE_analytic.T,
            )

        # Random feedback weights
        assert not (self.global_fb and self.random_fb_per_ensemble), (
            "Cannot have global and random feedback"
        )
        assert not (self.global_fb and self.random_fb_per_neuron), (
            "Cannot have global and random feedback"
        )
        assert not (self.random_fb_per_ensemble and self.random_fb_per_neuron), (
            "Cannot have per-ensemble and per-neuron random feedback"
        )

        key_FB, rng_key = random.split(rng_key)

        if self.random_fb_per_neuron:
            self.wFB = random.normal(
                key_FB, shape=(nb_outputs, nb_inh), dtype=jnp.float32
            )

        elif self.random_fb_per_ensemble:
            # Here, we generate a random W_OUT (nb_ensembles, nb_outputs)
            # and project it onto the excitatory population according to the low-rank
            # structure of the network
            self.wFB = random.normal(
                key_FB, shape=(nb_outputs, nb_ensembles), dtype=jnp.float32
            )
        else:
            self.wFB = None

    @property
    def membership_matrices(self):
        return self.M_E, self.M_I

    def _clip_weights(self, *weights):
        if not self.clip_weights:
            return weights
        return tuple(weight.clip(-self.clip_val, self.clip_val) for weight in weights)

    def __call__(
        self,
        state,
        t,
        data,
        target=None,
        closedloop=False,
        update_wFF=False,
        update_wOUT=False,
    ):
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
        uE = state["uE"]
        uI = state["uI"]
        uOut = state["uOut"]

        # firing rates
        rE = self.actE(uE)
        rI = self.actI(uI)

        # Unpack parameters
        W_FF = state["W_FF"]
        W_OUT = state["W_OUT"]
        B = state["B"]

        W_FF, B = self._clip_weights(W_FF, B)

        # Get weights
        W_XE, W_XI, W_EI, W_IE, W_EO, B_E, B_I = self.convert_params_to_weights(
            W_FF, W_OUT, B
        )

        # New state
        delta_state = {}

        # Controller
        ctrl, fb, delta_state["ctrl"] = self.get_ctrl_and_fb(
            state, W_IE, W_EO, y_pred, y, closedloop, W_OUT=W_OUT
        )

        # Excitatory population
        I_XE = jnp.dot(x, W_XE)
        I_IE = jnp.dot(rI, W_IE)
        delta_state["uE"] = 1 / self.tauE * (-uE + I_XE + B_E - I_IE)

        # Inhibitory population
        I_XI = jnp.dot(x, W_XI)
        delta_state["uI"] = 1 / self.tauI * (-uI + I_XI + jnp.dot(rE, W_EI) + B_I - fb)

        # Output population
        delta_state["uOut"] = 1 / self.tauOut * (-uOut + jnp.dot(rE, W_EO))

        # presynaptic eligibility traces
        delta_state["eligX"] = 1 / self.tauPre * (-state["eligX"] + x)
        delta_state["eligX2"] = (
            1 / self.tauOut * (-state["eligX2"] + state["eligX"])
        )  # Not used in learning rule
        delta_state["eligR"] = 1 / self.tauPre * (-state["eligR"] + rE)

        # Postsynaptic traces for learning
        delta_state["I_FF_bar"] = 1 / self.tau_bar * (-state["I_FF_bar"] + I_XE + B_E)

        error_hidden = self.alpha * jax.nn.relu(state["I_FF_bar"]) - I_IE

        # Update params
        if update_wFF:
            scaling_factor = self.eta_FF / (self.nb_exc_per_ens)

            dWXE = jnp.outer(state["eligX"], error_hidden) - self.weight_decay * W_XE
            delta_state["W_FF"] = scaling_factor * (jnp.dot(dWXE, self.M_E))

            if self.use_bias:
                # Bias
                dB = error_hidden - self.weight_decay * B_E
                delta_state["B"] = scaling_factor * jnp.dot(dB, self.M_E)
            else:
                delta_state["B"] = jnp.zeros_like(B)

        else:
            delta_state["W_FF"] = jnp.zeros_like(W_FF)
            delta_state["B"] = jnp.zeros_like(B)

        if update_wOUT:
            scaling_factor = self.eta_OUT / (self.nb_exc_per_ens)
            dWEO = jnp.outer(state["eligR"], out_error)
            delta_state["W_OUT"] = scaling_factor * jnp.dot(self.M_E.T, dWEO)

        else:
            delta_state["W_OUT"] = jnp.zeros_like(W_OUT)

        return delta_state

    def call_fixed_control(self, state, t, data, fb):
        """
        Call to the model with a fixed feedback signal
        """
        # Evaluate data and feedback
        x = data.evaluate(t)
        fb = fb.evaluate(t)

        # Unpack state
        uE = state["uE"]
        uI = state["uI"]
        uOut = state["uOut"]

        # firing rates
        rE = self.actE(uE)
        rI = self.actI(uI)

        # Unpack parameters
        W_FF = state["W_FF"]
        W_OUT = state["W_OUT"]
        B = state["B"]

        W_FF, B = self._clip_weights(W_FF, B)

        # Get weights
        W_XE, W_XI, W_EI, W_IE, W_EO, B_E, B_I = self.convert_params_to_weights(
            W_FF, W_OUT, B
        )

        # New state
        delta_state = {}

        # No update to the controller because it remains unused here
        delta_state["ctrl"] = self.controller.get_initial_state_onlineVF()

        # Excitatory population
        I_XE = jnp.dot(x, W_XE)
        I_IE = jnp.dot(rI, W_IE)
        delta_state["uE"] = 1 / self.tauE * (-uE + I_XE + B_E - I_IE)

        # Inhibitory population
        I_XI = jnp.dot(x, W_XI)
        delta_state["uI"] = 1 / self.tauI * (-uI + I_XI + jnp.dot(rE, W_EI) + B_I - fb)

        # Output population
        delta_state["uOut"] = 1 / self.tauOut * (-uOut + jnp.dot(rE, W_EO))

        # presynaptic eligibility traces
        delta_state["eligX"] = 1 / self.tauPre * (-state["eligX"] + x)
        delta_state["eligX2"] = (
            1 / self.tauOut * (-state["eligX2"] + state["eligX"])
        )  # Not used in learning rule
        delta_state["eligR"] = 1 / self.tauPre * (-state["eligR"] + rE)

        # Postsynaptic traces for learning
        delta_state["I_FF_bar"] = 1 / self.tau_bar * (-state["I_FF_bar"] + I_XE + B_E)

        # NO UPDATE TO PARAMS (THIS IS JUST FOR ILLUSTRATION)
        delta_state["W_FF"] = jnp.zeros_like(W_FF)
        delta_state["B"] = jnp.zeros_like(B)
        delta_state["W_OUT"] = jnp.zeros_like(W_OUT)

        return delta_state

    def get_ctrl_and_fb(self, state, wIE, wEO, y_pred, y, closedloop, W_OUT=None):
        # Controller
        if closedloop:
            ctrl, delta_state_ctrl = self.controller(y_pred, y, state["ctrl"])

            if self.global_fb:
                fb = ctrl.sum()
            else:
                if self.random_fb_per_ensemble:
                    W_FB = jnp.dot(self.wFB, self.M_I.T)

                if self.random_fb_per_neuron:
                    W_FB = self.wFB

                else:
                    W_OUT_for_fb = state["W_OUT"] if W_OUT is None else W_OUT
                    W_FB = jnp.dot(W_OUT_for_fb.T, self.M_I.T)

                fb = jnp.dot(ctrl, W_FB)

                if self.only_disinhibitory_feedback:
                    fb = jax.nn.relu(fb)

        else:
            ctrl = jnp.zeros(self.nb_outputs)
            delta_state_ctrl = self.controller.get_initial_state_onlineVF()
            fb = jnp.zeros(self.nb_inh)

        return ctrl, fb, delta_state_ctrl

    def out(self, state):
        return state["uOut"]

    def get_initial_state(self, rng_key=None):
        state = {}

        if rng_key is None:
            rng_key = self.rng_key

        # Weights
        key1, key2 = random.split(rng_key)

        # Scaling factors
        w_FF_scale = 2 / self.data_dim
        w_OUT_scale = 2 / self.nb_ensembles

        state["W_FF"] = (
            random.normal(key1, shape=(self.data_dim, self.nb_ensembles)) * w_FF_scale
        )
        state["W_OUT"] = (
            random.normal(key2, shape=(self.nb_ensembles, self.nb_outputs))
            * w_OUT_scale
        )
        state["B"] = jnp.zeros(shape=(1, self.nb_ensembles))

        # dynamics
        state["uE"] = jnp.zeros(shape=(1, self.nb_exc))
        state["uI"] = jnp.zeros(shape=(1, self.nb_inh))
        state["uOut"] = jnp.zeros(shape=(1, self.nb_outputs))

        # learning traces postsynaptic
        state["I_FF_bar"] = jnp.zeros(shape=(1, self.nb_exc))

        # learning traces presynaptic
        state["eligX"] = jnp.zeros(shape=(1, self.data_dim))
        state["eligX2"] = jnp.zeros(shape=(1, self.data_dim))
        state["eligR"] = jnp.zeros(shape=(1, self.nb_exc))

        # controller
        state["ctrl"] = self.controller.get_initial_state_onlineVF()

        return state

    @partial(jax.jit, static_argnums=(0,))
    def convert_params_to_weights(self, W_FF, W_OUT, B):

        # Forward
        W_XE = jnp.dot(W_FF, self.M_E.T)  # [data_dim, nb_exc]
        W_XI = jnp.dot(W_FF, self.M_I.T)  # [data_dim, nb_inh]

        # Bias
        B_E = jnp.dot(B, self.M_E.T)
        B_I = jnp.dot(B, self.M_I.T)

        # Recurrent connections
        W_EI = self.W_EI
        W_IE = self.W_IE

        # Output
        W_EO = jnp.dot(self.M_E, W_OUT)  # [nb_exc, nb_outputs]

        return W_XE, W_XI, W_EI, W_IE, W_EO, B_E, B_I

    def analyze_run(self, inputs, sol, dt, rec_dt, targets=None, closedloop=False):
        out_dict = sol.ys.copy()
        out_dict.pop("ctrl")

        out_dict["rE"] = self.actE(out_dict["uE"])
        out_dict["rI"] = self.actI(out_dict["uI"])
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
            # out_dict["output_error"] = targets - y_pred
        else:
            y = jnp.zeros((inputs.shape[0], self.nb_outputs))
            y_pred = jnp.zeros((inputs.shape[0], self.nb_outputs))
            # out_dict["output_error"] = jnp.zeros((inputs.shape[0], self.nb_outputs))

        def get_ctrl_fb_currents(x, y, y_pred, state):
            W_FF = state["W_FF"]
            W_OUT = state["W_OUT"]
            B = state["B"]

            W_FF, B = self._clip_weights(W_FF, B)

            W_XE, W_XI, W_EI, W_IE, W_EO, B_E, B_I = self.convert_params_to_weights(
                W_FF, W_OUT, B
            )

            # get control
            ctrl, fb, _ = self.get_ctrl_and_fb(
                state, W_IE, W_EO, y_pred, y, closedloop, W_OUT=W_OUT
            )

            # unpack state
            uE = state["uE"]
            uI = state["uI"]

            # firing rates
            rE = self.actE(uE)
            rI = self.actI(uI)

            # Input currents
            I_XE = jnp.dot(x, W_XE) + B_E
            I_IE = jnp.dot(rI, W_IE)

            # Error at hidden layer
            error_hidden = self.alpha * jax.nn.relu(state["I_FF_bar"]) - I_IE

            return (
                ctrl.squeeze(),
                fb.squeeze(),
                I_XE.squeeze(),
                I_IE.squeeze(),
                error_hidden.squeeze(),
            )

        ctrl, fb, I_XE, I_IE, error_hidden = vmap(get_ctrl_fb_currents)(
            inputs, y, y_pred, sol.ys
        )

        out_dict["ctrl"] = ctrl
        out_dict["fb"] = fb
        out_dict["I_XE"] = I_XE
        out_dict["I_IE"] = I_IE
        out_dict["error_hidden"] = error_hidden
        out_dict["Loss"] = float(MSELoss(y, y_pred))
        out_dict["R2"] = R2score(y, y_pred)

        return out_dict


# CONTROL VF: No hidden layer
# # # # # # # # # # # # # # # # #


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
        closedloop=False,
        update_wFF=False,
        update_wOUT=False,
    ):
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
