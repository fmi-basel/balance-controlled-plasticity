import jax
import jax.numpy as jnp

import time
import warnings
from dataclasses import dataclass

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
    get_max_gEE,
)


@dataclass(frozen=True)
class OnlineLearningMode:
    closedloop: bool = False
    update_wFF: bool = False
    update_wOUT: bool = False
    update_wEE: bool = False
    update_wIE: bool = False

    @classmethod
    def open_loop_eval(cls):
        return cls(closedloop=False)

    @classmethod
    def closed_loop_eval(cls):
        return cls(closedloop=True)

    @classmethod
    def train_full(cls, update_wEE=False):
        return cls(
            closedloop=True,
            update_wFF=True,
            update_wOUT=True,
            update_wEE=update_wEE,
            update_wIE=False,
        )

    @classmethod
    def train_readout(cls):
        return cls(
            closedloop=False,
            update_wFF=False,
            update_wOUT=True,
            update_wEE=False,
            update_wIE=False,
        )

    @classmethod
    def train_wie_only(cls):
        return cls(
            closedloop=False,
            update_wFF=False,
            update_wOUT=False,
            update_wEE=False,
            update_wIE=True,
        )


# ---------------------------------------------------------------------------
# Feedback projectors
# ---------------------------------------------------------------------------


class _GlobalFeedbackProjector:
    """Sums all controller dimensions into a single scalar broadcast to all inh. neurons."""

    def __init__(self, feedback_to_excitatory=False):
        self.feedback_to_excitatory = feedback_to_excitatory

    def __call__(self, ctrl, W_OUT):
        fb_inh = ctrl.sum()
        fb_exc = ctrl.sum() if self.feedback_to_excitatory else 0.0
        return fb_inh, fb_exc


class _RandomFeedbackProjector:
    """Fixed random feedback matrix, pre-projected onto the inhibitory population."""

    def __init__(
        self,
        W_FB_inh,
        W_FB_exc=None,
        nb_exc=0,
        only_disinhibitory=False,
    ):
        # W_FB_inh: (nb_outputs, nb_inh)
        self.W_FB_inh = W_FB_inh
        self.W_FB_exc = W_FB_exc
        self.nb_exc = nb_exc
        self.only_disinhibitory = only_disinhibitory

    def __call__(self, ctrl, W_OUT):
        fb_inh = jnp.dot(ctrl, self.W_FB_inh)
        if self.only_disinhibitory:
            fb_inh = jax.nn.relu(fb_inh)

        if self.W_FB_exc is not None:
            fb_exc = jnp.dot(ctrl, self.W_FB_exc)
        else:
            fb_exc = jnp.zeros(self.nb_exc)

        return fb_inh, fb_exc


class _StructuredFeedbackProjector:
    """Uses the learned readout W_OUT to project controller feedback (default)."""

    def __init__(
        self, M_I, M_E, only_disinhibitory=False, feedback_to_excitatory=False
    ):
        self.M_I = M_I
        self.M_E = M_E
        self.only_disinhibitory = only_disinhibitory
        self.feedback_to_excitatory = feedback_to_excitatory
        self.nb_exc = M_E.shape[0]

    def __call__(self, ctrl, W_OUT):
        W_FB_I = jnp.dot(W_OUT.T, self.M_I.T)
        fb_inh = jnp.dot(ctrl, W_FB_I)
        if self.only_disinhibitory:
            fb_inh = jax.nn.relu(fb_inh)

        if self.feedback_to_excitatory:
            W_FB_E = jnp.dot(W_OUT.T, self.M_E.T)
            fb_exc = jnp.dot(ctrl, W_FB_E)
        else:
            fb_exc = jnp.zeros(self.nb_exc)

        return fb_inh, fb_exc


# ---------------------------------------------------------------------------
# Main vector-field class
# ---------------------------------------------------------------------------


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
        tauPre,
        eta_OUT,
        eta_FF,
        eta_EE,
        alpha,
        g_EI="default",  # 'auto' means set to 1 / nb_exc_per_ens or float
        g_EE=0.0,
        g_II=0.0,
        g_XI=1.0,
        use_bias=True,
        rng_key=None,
        overlap=0.0,
        controller=None,
        global_fb=False,
        random_fb_per_ensemble=False,
        random_fb_per_neuron=False,
        only_disinhibitory_feedback=False,
        feedback_to_excitatory=False,
        eta_IE=0.0,
        compute_wIE_method=None,
        w_ie_init_mode="auto",
        w_ie_init_scale=0.0,
        w_ie_project_positive=True,
        w_ie_pruning=False,
        w_ie_pruning_thresh=0.01,
        w_ie_pruning_start_iter=0,
        clip_weights=False,
        clip_val=1.0,
        weight_decay=0.000,
        gEE_min=0.0,
        gEE_max="default",  # set by get_max_gEE if 'default'
        w_ie_log_every=1000,
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
        self.feedback_to_excitatory = feedback_to_excitatory

        # Learning
        self.use_bias = use_bias
        self.alpha = alpha
        self.eta_FF = eta_FF
        self.eta_OUT = eta_OUT
        self.eta_EE = eta_EE
        self.eta_IE = eta_IE
        self.clip_weights = clip_weights
        self.clip_val = clip_val
        self.weight_decay = weight_decay
        self.gEE_min = gEE_min
        self.gEE_max = (
            get_max_gEE(self.g_II, self.g_EI, self.g_EE, self.tauE, self.tauI) * 0.95
            if gEE_max == "default"
            else gEE_max
        )
        self.w_ie_log_every = w_ie_log_every
        self.w_ie_init_mode = (
            compute_wIE_method if compute_wIE_method is not None else w_ie_init_mode
        )
        self.w_ie_init_scale = float(max(w_ie_init_scale, 0.0))
        self.w_ie_project_positive = w_ie_project_positive
        self.w_ie_pruning = w_ie_pruning
        self.w_ie_pruning_thresh = w_ie_pruning_thresh
        self.w_ie_pruning_start_iter = int(max(w_ie_pruning_start_iter, 0))

        # Override tau_bar with min(tauE, tauI) to match the fastest network
        # mode under slow inputs.
        self.tau_bar = min(self.tauE, self.tauI)

        # Generate ensemble membership matrices
        self.M_E, self.M_I = make_membership_matrices(
            rng_key,
            nb_ensembles,
            nb_exc,
            nb_inh,
            overlap=overlap,
        )

        # Fixed recurrent weight matrices
        self.W_EI = get_W_from_g_and_M(self.g_EI, self.M_E, self.M_I)
        self.W_EE = get_W_from_g_and_M(self.g_EE, self.M_E, self.M_E)
        self.W_II = get_W_from_g_and_M(self.g_II, self.M_I, self.M_I)

        # I-to-E weights (analytic or optimized depending on overlap)
        self.W_IE = self._init_W_IE(alpha, overlap, w_ie_log_every, self.w_ie_init_mode)
        self.W_IE_fixed = self.W_IE

        # Random feedback weights (stored here for accessibility)
        key_FB, rng_key = random.split(rng_key)
        if random_fb_per_neuron:
            self.wFB = random.normal(
                key_FB, shape=(nb_outputs, nb_inh), dtype=jnp.float32
            )
        elif random_fb_per_ensemble:
            self.wFB = random.normal(
                key_FB, shape=(nb_outputs, nb_ensembles), dtype=jnp.float32
            )
        else:
            self.wFB = None

        # Build feedback projector
        self.fb_projector = self._build_fb_projector(
            global_fb,
            random_fb_per_ensemble,
            random_fb_per_neuron,
            only_disinhibitory_feedback,
            feedback_to_excitatory,
        )

    # -----------------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------------

    def _init_W_IE(self, alpha, overlap, w_ie_log_every, init_mode):
        """Initialize I-to-E weight matrix with static or trainable-friendly modes."""
        g_IE_analytic = get_gIE_analytic(
            alpha, self.g_II, self.g_EI, self.g_EE, self.g_XI
        )
        W_IE_analytic = get_W_from_g_and_M(g_IE_analytic, self.M_I, self.M_E)
        self.g_IE_analytic = g_IE_analytic
        self.W_IE_analytic = W_IE_analytic

        mode = init_mode

        if mode == "auto":
            mode = "analytic" if overlap == 0.0 else "optimized"

        if mode == "analytic":
            self.g_IE = g_IE_analytic
            return W_IE_analytic

        if mode == "optimized":
            self.g_IE = None
            W_IE, _ = get_W_IE_optimized(
                self.W_EI.T,
                self.M_E,
                self.M_I,
                self.g_XI,
                alpha,
                progress_every=w_ie_log_every,
                initial_W_IE=W_IE_analytic.T,
            )
            return W_IE

        self.g_IE = None
        if mode == "zeros":
            return jnp.zeros_like(W_IE_analytic)

        if mode == "random_scaled_uniform":
            key_wie = random.fold_in(self.rng_key, 17)
            low = float(self.w_ie_pruning_thresh)
            high = float(2.0 * self.w_ie_pruning_thresh)
            return random.uniform(
                key_wie,
                shape=W_IE_analytic.shape,
                minval=low,
                maxval=high,
                dtype=jnp.float32,
            )

        raise ValueError(f"Unsupported w_ie_init_mode: {mode}")

    def _project_W_IE(self, W_IE, phase_iter=0):
        W_IE_proj = W_IE
        if self.w_ie_project_positive:
            W_IE_proj = jax.nn.relu(W_IE_proj)
        if self.w_ie_pruning:
            pruning_active = jnp.asarray(
                phase_iter >= self.w_ie_pruning_start_iter, dtype=jnp.bool_
            )
            W_IE_pruned = jnp.where(
                W_IE_proj < self.w_ie_pruning_thresh, 0.0, W_IE_proj
            )
            W_IE_proj = jnp.where(pruning_active, W_IE_pruned, W_IE_proj)
        return W_IE_proj

    def _resolve_W_IE(self, state, phase_iter=0, W_IE_override=None):
        if W_IE_override is not None:
            return self._project_W_IE(W_IE_override, phase_iter=phase_iter)
        if "W_IE" in state:
            return self._project_W_IE(state["W_IE"], phase_iter=phase_iter)
        return self._project_W_IE(self.W_IE_fixed, phase_iter=phase_iter)

    def _build_fb_projector(
        self,
        global_fb,
        random_fb_per_ensemble,
        random_fb_per_neuron,
        only_disinhibitory,
        feedback_to_excitatory,
    ):
        """Instantiate the feedback projector matching the configured mode."""
        assert not (global_fb and random_fb_per_ensemble), (
            "Cannot have global and random feedback"
        )
        assert not (global_fb and random_fb_per_neuron), (
            "Cannot have global and random feedback"
        )
        assert not (random_fb_per_ensemble and random_fb_per_neuron), (
            "Cannot have per-ensemble and per-neuron random feedback"
        )

        if global_fb:
            return _GlobalFeedbackProjector(
                feedback_to_excitatory=feedback_to_excitatory,
            )
        elif random_fb_per_neuron:
            if feedback_to_excitatory:
                raise ValueError(
                    "random_fb_per_neuron does not support feedback_to_excitatory=True"
                )

            # wFB already has shape (nb_outputs, nb_inh)
            return _RandomFeedbackProjector(
                self.wFB,
                only_disinhibitory=only_disinhibitory,
                nb_exc=self.nb_exc,
            )
        elif random_fb_per_ensemble:
            W_FB_inh = jnp.dot(self.wFB, self.M_I.T)
            W_FB_exc = jnp.dot(self.wFB, self.M_E.T) if feedback_to_excitatory else None
            return _RandomFeedbackProjector(
                W_FB_inh,
                W_FB_exc=W_FB_exc,
                only_disinhibitory=only_disinhibitory,
                nb_exc=self.nb_exc,
            )
        else:
            return _StructuredFeedbackProjector(
                self.M_I,
                self.M_E,
                only_disinhibitory=only_disinhibitory,
                feedback_to_excitatory=feedback_to_excitatory,
            )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def membership_matrices(self):
        return self.M_E, self.M_I

    # -----------------------------------------------------------------------
    # Weight helpers
    # -----------------------------------------------------------------------

    def _clip_weights(self, *weights):
        if not self.clip_weights:
            return weights
        return tuple(w.clip(-self.clip_val, self.clip_val) for w in weights)

    @partial(jax.jit, static_argnums=(0,))
    def _get_weights(self, W_FF, W_OUT, B, g_EE_A, W_IE):
        """Project low-rank parameters onto the full neuron space.

        Recurrent weights W_EI/W_IE/W_II are fixed class attributes.
        W_EE is either fixed (eta_EE == 0) or built from the stateful assembly
        vector state["g_EE_A"] when eta_EE != 0.
        """
        W_XE = jnp.dot(W_FF, self.M_E.T)  # [data_dim, nb_exc]
        W_XI = jnp.dot(W_FF, self.M_I.T)  # [data_dim, nb_inh]
        B_E = jnp.dot(B, self.M_E.T)
        B_I = jnp.dot(B, self.M_I.T)
        W_EO = jnp.dot(self.M_E, W_OUT)  # [nb_exc, nb_outputs]

        if self.eta_EE == 0.0:
            W_EE = self.W_EE
        else:
            W_EE = self._build_W_EE_from_g_EE_A(g_EE_A)

        return W_XE, W_XI, self.W_EI, W_IE, W_EE, self.W_II, W_EO, B_E, B_I

    def convert_params_to_weights(self, W_FF, W_OUT, B, g_EE_A=None, W_IE=None):
        """Backward-compatible alias for _get_weights.

        g_EE_A is only required when eta_EE != 0. When omitted (eta_EE == 0
        case), the fixed self.W_EE class attribute is used instead.
        """
        if g_EE_A is None:
            g_EE_A = jnp.ones(self.nb_ensembles) * self.g_EE
        if W_IE is None:
            W_IE = self._project_W_IE(self.W_IE_fixed, phase_iter=0)
        return self._get_weights(W_FF, W_OUT, B, g_EE_A, W_IE)

    def _project_g_EE_A(self, g_EE_A):
        return g_EE_A.clip(self.gEE_min, self.gEE_max)

    def _bound_g_EE_A_update(self, g_EE_A, dg_EE_A):
        # state["g_EE_A"] can drift slightly outside [gEE_min, gEE_max] between
        # ODE steps. We re-clamp here so the bound check compares
        # against the effective value, not the drifted one.
        g_EE_A = self._project_g_EE_A(g_EE_A)
        dg_EE_A = jnp.nan_to_num(dg_EE_A, nan=0.0, posinf=0.0, neginf=0.0)

        # Hard bound: at lower/upper bound, block updates that push outward.
        dg_EE_A = jnp.where((g_EE_A <= self.gEE_min) & (dg_EE_A < 0.0), 0.0, dg_EE_A)
        dg_EE_A = jnp.where((g_EE_A >= self.gEE_max) & (dg_EE_A > 0.0), 0.0, dg_EE_A)

        return dg_EE_A

    def _build_W_EE_from_g_EE_A(self, g_EE_A):
        g_EE_A = self._project_g_EE_A(g_EE_A)
        return jnp.dot(self.M_E * g_EE_A[None, :], self.M_E.T)

    # -----------------------------------------------------------------------
    # Core computation — dynamics and learning
    # -----------------------------------------------------------------------

    def _compute_currents(
        self,
        state,
        x,
        W_XE,
        W_XI,
        W_EI,
        W_IE,
        W_EE,
        W_II,
        W_EO,
    ):
        """Compute all synaptic currents from state and weight matrices."""
        rE = self.actE(state["uE"])
        rI = self.actI(state["uI"])
        return {
            "rE": rE,
            "rI": rI,
            "I_XE": jnp.dot(x, W_XE),
            "I_XI": jnp.dot(x, W_XI),
            "I_IE": jnp.dot(rI, W_IE),
            "I_EI": jnp.dot(rE, W_EI),
            # These are exactly zero when g_EE / g_II = 0
            "I_EE": jnp.dot(rE, W_EE),
            "I_II": jnp.dot(rI, W_II),
            "I_EO": jnp.dot(rE, W_EO),
        }

    def _dynamics(self, state, x, currents, B_E, B_I, fb, fb_exc):
        """ODE right-hand side for all dynamic state variables."""
        uE, uI, uOut = state["uE"], state["uI"], state["uOut"]
        rE = currents["rE"]

        # Total feedforward drive to the excitatory population
        I_ff = currents["I_XE"] + B_E + currents["I_EE"]

        duE = (
            1 / self.tauE * (-uE + I_ff - currents["I_IE"] + fb_exc)
        )
        duI = (
            1
            / self.tauI
            * (-uI + currents["I_XI"] + B_I + currents["I_EI"] - currents["I_II"] - fb)
        )
        duOut = 1 / self.tauOut * (-uOut + currents["I_EO"])

        return {
            "uE": duE,
            "uI": duI,
            "uOut": duOut,
            "eligX": 1 / self.tauPre * (-state["eligX"] + x),
            "eligR": 1 / self.tauPre * (-state["eligR"] + rE),
            "I_FF_bar": 1 / self.tau_bar * (-state["I_FF_bar"] + I_ff),
        }

    def _learning(
        self,
        state,
        currents,
        out_error,
        W_FF,
        W_OUT,
        B,
        W_XE,
        B_E,
        W_IE_state,
        update_wFF,
        update_wOUT,
        update_wEE,
        update_wIE,
    ):
        """Compute weight update deltas.

        Weight decay for W_FF and B is applied in the expanded (neuron-space)
        representation — W_XE and B_E — to match the original formulation.
        """
        error_hidden = self.alpha * jax.nn.relu(state["I_FF_bar"]) - currents["I_IE"]

        if update_wFF:
            scale = self.eta_FF / self.nb_exc_per_ens
            dWXE = jnp.outer(state["eligX"], error_hidden) - self.weight_decay * W_XE
            dWFF = scale * jnp.dot(dWXE, self.M_E)

            if self.use_bias:
                dB = scale * jnp.dot(error_hidden - self.weight_decay * B_E, self.M_E)
            else:
                dB = jnp.zeros_like(B)
        else:
            dWFF = jnp.zeros_like(W_FF)
            dB = jnp.zeros_like(B)

        if update_wOUT:
            scale = self.eta_OUT / self.nb_exc_per_ens
            dWOUT = scale * jnp.dot(self.M_E.T, jnp.outer(state["eligR"], out_error))
        else:
            dWOUT = jnp.zeros_like(W_OUT)

        if update_wEE and self.eta_EE != 0.0:
            eligR_A = jnp.dot(state["eligR"], self.M_E).squeeze(0)
            err_A = jnp.dot(error_hidden, self.M_E).squeeze(0)
            g_EE_A = state["g_EE_A"]

            dg_EE_A = self.eta_EE * (eligR_A * err_A - self.weight_decay * g_EE_A)

            dg_EE_A = self._bound_g_EE_A_update(g_EE_A, dg_EE_A)
        else:
            dg_EE_A = jnp.zeros_like(state["g_EE_A"])

        if update_wIE and self.eta_IE != 0.0:
            dWIE = self.eta_IE * jnp.outer(
                currents["rI"].squeeze(0), error_hidden.squeeze(0)
            )
        else:
            dWIE = jnp.zeros_like(W_IE_state)

        return {
            "W_FF": dWFF,
            "B": dB,
            "W_OUT": dWOUT,
            "g_EE_A": dg_EE_A,
            "W_IE": dWIE,
        }

    # -----------------------------------------------------------------------
    # Controller / feedback
    # -----------------------------------------------------------------------

    def get_ctrl_and_fb(self, state, y_pred, y, closedloop, W_OUT=None):
        if closedloop:
            ctrl, delta_state_ctrl = self.controller(y_pred, y, state["ctrl"])
            W_OUT_for_fb = state["W_OUT"] if W_OUT is None else W_OUT
            fb, fb_exc = self.fb_projector(ctrl, W_OUT_for_fb)
        else:
            ctrl = jnp.zeros(self.nb_outputs)
            delta_state_ctrl = self.controller.get_initial_state_onlineVF()
            fb = jnp.zeros(self.nb_inh)
            fb_exc = jnp.zeros(self.nb_exc)

        return ctrl, fb, fb_exc, delta_state_ctrl

    # -----------------------------------------------------------------------
    # Public class interface
    # -----------------------------------------------------------------------

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
            closedloop = mode.closedloop
            update_wFF = mode.update_wFF
            update_wOUT = mode.update_wOUT
            update_wEE = mode.update_wEE
            update_wIE = mode.update_wIE

        # Evaluate input
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

        # Unpack and clip learnable parameters
        W_FF, W_OUT, B, g_EE_A = (
            state["W_FF"],
            state["W_OUT"],
            state["B"],
            state["g_EE_A"],
        )
        W_FF, B = self._clip_weights(W_FF, B)
        W_IE = self._resolve_W_IE(
            state,
            phase_iter=phase_iter,
            W_IE_override=W_IE_override,
        )

        # Project to full neuron space
        W_XE, W_XI, W_EI, W_IE, W_EE, W_II, W_EO, B_E, B_I = self._get_weights(
            W_FF, W_OUT, B, g_EE_A, W_IE
        )

        # Controller and feedback signal
        ctrl, fb, fb_exc, delta_ctrl = self.get_ctrl_and_fb(
            state, y_pred, y, closedloop, W_OUT=W_OUT
        )

        # Currents, dynamics, learning
        currents = self._compute_currents(
            state, x, W_XE, W_XI, W_EI, W_IE, W_EE, W_II, W_EO
        )
        delta_state = self._dynamics(state, x, currents, B_E, B_I, fb, fb_exc)
        delta_state.update(
            self._learning(
                state,
                currents,
                out_error,
                W_FF,
                W_OUT,
                B,
                W_XE,
                B_E,
                W_IE,
                update_wFF,
                update_wOUT,
                update_wEE,
                update_wIE,
            )
        )
        delta_state["ctrl"] = delta_ctrl
        if "W_IE" not in state:
            delta_state.pop("W_IE", None)

        return delta_state

    def call_fixed_control(self, state, t, data, fb):
        """Call to the model with a fixed (externally provided) feedback signal."""
        x = data.evaluate(t)
        fb_val = fb.evaluate(t)

        W_FF, W_OUT, B = state["W_FF"], state["W_OUT"], state["B"]
        W_FF, B = self._clip_weights(W_FF, B)
        W_IE = self._resolve_W_IE(state, phase_iter=0)

        W_XE, W_XI, W_EI, W_IE, W_EE, W_II, W_EO, B_E, B_I = self._get_weights(
            W_FF, W_OUT, B, state["g_EE_A"], W_IE
        )

        currents = self._compute_currents(
            state, x, W_XE, W_XI, W_EI, W_IE, W_EE, W_II, W_EO
        )
        delta_state = self._dynamics(
            state,
            x,
            currents,
            B_E,
            B_I,
            fb_val,
            jnp.zeros(self.nb_exc),
        )

        # No controller update, no weight updates
        delta_state["ctrl"] = self.controller.get_initial_state_onlineVF()
        delta_state["W_FF"] = jnp.zeros_like(W_FF)
        delta_state["B"] = jnp.zeros_like(B)
        delta_state["W_OUT"] = jnp.zeros_like(W_OUT)
        delta_state["g_EE_A"] = jnp.zeros_like(state["g_EE_A"])
        if "W_IE" in state:
            delta_state["W_IE"] = jnp.zeros_like(W_IE)

        return delta_state

    def out(self, state):
        return state["uOut"]

    def get_initial_state(self, rng_key=None):
        state = {}

        if rng_key is None:
            rng_key = self.rng_key

        key1, key2 = random.split(rng_key)

        w_FF_scale = 2 / self.data_dim
        w_OUT_scale = 2 / self.nb_ensembles

        # If using W_EE, we scale down the initial feedforward and output weights
        if self.g_EE > 0.0:
            w_FF_scale *= 0.2
            w_OUT_scale *= 0.2

        state["W_FF"] = (
            random.normal(key1, shape=(self.data_dim, self.nb_ensembles)) * w_FF_scale
        )
        state["W_OUT"] = (
            random.normal(key2, shape=(self.nb_ensembles, self.nb_outputs))
            * w_OUT_scale
        )
        state["B"] = jnp.zeros(shape=(1, self.nb_ensembles))

        # Assembly-space E-E gain state (diagonal of old W_EE_A representation).
        state["g_EE_A"] = jnp.ones(shape=(self.nb_ensembles,)) * self.g_EE
        state["W_IE"] = self._project_W_IE(self.W_IE, phase_iter=0)

        # Dynamics
        state["uE"] = jnp.zeros(shape=(1, self.nb_exc))
        state["uI"] = jnp.zeros(shape=(1, self.nb_inh))
        state["uOut"] = jnp.zeros(shape=(1, self.nb_outputs))

        # Postsynaptic eligibility trace
        state["I_FF_bar"] = jnp.zeros(shape=(1, self.nb_exc))

        # Presynaptic eligibility traces
        state["eligX"] = jnp.zeros(shape=(1, self.data_dim))
        state["eligR"] = jnp.zeros(shape=(1, self.nb_exc))

        # Controller
        state["ctrl"] = self.controller.get_initial_state_onlineVF()

        return state

    def project_state(self, state, phase_iter=0):
        state_proj = dict(state)
        if "W_IE" in state_proj:
            state_proj["W_IE"] = self._project_W_IE(
                state_proj["W_IE"], phase_iter=phase_iter
            )
        if "g_EE_A" in state_proj:
            state_proj["g_EE_A"] = self._project_g_EE_A(state_proj["g_EE_A"])
        return state_proj

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
        out_dict = sol.ys.copy()
        out_dict.pop("ctrl")

        out_dict["rE"] = self.actE(out_dict["uE"])
        out_dict["rI"] = self.actI(out_dict["uI"])
        out_dict = {key: val.squeeze() for key, val in out_dict.items()}

        # Align inputs / targets with recording times
        if rec_dt != dt:
            assert rec_dt > dt, "rec_dt must be larger than dt"
            diff = int(rec_dt / dt)
            inputs = inputs[::diff]
            if targets is not None:
                targets = targets[::diff]

        if targets is not None:
            y = targets
            y_pred = self.out(out_dict)
        else:
            y = jnp.zeros((inputs.shape[0], self.nb_outputs))
            y_pred = jnp.zeros((inputs.shape[0], self.nb_outputs))

        if W_IE_override is None and "W_IE" not in sol.ys:
            warnings.warn(
                "analyze_run() received trajectories without 'W_IE' and no "
                "W_IE_override; falling back to self.W_IE_fixed for current "
                "reconstruction.",
                RuntimeWarning,
                stacklevel=2,
            )

        def get_ctrl_fb_currents(x, y, y_pred, state):
            W_FF, W_OUT, B = state["W_FF"], state["W_OUT"], state["B"]
            W_FF, B = self._clip_weights(W_FF, B)
            W_IE = self._resolve_W_IE(
                state,
                phase_iter=phase_iter,
                W_IE_override=W_IE_override,
            )

            W_XE, W_XI, W_EI, W_IE, W_EE, W_II, W_EO, B_E, B_I = self._get_weights(
                W_FF, W_OUT, B, state["g_EE_A"], W_IE
            )
            ctrl, fb, _, _ = self.get_ctrl_and_fb(
                state, y_pred, y, closedloop, W_OUT=W_OUT
            )
            currents = self._compute_currents(
                state, x, W_XE, W_XI, W_EI, W_IE, W_EE, W_II, W_EO
            )

            # I_XE reported in the output dict includes the bias
            I_ff = currents["I_XE"] + B_E + currents["I_EE"]
            error_hidden = (
                self.alpha * jax.nn.relu(state["I_FF_bar"]) - currents["I_IE"]
            )

            return (
                ctrl.squeeze(),
                fb.squeeze(),
                I_ff.squeeze(),
                currents["I_IE"].squeeze(),
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
