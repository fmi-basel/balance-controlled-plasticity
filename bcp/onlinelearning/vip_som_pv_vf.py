import jax
import jax.numpy as jnp

import math
import time

from jax import vmap
from jax import random
from functools import partial

from ..models.assemblies.utils import (
    make_membership_matrices,
    get_W_from_g_and_M,
    R2score,
    MSELoss,
    get_gIE_analytic,
)


# ---------------------------------------------------------------------------
# E-PV-VIP-SOM circuit mapping
# ---------------------------------------------------------------------------


class E_PV_VIP_SOM_OnlineLearningVF:
    def __init__(
        self,
        data_dim,
        nb_ensembles,
        nb_exc,
        nb_som,
        nb_pv,
        nb_vip,
        nb_outputs,
        actE,
        actSOM,
        actPV,
        actVIP,
        tauE,
        tauSOM,
        tauPV,
        tauVIP,
        tauOut,
        tauPre,
        eta_OUT,
        eta_FF,
        alpha,
        g_ESOM="default",  # 'default' means set to 1 / nb_exc_per_ens or float
        g_EE=0.0,
        g_SOMSOM=0.0,
        g_XSOM=1.0,
        g_XPV=1.0,
        g_PVPV=1.0,
        g_EPV=1.0,
        g_SOMPV=0.14, 
        g_VIPSOM=1.0, 
        g_SOMVIP=0.0,  
        g_EVIP=0.0,
        g_base_VIP=0.0,  # scalar VIP background gain, projected through M_VIP
        control_to_vip_only=True,  # True: control -> VIP only; False: split VIP/SOM
        sparsity_to_PV=1.0,  # connection probability for X/PV/E/SOM -> PV weights
        use_bias=True,
        rng_key=None,
        controller=None,
        clip_weights=False,
        clip_val=1.0,
        weight_decay=0.000,
    ):
        if rng_key is None:
            seed = int(1000 * time.time())
            rng_key = random.PRNGKey(seed)

        self.rng_key = rng_key

        self.data_dim = data_dim
        self.nb_ensembles = nb_ensembles
        self.nb_exc = nb_exc
        self.nb_som = nb_som
        self.nb_pv = nb_pv
        self.nb_vip = nb_vip
        self.nb_outputs = nb_outputs

        self.nb_exc_per_ens = nb_exc / nb_ensembles
        self.nb_som_per_ens = nb_som / nb_ensembles

        # Network parameters
        self.actE = actE
        self.actSOM = actSOM
        self.actPV = actPV
        self.actVIP = actVIP

        self.tauE = tauE
        self.tauSOM = tauSOM
        self.tauPV = tauPV
        self.tauVIP = tauVIP
        self.tauOut = tauOut
        self.tauPre = tauPre

        # weight scale parameters
        self.g_ESOM = g_ESOM if g_ESOM != "default" else 1 / self.nb_exc_per_ens
        self.g_EE = g_EE
        self.g_SOMSOM = g_SOMSOM
        self.g_XSOM = g_XSOM
        self.g_XPV = g_XPV
        self.g_PVPV = g_PVPV
        self.g_EPV = g_EPV
        self.g_SOMPV = g_SOMPV
        self.g_VIPSOM = g_VIPSOM
        self.g_SOMVIP = g_SOMVIP
        self.g_EVIP = g_EVIP

        # Connection probability for all weights projecting onto PV (from
        # input, E, PV and SOM). 1.0 = fully connected (no sparsity).
        if not (0.0 < sparsity_to_PV <= 1.0):
            raise ValueError(
                f"sparsity_to_PV must be in (0, 1], got {sparsity_to_PV}."
            )
        self.sparsity_to_PV = sparsity_to_PV

        # VIP drive / control routing
        self.g_base_VIP = g_base_VIP
        self.control_to_vip_only = control_to_vip_only

        # SOM<->VIP mutual-inhibition loop gain. When the reciprocal SOM->VIP
        # connection is active the loop amplifies SOM activity by 1/(1 - p_loop),
        # which requires p_loop < 1 for stability.
        self.p_loop = self.g_SOMVIP * self.g_VIPSOM
        if self.g_SOMVIP > 0 and self.p_loop >= 1.0:
            raise ValueError(
                "SOM<->VIP mutual-inhibition is unstable: "
                f"g_SOMVIP*g_VIPSOM = {self.p_loop} >= 1 (set to <1 if you want this to work)."
            )
            
        # Re-scaling g_ESOM if direct E-VIP connection is present
        self.g_ESOM_eff = self.g_ESOM + self.g_EVIP * self.g_VIPSOM

        # Controller
        self.controller = controller

        # Learning
        self.use_bias = use_bias
        self.alpha = alpha
        self.eta_FF = eta_FF
        self.eta_OUT = eta_OUT
        self.clip_weights = clip_weights
        self.clip_val = clip_val
        self.weight_decay = weight_decay

        # Override tau_bar with min(tauE, tauSOM) to match the fastest network
        # mode under slow inputs (PV is not included, mirroring the base class).
        self.tau_bar = min(self.tauE, self.tauSOM)

        # Generate ensemble membership matrices (PV has no assembly membership).
        # Assemblies are always non-overlapping (overlap=0).
        self.M_E, self.M_SOM = make_membership_matrices(
            rng_key,
            nb_ensembles,
            nb_exc,
            nb_som,
            overlap=0.0,
        )

        # Fixed recurrent weight matrices
        self.W_ESOM = get_W_from_g_and_M(self.g_ESOM_eff, self.M_E, self.M_SOM)
        self.W_EE = get_W_from_g_and_M(self.g_EE, self.M_E, self.M_E)
        self.W_SOMSOM = get_W_from_g_and_M(self.g_SOMSOM, self.M_SOM, self.M_SOM)

        # SOM-to-E weights: analytic solution, always fixed and projected
        # non-negative.
        self.W_SOME = jax.nn.relu(self._init_W_SOME())

        # With a reciprocal SOM<->VIP loop, SOM activity is amplified by 1/(1 - p_loop)
        # Needs to be compensated with by rescaling the SOM->E weights by
        # (1 - p_loop):
        if self.g_SOMVIP > 0:
            self.W_SOME = self.W_SOME * (1.0 - self.p_loop)
            self.g_SOME = self.g_SOME * (1.0 - self.p_loop)

        # Fixed random PV weights. Each matrix is scaled with nb_presynaptic
        # so that the sum of incoming weights averages g_*.
        # Matrices are sparse accoridng to ``sparsity_to_PV``
        key_xpv, key_pvpv, key_epv, key_sompv, self.rng_key = random.split(rng_key, 5)

        # Input -> PV (excitatory)
        self.W_XPV = self._scaled_random_weights(
            key_xpv, self.g_XPV, data_dim, (data_dim, nb_pv), self.sparsity_to_PV
        )
        # PV -> PV lateral
        W_PVPV = self._scaled_random_weights(
            key_pvpv, self.g_PVPV, nb_pv - 1, (nb_pv, nb_pv), self.sparsity_to_PV
        )
        # remove autapses
        self.W_PVPV = W_PVPV * (1.0 - jnp.eye(nb_pv, dtype=jnp.float32))

        # E -> PV (excitatory)
        self.W_EPV = self._scaled_random_weights(
            key_epv, self.g_EPV, nb_exc, (nb_exc, nb_pv), self.sparsity_to_PV
        )

        # SOM -> PV
        self.W_SOMPV = self._scaled_random_weights(
            key_sompv, self.g_SOMPV, nb_som, (nb_som, nb_pv), self.sparsity_to_PV
        )

        # VIP population.
        _, self.M_VIP = make_membership_matrices(
            rng_key,
            nb_ensembles,
            nb_exc,
            nb_vip,
            overlap=0.0,
        )

        # VIP -> SOM inhibition 
        self.W_VIPSOM = get_W_from_g_and_M(self.g_VIPSOM, self.M_VIP, self.M_SOM)
        
        # SOM -> VIP and E -> VIP 
        self.W_SOMVIP = get_W_from_g_and_M(self.g_SOMVIP, self.M_SOM, self.M_VIP)
        self.W_EVIP = get_W_from_g_and_M(self.g_EVIP, self.M_E, self.M_VIP)

        # VIP background current
        b_VIP = self.g_base_VIP * jnp.ones(nb_ensembles)  # [nb_ensembles]
        self.I_BG_VIP = jnp.dot(self.M_VIP, b_VIP)[None, :]  # [1, nb_vip]
        
        # Compensating SOM background: cancels the tonic VIP -> SOM inhibition
        self.I_base_SOM = jnp.dot(self.I_BG_VIP, self.W_VIPSOM)  # [1, nb_som]

    # -----------------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------------

    def _scaled_random_weights(self, key, g, n_pre, shape, sparsity=1.0):
        """Sparse, non-negative random weights whose mean column sum equals ``g``
        """
        key_mask, key_mag = random.split(key)
        mask = random.bernoulli(key_mask, p=sparsity, shape=shape)
        scale = g * math.sqrt(math.pi / 2.0) / (n_pre * sparsity)
        magnitude = scale * jnp.abs(
            random.normal(key_mag, shape=shape, dtype=jnp.float32)
        )
        return mask * magnitude

    def _init_W_SOME(self):
        """Initialize SOM-to-E weight matrix.
        We always use analytic calibration here (no overlap) 
        """
        self.g_SOME = get_gIE_analytic(
            self.alpha, self.g_SOMSOM, self.g_ESOM, self.g_EE, self.g_XSOM
        )
        return get_W_from_g_and_M(self.g_SOME, self.M_SOM, self.M_E)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def membership_matrices(self):
        return self.M_E, self.M_SOM

    # -----------------------------------------------------------------------
    # Weight helpers
    # -----------------------------------------------------------------------

    def _clip_weights(self, *weights):
        if not self.clip_weights:
            return weights
        return tuple(w.clip(-self.clip_val, self.clip_val) for w in weights)

    @partial(jax.jit, static_argnums=(0,))
    def _get_weights(self, W_FF_E, W_FF_I, W_OUT, B):
        """Project low-rank (assembly-space) parameters onto the full neuron space.

        ``W_FF_E`` and ``W_FF_I`` are rectified here so the effective weights are
        strictly non-negative 
        """
        W_FF_E = jax.nn.relu(W_FF_E)
        W_FF_I = jax.nn.relu(W_FF_I)

        W_XE = jnp.dot(W_FF_E, self.M_E.T)  # [data_dim, nb_exc]
        W_PVE = jnp.dot(W_FF_I, self.M_E.T)  # [nb_pv, nb_exc]
        B_E = jnp.dot(B, self.M_E.T)
        
        W_XSOM = self.g_XSOM * jnp.dot(W_FF_E, self.M_SOM.T)  # [data_dim, nb_som]
        W_PVSOM = self.g_XSOM * jnp.dot(W_FF_I, self.M_SOM.T)  # [nb_pv, nb_som]
        B_SOM = self.g_XSOM * jnp.dot(B, self.M_SOM.T)
        W_EO = jnp.dot(self.M_E, W_OUT)  # [nb_exc, nb_outputs]

        return W_XE, W_XSOM, W_PVE, W_PVSOM, W_EO, B_E, B_SOM

    def convert_params_to_weights(self, W_FF_E, W_FF_I, W_OUT, B):
        """Backward-compatible alias for :meth:`_get_weights`."""
        return self._get_weights(W_FF_E, W_FF_I, W_OUT, B)

    # -----------------------------------------------------------------------
    # Core computation — dynamics and learning
    # -----------------------------------------------------------------------

    def _compute_currents(self, state, x, W_XE, W_XSOM, W_PVE, W_PVSOM, W_EO):
        """Compute all synaptic currents from state and weight matrices."""
        rE = self.actE(state["uE"])
        rSOM = self.actSOM(state["uSOM"])
        rPV = self.actPV(state["uPV"])
        rVIP = self.actVIP(state["uVIP"])
        return {
            "rE": rE,
            "rSOM": rSOM,
            "rPV": rPV,
            "rVIP": rVIP,
            # Feed-forward drive
            "I_XE": jnp.dot(x, W_XE),
            "I_XSOM": jnp.dot(x, W_XSOM),
            # Inhibition onto E
            "I_PVE": jnp.dot(rPV, W_PVE),
            "I_PVSOM": jnp.dot(rPV, W_PVSOM),
            "I_SOME": jnp.dot(rSOM, self.W_SOME),
            # SOM drive
            "I_ESOM": jnp.dot(rE, self.W_ESOM),
            # These are exactly zero when g_EE / g_SOMSOM = 0
            "I_EE": jnp.dot(rE, self.W_EE),
            "I_SOMSOM": jnp.dot(rSOM, self.W_SOMSOM),
            # Readout
            "I_EO": jnp.dot(rE, W_EO),
            # PV drive
            "I_XPV": jnp.dot(x, self.W_XPV),
            "I_PVPV": jnp.dot(rPV, self.W_PVPV),
            "I_EPV": jnp.dot(rE, self.W_EPV),
            "I_SOMPV": jnp.dot(rSOM, self.W_SOMPV),
            # VIP: inhibition onto SOM, and (default-zero) drive onto VIP
            "I_VIPSOM": jnp.dot(rVIP, self.W_VIPSOM),
            "I_SOMVIP": jnp.dot(rSOM, self.W_SOMVIP),
            "I_EVIP": jnp.dot(rE, self.W_EVIP),
        }

    def _dynamics(self, state, x, currents, B_E, B_SOM, fb_vip, fb_som):
        """ODE right-hand side for all dynamic state variables.
        """
        uE, uSOM, uPV, uVIP, uOut = (
            state["uE"],
            state["uSOM"],
            state["uPV"],
            state["uVIP"],
            state["uOut"],
        )
        rE = currents["rE"]
        rPV = currents["rPV"]

        # Total feed-forward drive to E: excitation minus PV feed-forward
        # inhibition (PV replaces the old negative part of the FF weights).
        I_ff = currents["I_XE"] + B_E + currents["I_EE"] - currents["I_PVE"]

        duE = 1 / self.tauE * (-uE + I_ff - currents["I_SOME"])
        duSOM = (
            1
            / self.tauSOM
            * (
                -uSOM
                + currents["I_XSOM"]
                + B_SOM
                + currents["I_ESOM"]
                - currents["I_PVSOM"]
                - currents["I_SOMSOM"]
                - currents["I_VIPSOM"]
                + self.I_base_SOM
                + fb_som
            )
        )
        duPV = (
            1
            / self.tauPV
            * (
                -uPV
                + currents["I_XPV"]
                - currents["I_PVPV"]
                + currents["I_EPV"]
                - currents["I_SOMPV"]
            )
        )
        # VIP is driven by top-down control (fb_vip) plus optional, default-zero
        # baseline / E->VIP excitation and SOM->VIP inhibition.
        duVIP = (
            1
            / self.tauVIP
            * (
                -uVIP
                + self.I_BG_VIP
                + currents["I_EVIP"]
                - currents["I_SOMVIP"]
                + fb_vip
            )
        )
        duOut = 1 / self.tauOut * (-uOut + currents["I_EO"])

        return {
            "uE": duE,
            "uSOM": duSOM,
            "uPV": duPV,
            "uVIP": duVIP,
            "uOut": duOut,
            "eligX": 1 / self.tauPre * (-state["eligX"] + x),
            "eligR": 1 / self.tauPre * (-state["eligR"] + rE),
            "eligPV": 1 / self.tauPre * (-state["eligPV"] + rPV),
            "I_FF_bar": 1 / self.tau_bar * (-state["I_FF_bar"] + I_ff),
        }

    def _learning(
        self,
        state,
        currents,
        out_error,
        W_FF_E,
        W_FF_I,
        W_OUT,
        B,
        W_XE,
        W_PVE,
        B_E,
        update_wFF,
        update_wOUT,
    ):
        """Compute weight-update deltas.
        """
        error_hidden = self.alpha * jax.nn.relu(state["I_FF_bar"]) - currents["I_SOME"]

        if update_wFF:
            scale = self.eta_FF / self.nb_exc_per_ens

            # Excitatory feed-forward weights (input -> E): standard rule.
            dWXE = jnp.outer(state["eligX"], error_hidden) - self.weight_decay * W_XE
            dWFF_E = scale * jnp.dot(dWXE, self.M_E)

            # Feed-forward inhibition (PV -> E): filtered PV activity as the
            # presynaptic trace, and a reversed sign (PV inhibition enters the
            # balance with a minus sign).
            dWXI = (
                -jnp.outer(state["eligPV"], error_hidden) - self.weight_decay * W_PVE
            )
            dWFF_I = scale * jnp.dot(dWXI, self.M_E)

            if self.use_bias:
                dB = scale * jnp.dot(error_hidden - self.weight_decay * B_E, self.M_E)
            else:
                dB = jnp.zeros_like(B)
        else:
            dWFF_E = jnp.zeros_like(W_FF_E)
            dWFF_I = jnp.zeros_like(W_FF_I)
            dB = jnp.zeros_like(B)

        if update_wOUT:
            scale = self.eta_OUT / self.nb_exc_per_ens
            dWOUT = scale * jnp.dot(self.M_E.T, jnp.outer(state["eligR"], out_error))
        else:
            dWOUT = jnp.zeros_like(W_OUT)

        return {
            "W_FF_E": dWFF_E,
            "W_FF_I": dWFF_I,
            "B": dB,
            "W_OUT": dWOUT,
        }

    # -----------------------------------------------------------------------
    # Controller / feedback
    # -----------------------------------------------------------------------

    def get_ctrl_and_fb(self, state, y_pred, y, closedloop, W_OUT=None):
        """Structured controller feedback projected onto VIP and/or SOM.

        Returns ``(ctrl, fb_vip, fb_som, delta_state_ctrl)`` where ``fb_vip`` is
        added to the VIP and ``fb_som`` is added to
        the SOM membrane.
        
        Usually only VIP received FB.
        There is an option to route dis-inhibitory feedback to VIP and inhibitory feedback to SOM
        (Not used)
        """
        if closedloop:
            ctrl, delta_state_ctrl = self.controller(y_pred, y, state["ctrl"])
            W_OUT_for_fb = state["W_OUT"] if W_OUT is None else W_OUT
            W_FB_VIP = jnp.dot(W_OUT_for_fb.T, self.M_VIP.T)  # [nb_outputs, nb_vip]
            W_FB_SOM = jnp.dot(W_OUT_for_fb.T, self.M_SOM.T)  # [nb_outputs, nb_som]
            f_vip = jnp.dot(ctrl, W_FB_VIP)
            f_som = jnp.dot(ctrl, W_FB_SOM)
            if self.control_to_vip_only:
                fb_vip = f_vip
                fb_som = jnp.zeros(self.nb_som)
            else:
                fb_vip = jax.nn.relu(f_vip)  # positive part -> VIP excitation
                fb_som = jax.nn.relu(-f_som)  # negative part -> SOM
        else:
            ctrl = jnp.zeros(self.nb_outputs)
            delta_state_ctrl = self.controller.get_initial_state_onlineVF()
            fb_vip = jnp.zeros(self.nb_vip)
            fb_som = jnp.zeros(self.nb_som)

        return ctrl, fb_vip, fb_som, delta_state_ctrl

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
        # update_wEE / update_wIE / phase_iter / W_IE_override are accepted for
        # interface compatibility but ignored: W_SOME and W_EE are always fixed.
        if mode is not None:
            closedloop = mode.closedloop
            update_wFF = mode.update_wFF
            update_wOUT = mode.update_wOUT

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

        # Unpack learnable parameters. Feed-forward weights are rectified
        # (strictly non-negative) and optionally clipped.
        W_FF_E, W_FF_I, W_OUT, B = (
            state["W_FF_E"],
            state["W_FF_I"],
            state["W_OUT"],
            state["B"],
        )
        W_FF_E = jax.nn.relu(W_FF_E)
        W_FF_I = jax.nn.relu(W_FF_I)
        W_FF_E, W_FF_I, B = self._clip_weights(W_FF_E, W_FF_I, B)

        # Project to full neuron space
        W_XE, W_XSOM, W_PVE, W_PVSOM, W_EO, B_E, B_SOM = self._get_weights(
            W_FF_E, W_FF_I, W_OUT, B
        )

        # Controller and feedback signals (onto VIP and/or SOM)
        ctrl, fb_vip, fb_som, delta_ctrl = self.get_ctrl_and_fb(
            state, y_pred, y, closedloop, W_OUT=W_OUT
        )

        # Currents, dynamics, learning
        currents = self._compute_currents(state, x, W_XE, W_XSOM, W_PVE, W_PVSOM, W_EO)
        delta_state = self._dynamics(state, x, currents, B_E, B_SOM, fb_vip, fb_som)
        delta_state.update(
            self._learning(
                state,
                currents,
                out_error,
                W_FF_E,
                W_FF_I,
                W_OUT,
                B,
                W_XE,
                W_PVE,
                B_E,
                update_wFF,
                update_wOUT,
            )
        )
        delta_state["ctrl"] = delta_ctrl

        return delta_state

    def call_fixed_control(self, state, t, data, fb):
        """Call the model with a fixed (externally provided) feedback signal.
        """
        x = data.evaluate(t)
        fb_val = fb.evaluate(t)

        W_FF_E = jax.nn.relu(state["W_FF_E"])
        W_FF_I = jax.nn.relu(state["W_FF_I"])
        W_OUT, B = state["W_OUT"], state["B"]
        W_FF_E, W_FF_I, B = self._clip_weights(W_FF_E, W_FF_I, B)

        W_XE, W_XSOM, W_PVE, W_PVSOM, W_EO, B_E, B_SOM = self._get_weights(
            W_FF_E, W_FF_I, W_OUT, B
        )

        currents = self._compute_currents(state, x, W_XE, W_XSOM, W_PVE, W_PVSOM, W_EO)
        delta_state = self._dynamics(
            state, x, currents, B_E, B_SOM, fb_val, jnp.zeros(self.nb_som)
        )

        # No controller update, no weight updates
        delta_state["ctrl"] = self.controller.get_initial_state_onlineVF()
        delta_state["W_FF_E"] = jnp.zeros_like(W_FF_E)
        delta_state["W_FF_I"] = jnp.zeros_like(W_FF_I)
        delta_state["W_OUT"] = jnp.zeros_like(W_OUT)
        delta_state["B"] = jnp.zeros_like(B)

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

        # If using W_EE, scale down the initial feed-forward and output weights.
        if self.g_EE > 0.0:
            w_FF_scale *= 0.2
            w_OUT_scale *= 0.2

        # Excitatory feed-forward weights (input -> E): non-negative init.
        state["W_FF_E"] = jax.nn.relu(
            random.normal(key1, shape=(self.data_dim, self.nb_ensembles)) * w_FF_scale
        )
        # Feed-forward inhibition weights (PV -> E): start at zero.
        state["W_FF_I"] = jnp.zeros(shape=(self.nb_pv, self.nb_ensembles))

        state["W_OUT"] = (
            random.normal(key2, shape=(self.nb_ensembles, self.nb_outputs))
            * w_OUT_scale
        )
        state["B"] = jnp.zeros(shape=(1, self.nb_ensembles))

        # Dynamics
        state["uE"] = jnp.zeros(shape=(1, self.nb_exc))
        state["uSOM"] = jnp.zeros(shape=(1, self.nb_som))
        state["uPV"] = jnp.zeros(shape=(1, self.nb_pv))
        state["uVIP"] = jnp.zeros(shape=(1, self.nb_vip))
        state["uOut"] = jnp.zeros(shape=(1, self.nb_outputs))

        # Postsynaptic feed-forward trace (balance target)
        state["I_FF_bar"] = jnp.zeros(shape=(1, self.nb_exc))

        # Presynaptic eligibility traces
        state["eligX"] = jnp.zeros(shape=(1, self.data_dim))
        state["eligR"] = jnp.zeros(shape=(1, self.nb_exc))
        state["eligPV"] = jnp.zeros(shape=(1, self.nb_pv))

        # Controller
        state["ctrl"] = self.controller.get_initial_state_onlineVF()

        return state

    def project_state(self, state, phase_iter=0):
        state_proj = dict(state)
        if "W_FF_E" in state_proj:
            state_proj["W_FF_E"] = jax.nn.relu(state_proj["W_FF_E"])
        if "W_FF_I" in state_proj:
            state_proj["W_FF_I"] = jax.nn.relu(state_proj["W_FF_I"])
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
        out_dict["rSOM"] = self.actSOM(out_dict["uSOM"])
        out_dict["rPV"] = self.actPV(out_dict["uPV"])
        out_dict["rVIP"] = self.actVIP(out_dict["uVIP"])
        # Backward-compatible alias for recording code that expects "rI".
        out_dict["rI"] = out_dict["rSOM"]
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

        def get_ctrl_fb_currents(x, y, y_pred, state):
            W_FF_E = jax.nn.relu(state["W_FF_E"])
            W_FF_I = jax.nn.relu(state["W_FF_I"])
            W_OUT, B = state["W_OUT"], state["B"]
            W_FF_E, W_FF_I, B = self._clip_weights(W_FF_E, W_FF_I, B)

            W_XE, W_XSOM, W_PVE, W_PVSOM, W_EO, B_E, B_SOM = self._get_weights(
                W_FF_E, W_FF_I, W_OUT, B
            )
            ctrl, fb_vip, fb_som, _ = self.get_ctrl_and_fb(
                state, y_pred, y, closedloop, W_OUT=W_OUT
            )
            currents = self._compute_currents(state, x, W_XE, W_XSOM, W_PVE, W_PVSOM, W_EO)

            # I_XE reported in the output dict includes the bias and net FF.
            I_ff = currents["I_XE"] + B_E + currents["I_EE"] - currents["I_PVE"]
            error_hidden = (
                self.alpha * jax.nn.relu(state["I_FF_bar"]) - currents["I_SOME"]
            )

            return (
                ctrl.squeeze(),
                fb_vip.squeeze(),
                fb_som.squeeze(),
                I_ff.squeeze(),
                currents["I_SOME"].squeeze(),
                currents["I_PVE"].squeeze(),
                currents["I_VIPSOM"].squeeze(),
                error_hidden.squeeze(),
            )

        (
            ctrl,
            fb_vip,
            fb_som,
            I_XE,
            I_SOME,
            I_PVE,
            I_VIPSOM,
            error_hidden,
        ) = vmap(get_ctrl_fb_currents)(inputs, y, y_pred, sol.ys)

        out_dict["ctrl"] = ctrl
        out_dict["fb_vip"] = fb_vip
        out_dict["fb_som"] = fb_som
        out_dict["I_XE"] = I_XE
        out_dict["I_SOME"] = I_SOME
        out_dict["I_PVE"] = I_PVE
        out_dict["I_VIPSOM"] = I_VIPSOM
        # Backward-compatible alias for recording code that expects "I_IE".
        out_dict["I_IE"] = I_SOME
        out_dict["error_hidden"] = error_hidden
        out_dict["Loss"] = float(MSELoss(y, y_pred))
        out_dict["R2"] = R2score(y, y_pred)

        return out_dict
