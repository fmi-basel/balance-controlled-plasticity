# Learned-feedback vector field
# Julian Rossbroich

import jax
import jax.numpy as jnp

from typing import Any

from bcp.models.assemblies.vf import ExcInhAssemblyVectorField


class LearnedFeedbackAssemblyVectorField(ExcInhAssemblyVectorField):
    """Assembly vector field with a noise-driven rule for learning the feedback Q.
    """

    # OU noise (scalar or per-layer tuple)
    sigma: Any = 0.005        # amplitude injected into the interneurons
    tau_eps: Any = 0.02       # OU correlation time
    tau_filt: float = 1.0

    beta: float = 0.01                # Q leak (if not trainer.norm_fb_weights)
    signal_source: str = "membrane"   # 'membrane' = u_I - LP(u_I); 'compartment' = -Qc+noise
    t_settle: float = 0.0             # gate source accumulation to t > t_settle

    def _sigma_per_layer(self, sigma=None):
        return self._as_per_layer(self.sigma if sigma is None else sigma)

    def _tau_eps_per_layer(self, tau_eps=None):
        return self._as_per_layer(self.tau_eps if tau_eps is None else tau_eps)

    def _as_per_layer(self, val):
        """Broadcast a scalar to one value per layer"""
        L = self.nb_hidden
        if isinstance(val, (int, float)):
            return tuple(float(val) for _ in range(L))
        if isinstance(val, (list, tuple)):
            assert len(val) == L, f"per-layer value length {len(val)} != nb_hidden {L}"
            return tuple(val)
        arr = jnp.asarray(val)
        if arr.ndim == 0:
            return tuple(arr for _ in range(L))
        assert arr.shape[0] == L, f"per-layer value length {arr.shape[0]} != nb_hidden {L}"
        return tuple(arr[l] for l in range(L))

    def layer_scaling(self, tau_eps=None):
        """`s_l = (1 + tau_v/tau_eps)^(L-1-l)` (see Meulemans et al. 2022)
        """
        L = self.nb_hidden
        tau_v = self.tauE + self.tauI
        tau_eps = self._tau_eps_per_layer(tau_eps)
        return tuple((1.0 + tau_v / tau_eps[l]) ** (L - 1 - l) for l in range(L))

    # ------------------------------------------------------------------ #
    # Feedback-weight helper functions
    # ------------------------------------------------------------------ #
    def fb_from_Q(self, Q):
        """project fb weights through inhibitory memberships.
        """
        return [jnp.dot(Q[l], self.M_I[l].T) for l in range(self.nb_hidden)]

    # ------------------------------------------------------------------ #
    # Noisy dynamics for feedback (Q) learning
    # ------------------------------------------------------------------ #
    def augmented_initial_state(self, x, y, ol_state, accumulate_ff=False,
                                controller_overrides=None):
        """Start the noisy trajectory at the open-loop equilibrium and add filter states.
        """
        ensemble_sizes, _, _ = self._get_hidden_sizes()

        state = {"vf": ol_state["vf"], "ctrl": ol_state["ctrl"]}

        # control at the open-loop state
        controller_overrides = {} if controller_overrides is None else controller_overrides
        ctrl0, _ = self.controller(
            self.out(ol_state), y, ol_state["ctrl"], **controller_overrides)
        state["c_lp"] = ctrl0
        state["u_inh_lp"] = [ol_state["vf"][l]["inh"] for l in range(self.nb_hidden)]
        state["Qsrc"] = [
            jnp.zeros((self.dim_output, ensemble_sizes[l]), dtype=self.dtype)
            for l in range(self.nb_hidden)
        ]

        if accumulate_ff:
            # presynaptic input to each hidden layer (l=0 -> x), plus readout presyn
            in_dims = [x.shape[-1]] + [ensemble_sizes[l] for l in range(self.nb_hidden - 1)]
            h_lp = [x]
            for l in range(self.nb_hidden - 1):
                h_lp.append(jnp.dot(self.actE(ol_state["vf"][l]["exc"]), self.M_E[l]))
            h_lp.append(jnp.dot(
                self.actE(ol_state["vf"][self.nb_hidden - 1]["exc"]),
                self.M_E[self.nb_hidden - 1]))          # readout presyn
            state["h_lp"] = h_lp
            state["gW"] = [
                {"kernel": jnp.zeros((in_dims[l], ensemble_sizes[l]), dtype=self.dtype),
                 "bias": jnp.zeros((ensemble_sizes[l],), dtype=self.dtype)}
                for l in range(self.nb_hidden)
            ]
            state["gW_read"] = jnp.zeros((ensemble_sizes[-1], self.dim_output), dtype=self.dtype)

        return state

    def noisy_step(self, state, t, x, y, Q, eps_t, accumulate_ff=False,
                   use_fr_error=True, assembly_current=0.0,
                   sigma=None, tau_filt=None, controller_overrides=None):
        """ODE for the noisy feedback-learning trajectory (single example).
        """
        state_vf = state["vf"]
        state_ctrl = state["ctrl"]
        c_lp = state["c_lp"]
        u_inh_lp = state["u_inh_lp"]

        sig = self._sigma_per_layer(sigma)
        inv_tau_filt = 1.0 / (self.tau_filt if tau_filt is None else tau_filt)
        gate = jnp.asarray(t > self.t_settle, self.dtype)

        ff_inputs = self._compute_ff_inputs(x, state_vf)
        y_pred = self.out(state)
        controller_overrides = {} if controller_overrides is None else controller_overrides
        ctrl, delta_state_ctrl = self.controller(
            y_pred, y, state_ctrl, **controller_overrides)

        teach = ctrl - c_lp

        delta_state_vf = []
        delta_u_inh_lp = []
        delta_Qsrc = []

        if accumulate_ff:
            beta_bal = -self.alpha / (self.alpha - 1.0)
            h_lp = state["h_lp"]
            delta_h_lp = []
            delta_gW = []
            h_pre = x

        for l in range(self.nb_hidden):
            u_exc = state_vf[l]["exc"]
            r_exc = self.actE(u_exc)
            u_inh = state_vf[l]["inh"]
            r_inh = self.actI(u_inh)

            fb_drive = jnp.dot(ctrl, jnp.dot(Q[l], self.M_I[l].T))   # [sizes_inh]
            noise_l = sig[l] * jnp.dot(self.M_I[l], eps_t[l])        # [sizes_inh]
            current_l = assembly_current * jnp.ones(
                self.M_E[l].shape[1], dtype=self.dtype)
            exc_current = jnp.dot(self.M_E[l], current_l)
            inh_current = jnp.dot(self.M_I[l], current_l)

            I_XE = ff_inputs[l]["exc"]
            I_IE = jnp.dot(r_inh, self.W_IE[l])
            delta_exc = 1 / self.tauE * (-u_exc + I_XE - I_IE + exc_current)

            I_XI = ff_inputs[l]["inh"]
            I_EI = jnp.dot(r_exc, self.W_EI[l])
            delta_inh = 1 / self.tauI * (
                -u_inh + I_XI + I_EI - fb_drive + noise_l + inh_current)
            delta_state_vf.append({"exc": delta_exc, "inh": delta_inh})

            # Postsynaptic factor
            if self.signal_source == "compartment":
                post_l = noise_l - fb_drive
            else:  # 'membrane'
                post_l = u_inh - u_inh_lp[l]
            delta_u_inh_lp.append(inv_tau_filt * (-u_inh_lp[l] + u_inh))

            # source: outer(teach, post) in interneuron space, averaged across the
            # assembly by projecting through M_I
            src_inh = jnp.outer(teach, post_l)                       # [dim_output, sizes_inh]
            delta_Qsrc.append(gate * jnp.dot(src_inh, self.M_I[l]))  # [dim_output, nb_ens]

            if accumulate_ff:
                expected = beta_bal * r_exc if use_fr_error else jax.nn.relu(self.alpha * I_XE)
                eproj = jnp.dot(I_IE - expected, self.M_E[l])        # [nb_ensembles]
                delta_h_lp.append(inv_tau_filt * (-h_lp[l] + h_pre))  # FF-presyn debias LP (paper tau_f)
                delta_gW.append({"kernel": gate * jnp.outer(h_lp[l], eproj),
                                 "bias": gate * eproj})
                h_pre = jnp.dot(r_exc, self.M_E[l])

        # readout
        if self.fb_to_readout:
            delta_readout = 1 / self.tauE * (-state_vf[-1] + ff_inputs[-1] + ctrl)
        else:
            delta_readout = 1 / self.tauE * (-state_vf[-1] + ff_inputs[-1])
        delta_state_vf.append(delta_readout)

        delta = {
            "vf": delta_state_vf,
            "ctrl": delta_state_ctrl,
            "c_lp": inv_tau_filt * (-c_lp + ctrl),
            "u_inh_lp": delta_u_inh_lp,
            "Qsrc": delta_Qsrc,
        }

        if accumulate_ff:
            e_read = ff_inputs[-1] - state_vf[-1]                    # readout error (fb_to_readout)
            delta_h_lp.append(inv_tau_filt * (-h_lp[-1] + h_pre))    # FF-presyn debias LP (paper tau_f)
            delta["h_lp"] = delta_h_lp
            delta["gW"] = delta_gW
            delta["gW_read"] = gate * jnp.outer(h_lp[-1], e_read)

        return delta
