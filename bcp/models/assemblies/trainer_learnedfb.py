# Learned-feedback trainer
# Julian Rossbroich


import logging
import itertools
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import optax
from jax import vmap
from tqdm import tqdm

from flax.core.frozen_dict import FrozenDict, unfreeze, freeze

from diffrax import (
    diffeqsolve, ODETerm, SaveAt, ConstantStepSize, LinearInterpolation,
    Euler, Heun, Tsit5,
)

from bcp.core.trainer import (
    FeedbackControlTrainer,
    normalize_gradients,
    clip_nn_params,
)
from bcp.models.assemblies.trainer import BalanceControlled
from bcp.models.assemblies.vf_learnedfb import LearnedFeedbackAssemblyVectorField

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Per-assembly OU noise + noisy-trajectory integrators
# --------------------------------------------------------------------------- #
def make_ou_paths(key, ts, sizes, tau_eps, dt):
    """Unit-variance Ornstein-Uhlenbeck processes, one per assembly and layer.

    `sizes` is the per-layer number of ENSEMBLES (noise is per-assembly; it is
    projected to the interneurons through M_I inside `noisy_step`).
    """
    n = len(ts)
    tau_list = list(tau_eps)
    assert len(tau_list) == len(sizes), "tau_eps length must match number of layers"
    paths = []
    for s, tau_l in zip(sizes, tau_list):
        a = dt / tau_l
        amp = jnp.sqrt(2.0 * dt / tau_l)
        key, sub = jax.random.split(key)
        noise = jax.random.normal(sub, (n, s))

        def step(eps_prev, xi):
            eps = eps_prev + a * (-eps_prev) + amp * xi
            return eps, eps

        _, path = jax.lax.scan(step, jnp.zeros(s), noise)
        paths.append(path)
    return paths


def _noisy_solve(model, params, x, y, ol_state, Q, eps_paths, accumulate_ff, use_fr_error,
                 solver):
    """Runs the noisy feedback-learning trajectory (batched).
    """
    vf = model.vf
    T, dt = model.T, model.dt
    ts = jnp.arange(0, T, dt)
    nb_hidden = vf.nb_hidden

    def single(xi, yi, ol_i, eps_i):
        interps = [LinearInterpolation(ts=ts, ys=eps_i[l]) for l in range(nb_hidden)]
        s0 = vf.augmented_initial_state(xi, ol_i, accumulate_ff)

        def f(t, st, args):
            eps_t = [interps[l].evaluate(t) for l in range(nb_hidden)]
            return vf.apply(params, st, t, xi, yi, Q, eps_t, accumulate_ff, use_fr_error,
                            method=vf.noisy_step)

        sol = diffeqsolve(
            ODETerm(f), solver, t0=0.0, t1=T, dt0=dt, y0=s0,
            stepsize_controller=ConstantStepSize(), saveat=SaveAt(t1=True), max_steps=None,
        )
        return jax.tree_util.tree_map(lambda a: a[-1], sol.ys)

    return vmap(single, in_axes=(0, 0, 0, 0))(x, y, ol_state, eps_paths)


def accumulate_squash_source(model, params, x, y_ref, ol_state, Q, eps_paths,
                             use_fr_error=True, solver=None):
    """Controller-squashing feedback source (two-phase or pretrain)"""
    vf = model.vf
    window = jnp.maximum(model.T - vf.t_settle, model.dt)
    final = _noisy_solve(model, params, x, y_ref, ol_state, Q, eps_paths,
                         accumulate_ff=False, use_fr_error=use_fr_error,
                         solver=model.solver if solver is None else solver)
    sources = [final["Qsrc"][l] / window for l in range(vf.nb_hidden)]
    return [jnp.mean(sources[l], axis=0) for l in range(vf.nb_hidden)]


def accumulate_single_phase(model, params, x, y, ol_state, Q, eps_paths,
                            use_fr_error=True, solver=None):
    """Single learning phase noisy closed-loop with task target: returns both assembly-space
    feedback source and BCP feedforward gradient"""
    vf = model.vf
    nb_hidden = vf.nb_hidden
    window = jnp.maximum(model.T - vf.t_settle, model.dt)
    final = _noisy_solve(model, params, x, y, ol_state, Q, eps_paths,
                         accumulate_ff=True, use_fr_error=use_fr_error,
                         solver=model.solver if solver is None else solver)

    sources = [jnp.mean(final["Qsrc"][l] / window, axis=0) for l in range(nb_hidden)]

    ff_grad = {}
    for l in range(nb_hidden):
        ff_grad[f"hidden_{l}"] = {
            "kernel": jnp.mean(final["gW"][l]["kernel"], axis=0) / window,
            "bias": jnp.mean(final["gW"][l]["bias"], axis=0) / window,
        }
    ff_grad["readout"] = {"kernel": jnp.mean(final["gW_read"], axis=0) / window}
    return sources, ff_grad


class LearnedFeedbackBalanceControlled(BalanceControlled):

    # Q update-rule knobs
    q_lr: float = 0.001                # learning rate of the dedicated Q optimizer (own optax instance)
    q_update_every: int = 1            # refresh Q every N steps
    q_init_scale: float = 0.01         # init scale of the random Q direction
    q_seed: int = 0                    # RNG offset for Q init + noise paths
    noisy_solver: str = "euler"

    class ExtTrainState(BalanceControlled.ExtTrainState):
        Q: Any = None            # list per layer, ensemble space [dim_output, nb_ensembles_l]
        q_opt_state: Any = None  # optax state for the dedicated Q optimizer

    # ------------------------------------------------------------------ #
    # Mode parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_mode(feedback_mode):
        """Returns (is_learned, rule).  rule ∈ {'single_phase', 'two_phase'}."""
        fm = str(feedback_mode)
        if fm in ("learned", "learned-singlephase", "learned_singlephase"):
            return True, "single_phase"
        if fm in ("learned-twophase", "learned_twophase"):
            return True, "two_phase"
        return False, None

    def _noisy_solver_instance(self):
        """Fixed-step diffrax solver used for the noise-driven feedback solve."""
        name = str(self.noisy_solver).lower()
        if name in ("model", "default"):
            return self.model.solver
        mapping = {"euler": Euler, "heun": Heun, "tsit5": Tsit5}
        if name not in mapping:
            raise ValueError(
                f"Unknown noisy_solver={self.noisy_solver!r}; "
                f"choose from {sorted(mapping)} (or 'model')."
            )
        return mapping[name]()

    # ------------------------------------------------------------------ #
    # Train-state init
    # ------------------------------------------------------------------ #
    def init_trainstate_params(self, params, rng):
        extra = super().init_trainstate_params(params, rng)  # mean_activity (+ random_fb)

        is_learned, _ = self._parse_mode(self.feedback_mode)
        if not is_learned:
            return extra

        vf = self.model.vf
        if not isinstance(vf, LearnedFeedbackAssemblyVectorField):
            raise ValueError(
                "feedback_mode='learned*' requires a learned-feedback vector field. "
                "Add `model/vf=assemblies-learnedfb` "
            )

        ensemble_sizes, _, _ = vf._get_hidden_sizes()
        key = jax.random.fold_in(rng, self.q_seed)
        Q = []
        for l in range(vf.nb_hidden):
            key, sub = jax.random.split(key)
            Q.append(
                self.q_init_scale
                * jax.random.normal(sub, (vf.dim_output, ensemble_sizes[l]), dtype=self.model.dtype)
            )
        extra["Q"] = Q
        extra["q_opt_state"] = self._q_optimizer().init(Q)
        return extra

    # ------------------------------------------------------------------ #
    # Operative feedback weights from the learned Q
    # ------------------------------------------------------------------ #
    def _learned_fb_weights(self, train_state, batch):
        """Project ensemble Q through M_I, broadcast over the batch, normalize
        each per-sample matrix to `norm_val` (Q magnitude is gauge-free)."""
        vf = self.model.vf
        fb_inh = vf.apply(train_state.params, list(train_state.Q), method=vf.fb_from_Q)
        bs = batch[0].shape[0]

        def _norm_one(w):  # w: [pre, post]
            return w / (jnp.linalg.norm(w) + 1e-12) * self.norm_val

        fb = []
        for w in fb_inh:
            w = jnp.broadcast_to(w, (bs,) + w.shape)
            fb.append(jax.vmap(_norm_one)(w))
        return fb

    # ------------------------------------------------------------------ #
    # Noise, Q update, etc
    # ------------------------------------------------------------------ #
    def _make_eps(self, x, key):
        """Per-(sample, layer) OU noise paths (assembly space)"""
        vf = self.model.vf
        ensemble_sizes, _, _ = vf._get_hidden_sizes()
        ts = jnp.arange(0, self.model.T, self.model.dt)
        tau_eps = vf._tau_eps_per_layer()
        keys = jax.random.split(key, x.shape[0])
        return vmap(lambda k: make_ou_paths(k, ts, ensemble_sizes, tau_eps, self.model.dt))(keys)

    def _squash_reference(self, OL_y_pred):
        """makes the controller error vanish at the OL equilibrium, so
         control is purely noise-driven."""
        name = getattr(self.loss, "name", "")
        if name == "cross_entropy":
            return jax.nn.softmax(OL_y_pred, axis=-1)
        if name == "sigmoid_cross_entropy":
            return jax.nn.sigmoid(OL_y_pred)
        return OL_y_pred

    def _q_optimizer(self):
        """Dedicated optax optimizer for the feedback weights Q (own LR + own state).
        """
        return optax.adam(self.q_lr)

    # ------------------------------------------------------------------ #
    # Optimizer resets
    # ------------------------------------------------------------------ #
    def reset_q_optimizer(self, train_state):
        """
        Reset the dedicated feedback-weight (Q) optimizer state.

        Re-initializes the Q optimizer state from the current Q, clearing any
        accumulated statistics (e.g. Adam moments). The feedback weights Q
        themselves and the trainer are left unchanged.

        Returns a new train_state with a fresh Q optimizer state.
        """
        if train_state.Q is None:
            raise ValueError(
                "reset_q_optimizer requires a learned-feedback train_state "
                "(feedback_mode='learned*'); no Q found."
            )
        new_q_opt_state = self._q_optimizer().init(list(train_state.Q))
        return train_state.replace(q_opt_state=new_q_opt_state)

    def reset_optimizers(self, train_state):
        """
        Reset both the forward-weight and feedback-weight (Q) optimizers.

        Convenience wrapper around `reset_optimizer` (forward weights) and
        `reset_q_optimizer` (feedback weights Q).
        """
        return self.reset_q_optimizer(self.reset_optimizer(train_state))

    def _q_grads(self, Q, sources, beta):
        """Hand-built Q gradient (`beta` is the Q-leak/decay term).
        """
        vf = self.model.vf
        s_l = vf.layer_scaling()
        return [
            s_l[l] * (-vf.update_sign * sources[l] + beta * Q[l])
            for l in range(vf.nb_hidden)
        ]

    # ------------------------------------------------------------------ #
    # feedback (Q) pre-training
    # ------------------------------------------------------------------ #
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def pretrain_step(self, train_state, batch, u0):
        """One Q update from the controller-squashing source. FF weights
        (params) are untouched; only Q / q_opt_state / step change."""
        x = batch[0]
        vf = self.model.vf
        OL_y_pred, OL_state, _ = self.model.openloop(train_state.params, u0, x)
        y_ref = self._squash_reference(OL_y_pred)

        key = jax.random.fold_in(jax.random.PRNGKey(self.q_seed + 7), train_state.step)
        eps_paths = self._make_eps(x, key)
        sources = accumulate_squash_source(
            self.model, train_state.params, x, y_ref, OL_state,
            list(train_state.Q), eps_paths, self.use_fr_error,
            solver=self._noisy_solver_instance())

        # FF is frozen during pretraining -> no point decaying Q (beta=0).
        train_state = self._update_Q(train_state, sources, beta=0.0)
        train_state = train_state.replace(step=train_state.step + 1)

        # Metrics for the UPDATED Q (params/OL_state unchanged — FF is frozen).
        subalign = self._fb_subspace_alignment_from_ol(
            train_state.Q, train_state.params, OL_state)
        metrics = {}
        for l in range(vf.nb_hidden):
            metrics[f"fb_subalign_layer{l}"] = subalign[l]
        return train_state, metrics

    def pretrain_epoch(self, train_state, train_data, batchsize, max_batches=None, **kwargs):
        """One FF-frozen Q pre-training epoch over the training data."""
        total_batches = len(train_data)
        train_data = iter(train_data)

        batch = next(train_data)
        u0 = self.model.vf.get_initial_state_batchexp(batch[0])

        metrics = None
        for i, batch in enumerate(tqdm(itertools.chain([batch], train_data), total=total_batches)):
            if max_batches is not None and i >= max_batches:
                break
            train_state, bm = self.pretrain_step(train_state, batch, u0)
            if metrics is None:
                metrics = {k: [] for k in bm}
            for k, v in bm.items():
                metrics[k].append(v)

        metrics = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
        return train_state, metrics

    # ------------------------------------------------------------------ #
    # Train step (dispatch on feedback_mode; plain Python dispatcher)
    # ------------------------------------------------------------------ #
    def train_step(self, train_state, batch, u0):
        is_learned, rule = self._parse_mode(self.feedback_mode)
        if not is_learned:
            # analytic / random -> identical to BalanceControlled / FeedbackControlTrainer
            return FeedbackControlTrainer.train_step(self, train_state, batch, u0)
        if rule == "two_phase":
            return self._train_step_two_phase(train_state, batch, u0)
        return self._train_step_single_phase(train_state, batch, u0)

    def _norm_clip_grads(self, grads):
        if self.norm_grads:
            grads = normalize_gradients(grads, 1.0)
        if self.clip_grads:
            grads = jax.tree_util.tree_map(
                lambda z: jnp.clip(z, -self.clip_val_grads, self.clip_val_grads), grads)
        return grads

    def _apply_ff_grads(self, train_state, grads):
        """Shared: optional grad norm/clip, apply, optional param clip."""
        grads = self._norm_clip_grads(grads)
        train_state = train_state.apply_gradients(grads=grads)
        if self.clip_params:
            new_params = clip_nn_params(train_state.params, -self.clip_val_params, self.clip_val_params)
            train_state = train_state.replace(params=new_params)
        return train_state

    def _update_Q(self, train_state, sources, beta=None):
        """Apply one Q-optimizer step every `q_update_every` steps.

        `beta` overrides the Q-leak term (defaults to `vf.beta`); pretraining
        passes `beta=0.0` so the Q matrix does not decay while FF is frozen.
        """
        if beta is None:
            beta = self.model.vf.beta
        q_tx = self._q_optimizer()

        def do_step():
            gQ = self._q_grads(train_state.Q, sources, beta)
            gQ = self._norm_clip_grads(gQ)   # honor norm_grads / clip_grads for Q too
            updates, new_opt_state = q_tx.update(
                gQ, train_state.q_opt_state, train_state.Q)
            new_Q = optax.apply_updates(train_state.Q, updates)
            return list(new_Q), new_opt_state

        do_update = (train_state.step % self.q_update_every) == 0
        new_Q, new_opt_state = jax.lax.cond(
            do_update,
            do_step,
            lambda: (list(train_state.Q), train_state.q_opt_state),
        )
        return train_state.replace(Q=new_Q, q_opt_state=new_opt_state)

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _train_step_two_phase(self, train_state, batch, u0):
        """Two-phase feedback training: OL settle -> noisy squashing (learn Q) -> clean closed-loop
        with new Q -> standard BCP feedforward update."""
        x, y = batch[0], batch[1]
        OL_y_pred, OL_state, OL_vf_sol = self.model.openloop(train_state.params, u0, x)

        # ---- feedback-learning phase: accumulate source, update Q ----
        y_ref = self._squash_reference(OL_y_pred)
        key = jax.random.fold_in(jax.random.PRNGKey(self.q_seed + 1), train_state.step)
        eps_paths = self._make_eps(x, key)
        sources = accumulate_squash_source(
            self.model, train_state.params, x, y_ref, OL_state,
            list(train_state.Q), eps_paths, self.use_fr_error,
            solver=self._noisy_solver_instance())
        train_state = self._update_Q(train_state, sources)

        # ---- clean closed-loop with the updated Q -> FF (BCP) gradient ----
        fb_weights = self._learned_fb_weights(train_state, batch)
        y_targets = self.calculate_targets(OL_y_pred, y)
        CL_y_pred, CL_state, CL_vf_sol = self.model.closedloop(
            train_state.params, OL_state, x, y_targets, fb_weights)

        metrics = self.calc_metrics(CL_y_pred, y, train_state, CL_vf_sol)

        train_state = train_state.replace(
            **self.update_trainstate_params(train_state, OL_vf_sol, x))
        grads = self.get_gradients(
            train_state, x, y, OL_y_pred, CL_y_pred, OL_state, CL_state)
        train_state = self._apply_ff_grads(train_state, grads)

        return train_state, CL_vf_sol, metrics

    def _assemble_ff_grads(self, params, ff_grad):
        """Convert in-ODE computed FF gradient into params-shaped grad pytree"""
        g = unfreeze(jax.tree_util.tree_map(jnp.zeros_like, params))
        for l in range(self.model.vf.nb_hidden):
            g["params"][f"hidden_{l}"]["kernel"] = ff_grad[f"hidden_{l}"]["kernel"]
            if "bias" in g["params"][f"hidden_{l}"]:
                g["params"][f"hidden_{l}"]["bias"] = ff_grad[f"hidden_{l}"]["bias"]
        g["params"]["readout"]["kernel"] = ff_grad["readout"]["kernel"]
        return freeze(g) if isinstance(params, FrozenDict) else g

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _train_step_single_phase(self, train_state, batch, u0):
        """Single-phase feedback training: OL settle -> noisy closed-loop"""
        x, y = batch[0], batch[1]
        OL_y_pred, OL_state, OL_vf_sol = self.model.openloop(train_state.params, u0, x)
        y_targets = self.calculate_targets(OL_y_pred, y)

        key = jax.random.fold_in(jax.random.PRNGKey(self.q_seed + 1), train_state.step)
        eps_paths = self._make_eps(x, key)
        sources, ff_grad = accumulate_single_phase(
            self.model, train_state.params, x, y_targets, OL_state,
            list(train_state.Q), eps_paths, self.use_fr_error,
            solver=self._noisy_solver_instance())

        # metrics from the open-loop prediction (no clean closed-loop prediction exists)
        metrics = self.calc_metrics(OL_y_pred, y, train_state, OL_vf_sol)

        train_state = train_state.replace(
            **self.update_trainstate_params(train_state, OL_vf_sol, x))

        # Q update first (uses the pre-apply step, matching the two-phase ordering),
        # then the FF update (apply_gradients bumps the step).
        train_state = self._update_Q(train_state, sources)
        grads = self._assemble_ff_grads(train_state.params, ff_grad)
        train_state = self._apply_ff_grads(train_state, grads)

        return train_state, OL_vf_sol, metrics

    # ------------------------------------------------------------------ #
    # Alignment / compliance metrics
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fb_subspace_alignment_ratio(Q_l, J_l, ridge=1e-6):
        """FB-subspace alignment: fraction of ||Q_l||_F lying in row(J_l).

        Both `Q_l` and `J_l` live in inhibitory-population space
        ([dim_output, sizes_inh]): `Q_l` is the learned feedback projected through
        M_I, `J_l` is `vf.calculate_jacobian`. Returns a value in [0, 1]; 1 means the
        (row) space of the projected feedback lies entirely inside row(J).
        """
        m = J_l.shape[0]
        gram = J_l @ J_l.T + ridge * jnp.eye(m, dtype=J_l.dtype)
        P = J_l.T @ jnp.linalg.solve(gram, J_l)          # [sizes_inh, sizes_inh]
        return jnp.linalg.norm(Q_l @ P) / (jnp.linalg.norm(Q_l) + 1e-12)

    def _fb_subspace_alignment_from_ol(self, Q, params, ol_state):
        """Per-layer FB-subspace alignment from an existing (batched) open-loop state.

        Compares the learned feedback PROJECTED into inhibitory-population space
        (`fb_from_Q`, [dim_output, sizes_inh]) against the PER-SAMPLE inhibitory-space
        Jacobian (`calculate_jacobian`), averaging the ratio over the batch. This is the
        fair comparison: both operands are the connectivity the network actually uses.
        """
        vf = self.model.vf
        Q_proj = vf.apply(params, list(Q), method=vf.fb_from_Q)  # list [dim_output, sizes_inh]

        def _per_sample(vf_state):
            return vf.apply(params, vf_state, method=vf.calculate_jacobian)

        J = jax.vmap(_per_sample)(ol_state["vf"])  # list per layer [batch, dout, sizes_inh]
        out = []
        for l in range(vf.nb_hidden):
            Q_l = Q_proj[l]
            ratios = jax.vmap(lambda Jb: self._fb_subspace_alignment_ratio(Q_l, Jb))(J[l])
            out.append(jnp.mean(ratios))
        return out

    @partial(jax.jit, static_argnums=(0,))
    def fb_subspace_alignment(self, train_state, batch):
        """Per-layer FB-subspace alignment: how much of the projected learned feedback
        lies in the row space of the inhibitory-space Jacobian (in [0, 1])."""
        vf = self.model.vf
        u0 = vf.get_initial_state_batchexp(batch[0])
        _, OL_state, _ = self.model.openloop(train_state.params, u0, batch[0])
        return self._fb_subspace_alignment_from_ol(train_state.Q, train_state.params, OL_state)

    @partial(jax.jit, static_argnums=(0,))
    def fb_subspace_alignment_ceiling(self, train_state, batch):
        """Per-layer upper bound on the achievable FB-subspace alignment: the batch-mean
        inhibitory-space Jacobian used as a stand-in for the projected feedback, scored
        against each per-sample Jacobian."""
        vf = self.model.vf
        u0 = vf.get_initial_state_batchexp(batch[0])
        _, OL_state, _ = self.model.openloop(train_state.params, u0, batch[0])

        def _per_sample(vf_state):
            return vf.apply(train_state.params, vf_state, method=vf.calculate_jacobian)

        J = jax.vmap(_per_sample)(OL_state["vf"])  # list per layer [batch, dout, sizes_inh]
        out = []
        for l in range(vf.nb_hidden):
            jbar = jnp.mean(J[l], axis=0)
            ratios = jax.vmap(lambda Jb: self._fb_subspace_alignment_ratio(jbar, Jb))(J[l])
            out.append(jnp.mean(ratios))
        return out

    @staticmethod
    def _angle_deg(a, b):
        """Angle (degrees) between two grad pytrees, flattened to a single vector."""
        la = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(a)])
        lb = jnp.concatenate([x.ravel() for x in jax.tree_util.tree_leaves(b)])
        cos = jnp.dot(la, lb) / (jnp.linalg.norm(la) * jnp.linalg.norm(lb) + 1e-12)
        return jnp.degrees(jnp.arccos(jnp.clip(cos, -1.0, 1.0)))

    @partial(jax.jit, static_argnums=(0,))
    def ff_update_alignment(self, train_state, batch):
        """Per-block angle (deg) between the FF grads the ACTIVE FB-learning method
        applies and the FF grads the analytic Jacobian feedback would apply.

        The learned side is *exactly* what the current method applies on that batch:
        two-phase -> clean closed-loop with the learned feedback; single-phase -> the
        noisy closed-loop solve with in-ODE FF-grad accumulation. Both the learned and
        the analytic reference start from the open-loop steady state, as the real train
        steps do. Grads are compared raw (before norm/clip). Returns a list of length
        nb_hidden+1: hidden layers first, then the readout.
        """
        x, y = batch[0], batch[1]
        vf = self.model.vf
        u0 = vf.get_initial_state_batchexp(x)
        OL_y_pred, OL_state, _ = self.model.openloop(train_state.params, u0, x)
        y_targets = self.calculate_targets(OL_y_pred, y)

        # analytic reference: exactly what BalanceControlled applies (clean CL from OL steady state)
        fb_a = self.modify_fb_weights(
            self.model.get_fb_weights(train_state.params, OL_state), batch)
        CL_ya, CL_state_a, _ = self.model.closedloop(
            train_state.params, OL_state, x, y_targets, fb_a)
        grads_a = self.get_gradients(
            train_state, x, y, OL_y_pred, CL_ya, OL_state, CL_state_a)

        # learned side: exactly the FF grads the active method applies (dispatch on rule)
        _, rule = self._parse_mode(self.feedback_mode)  # static -> resolved at trace time
        if rule == "two_phase":
            fb_l = self._learned_fb_weights(train_state, batch)
            CL_yl, CL_state_l, _ = self.model.closedloop(
                train_state.params, OL_state, x, y_targets, fb_l)
            grads_l = self.get_gradients(
                train_state, x, y, OL_y_pred, CL_yl, OL_state, CL_state_l)
        else:  # single_phase: noisy CL solve + in-ODE FF grad accumulation
            key = jax.random.fold_in(jax.random.PRNGKey(self.q_seed + 1), train_state.step)
            eps_paths = self._make_eps(x, key)
            _, ff_grad = accumulate_single_phase(
                self.model, train_state.params, x, y_targets, OL_state,
                list(train_state.Q), eps_paths, self.use_fr_error,
                solver=self._noisy_solver_instance())
            grads_l = self._assemble_ff_grads(train_state.params, ff_grad)

        blocks = [f"hidden_{l}" for l in range(vf.nb_hidden)] + ["readout"]
        return [self._angle_deg(grads_l["params"][b], grads_a["params"][b]) for b in blocks]

    @partial(jax.jit, static_argnums=(0,))
    def feedback_strength_ratio(self, train_state, batch):
        """per-layer `||Q u||_F / ||W r||_F` at the
        closed-loop state. 
        """
        vf = self.model.vf
        nb_hidden = vf.nb_hidden
        x, y = batch[0], batch[1]
        u0 = vf.get_initial_state_batchexp(x)
        OL_y_pred, OL_state, _ = self.model.openloop(train_state.params, u0, x)
        y_targets = self.calculate_targets(OL_y_pred, y)
        fb_weights = self._learned_fb_weights(train_state, batch)   # list, each [bs, dout, sizes_inh]
        CL_y_pred, CL_state, _ = self.model.closedloop(
            train_state.params, OL_state, x, y_targets, fb_weights)

        def per_sample(x_i, vf_state, ctrl_state, yp, yt, fbw):
            ctrl = vf.controller(yp, yt, ctrl_state)[0]            # [dout]
            ff = vf.apply(train_state.params, x_i, vf_state, method=vf._compute_ff_inputs)
            num = [jnp.linalg.norm(jnp.dot(ctrl, fbw[l])) for l in range(nb_hidden)]
            den = [jnp.linalg.norm(ff[l]["exc"]) for l in range(nb_hidden)]
            return num, den

        num, den = jax.vmap(per_sample)(
            x, CL_state["vf"], CL_state["ctrl"], CL_y_pred, y_targets, fb_weights)
        return [jnp.mean(num[l]) / (jnp.mean(den[l]) + 1e-12) for l in range(nb_hidden)]
