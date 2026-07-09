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

    def _q_grads(self, Q, sources):
        """Hand-built Q gradient`.
        """
        vf = self.model.vf
        s_l = vf.layer_scaling()
        return [
            s_l[l] * (-vf.update_sign * sources[l] + vf.beta * Q[l])
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
        jbar = self._ensemble_jacobian_mean(train_state.params, OL_state)
        y_ref = self._squash_reference(OL_y_pred)

        key = jax.random.fold_in(jax.random.PRNGKey(self.q_seed + 7), train_state.step)
        eps_paths = self._make_eps(x, key)
        sources = accumulate_squash_source(
            self.model, train_state.params, x, y_ref, OL_state,
            list(train_state.Q), eps_paths, self.use_fr_error,
            solver=self._noisy_solver_instance())

        train_state = self._update_Q(train_state, sources)
        train_state = train_state.replace(step=train_state.step + 1)

        # Metrics for the UPDATED Q (params/OL_state unchanged — FF is frozen).
        # Condition 1 is the primary compliance metric; cos_F is kept as secondary.
        con1 = self._condition1_from_ol(train_state.Q, train_state.params, OL_state)
        metrics = {}
        for l in range(vf.nb_hidden):
            metrics[f"con1_layer{l}"] = con1[l]
            metrics[f"salign_layer{l}"] = self._cosF(train_state.Q[l], jbar[l])
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

    def _apply_ff_grads(self, train_state, grads):
        """Shared: optional grad norm/clip, apply, optional param clip."""
        if self.norm_grads:
            grads = normalize_gradients(grads, 1.0)
        if self.clip_grads:
            grads = jax.tree_util.tree_map(
                lambda z: jnp.clip(z, -self.clip_val_grads, self.clip_val_grads), grads)
        train_state = train_state.apply_gradients(grads=grads)
        if self.clip_params:
            new_params = clip_nn_params(train_state.params, -self.clip_val_params, self.clip_val_params)
            train_state = train_state.replace(params=new_params)
        return train_state

    def _update_Q(self, train_state, sources):
        """Apply one Q-optimizer step every `q_update_every` steps."""
        q_tx = self._q_optimizer()

        def do_step():
            gQ = self._q_grads(train_state.Q, sources)
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

        # ---- feedback-learning (squash) phase: accumulate source, update Q ----
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
    def _cosF(A, B):
        """Frobenius cosine <A,B> / (||A|| ||B||)."""
        return jnp.sum(A * B) / (jnp.linalg.norm(A) * jnp.linalg.norm(B) + 1e-12)

    @staticmethod
    def _condition1_ratio(Q_l, J_l, ridge=1e-6):
        """Strong-DFC Condition 1 (Eq 106-107): fraction of ||Q||_F lying in row(J).
        """
        m = J_l.shape[0]
        gram = J_l @ J_l.T + ridge * jnp.eye(m, dtype=J_l.dtype)
        P = J_l.T @ jnp.linalg.solve(gram, J_l)          # [nb_ens, nb_ens]
        return jnp.linalg.norm(Q_l @ P) / (jnp.linalg.norm(Q_l) + 1e-12)

    def _ensemble_jacobian_mean(self, params, ol_state):
        vf = self.model.vf

        def _calc(vf_state):
            return vf.apply(params, vf_state, method=vf.ensemble_jacobian)

        per_sample = jax.vmap(_calc)(ol_state["vf"])  # list per layer [batch, dout, nb_ens]
        return [jnp.mean(per_sample[l], axis=0) for l in range(vf.nb_hidden)]

    def _condition1_from_ol(self, Q, params, ol_state):
        """Per-layer Condition-1 ratio from an existing (batched) open-loop state.

        Uses the PER-SAMPLE assembly-space Jacobian (`ensemble_jacobian`, before the M_I
        inhibitory projection) and averages the ratio over the batch.
        """
        vf = self.model.vf

        def _per_sample(vf_state):
            return vf.apply(params, vf_state, method=vf.ensemble_jacobian)

        J = jax.vmap(_per_sample)(ol_state["vf"])  # list per layer [batch, dout, nb_ens]
        out = []
        for l in range(vf.nb_hidden):
            Q_l = Q[l]
            ratios = jax.vmap(lambda Jb: self._condition1_ratio(Q_l, Jb))(J[l])
            out.append(jnp.mean(ratios))
        return out

    @partial(jax.jit, static_argnums=(0,))
    def condition1(self, train_state, batch):
        vf = self.model.vf
        u0 = vf.get_initial_state_batchexp(batch[0])
        _, OL_state, _ = self.model.openloop(train_state.params, u0, batch[0])
        return self._condition1_from_ol(train_state.Q, train_state.params, OL_state)

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

    @partial(jax.jit, static_argnums=(0,))
    def feedback_alignment(self, train_state, batch):
        """per-layer signed Frobenius cosine between the learned Q and
        the batch-mean ensemble-space analytic Jacobian. """
        vf = self.model.vf
        u0 = vf.get_initial_state_batchexp(batch[0])
        _, OL_state, _ = self.model.openloop(train_state.params, u0, batch[0])
        jbar = self._ensemble_jacobian_mean(train_state.params, OL_state)
        return [self._cosF(train_state.Q[l], jbar[l]) for l in range(vf.nb_hidden)]
