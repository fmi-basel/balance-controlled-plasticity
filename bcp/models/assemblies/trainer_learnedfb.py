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
    """Ornstein-Uhlenbeck processes, one per assembly and layer.
    Integrates the linear SDE,
        eps[m+1] = eps[m] e^(-dt/tau) + N(0, var (1 - e^(-2 dt/tau))),   var = 1/(2 tau)
    """
    n = len(ts)
    tau_list = list(tau_eps)
    assert len(tau_list) == len(sizes), "tau_eps length must match number of layers"
    paths = []
    for s, tau_l in zip(sizes, tau_list):
        decay = jnp.exp(-dt / tau_l)
        std = jnp.sqrt((1.0 - decay ** 2) / (2.0 * tau_l))
        key, sub, sub0 = jax.random.split(key, 3)
        noise = jax.random.normal(sub, (n, s))
        eps0 = jax.random.normal(sub0, (s,)) / jnp.sqrt(2.0 * tau_l)

        def step(eps_prev, xi):
            eps = decay * eps_prev + std * xi
            return eps, eps

        _, path = jax.lax.scan(step, eps0, noise)
        paths.append(path)
    return paths


def _noisy_solve(model, params, x, y, ol_state, Q, eps_paths, accumulate_ff, use_fr_error,
                 solver, assembly_current=0.0, dt=None, sigma=None, tau_filt=None,
                 controller_overrides=None):
    """Runs the noisy feedback-learning trajectory (batched).

    `dt` overrides the model timestep only for this noisy solve.
    `sigma` / `tau_filt` override the VF params (only used by optimization of hyperparameters)
    """
    vf = model.vf
    T = model.T
    dt = model.dt if dt is None else dt
    ts = jnp.arange(0, T, dt)
    nb_hidden = vf.nb_hidden

    def single(xi, yi, ol_i, eps_i):
        interps = [LinearInterpolation(ts=ts, ys=eps_i[l]) for l in range(nb_hidden)]
        # bound call: the filter initialisation reads M_E and the controller
        s0 = vf.apply(params, xi, yi, ol_i, accumulate_ff, controller_overrides,
                      method=vf.augmented_initial_state)

        def f(t, st, args):
            eps_t = [interps[l].evaluate(t) for l in range(nb_hidden)]
            return vf.apply(params, st, t, xi, yi, Q, eps_t, accumulate_ff, use_fr_error,
                            assembly_current, sigma, tau_filt, controller_overrides,
                            method=vf.noisy_step)

        sol = diffeqsolve(
            ODETerm(f), solver, t0=0.0, t1=T, dt0=dt, y0=s0,
            stepsize_controller=ConstantStepSize(), saveat=SaveAt(t1=True), max_steps=None,
        )
        return jax.tree_util.tree_map(lambda a: a[-1], sol.ys)

    return vmap(single, in_axes=(0, 0, 0, 0))(x, y, ol_state, eps_paths)


def accumulate_squash_source(model, params, x, y_ref, ol_state, Q, eps_paths,
                             solver=None, assembly_current=0.0, dt=None,
                             sigma=None, tau_filt=None, controller_overrides=None):
    """Controller-squashing feedback source (two-phase or pretrain).
    """
    vf = model.vf
    dt = model.dt if dt is None else dt
    window = jnp.maximum(model.T - vf.t_settle, dt)
    final = _noisy_solve(model, params, x, y_ref, ol_state, Q, eps_paths,
                         accumulate_ff=False, use_fr_error=True,
                         solver=model.solver if solver is None else solver,
                         assembly_current=assembly_current, dt=dt,
                         sigma=sigma, tau_filt=tau_filt,
                         controller_overrides=controller_overrides)
    return [jnp.mean(final["Qsrc"][l] / window, axis=0) for l in range(vf.nb_hidden)]


def accumulate_single_phase(model, params, x, y, ol_state, Q, eps_paths,
                            use_fr_error=True, solver=None,
                            sigma=None, tau_filt=None, controller_overrides=None):
    """Single learning phase noisy closed-loop with task target: returns both assembly-space
    feedback source and BCP feedforward gradient"""
    vf = model.vf
    nb_hidden = vf.nb_hidden
    window = jnp.maximum(model.T - vf.t_settle, model.dt)
    final = _noisy_solve(model, params, x, y, ol_state, Q, eps_paths,
                         accumulate_ff=True, use_fr_error=use_fr_error,
                         solver=model.solver if solver is None else solver,
                         sigma=sigma, tau_filt=tau_filt,
                         controller_overrides=controller_overrides)

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
    q_lr: float = 0.001                # LR of the dedicated Q optimizer during joint (FF+Q) training
    q_lr_pretraining: float = None     # LR of the Q optimizer during FF-frozen pretraining; None -> q_lr
    q_update_every: int = 1            # refresh Q every N steps
    q_init_scale: float = 0.01         # init scale of raw Q; irrelevant when norm_fb_weights=True
    q_seed: int = 0                    # RNG offset for Q init + noise paths
    noisy_solver: str = "euler"
    fb_learning_dt: float = 1e-2       # timestep only for separate pretrain/two-phase Q solves
    inject_current_during_fb_learning: bool = False
    fb_learning_current: float = 1.0

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
        init_scale = 1.0 if self.norm_fb_weights else self.q_init_scale
        Q = []
        for l in range(vf.nb_hidden):
            key, sub = jax.random.split(key)
            Q.append(
                init_scale
                * jax.random.normal(sub, (vf.dim_output, ensemble_sizes[l]), dtype=self.model.dtype)
            )
        Q = self._normalize_learned_Q(params, Q)
        extra["Q"] = Q
        extra["q_opt_state"] = self._q_optimizer().init(Q)
        return extra

    # ------------------------------------------------------------------ #
    # Feedback weights from the learned Q
    # ------------------------------------------------------------------ #
    def _normalize_learned_Q(self, params, Q):
        """Rescale Q so its projection has Frobenius norm ``norm_val`` in every layer.
        """
        if not self.norm_fb_weights:
            return list(Q)

        vf = self.model.vf
        fb_inh = vf.apply(params, list(Q), method=vf.fb_from_Q)
        return [
            q * (self.norm_val / (jnp.linalg.norm(w) + 1e-12))
            for q, w in zip(Q, fb_inh)
        ]

    def _learned_fb_weights(self, train_state, batch):
        """Project Q and batch-broadcast
        """
        vf = self.model.vf
        fb_inh = vf.apply(train_state.params, list(train_state.Q), method=vf.fb_from_Q)
        bs = batch[0].shape[0]
        return [jnp.broadcast_to(w, (bs,) + w.shape) for w in fb_inh]

    def _applied_fb_weights(self, train_state, batch):
        """broadcast and modify feedback before applying
        """
        is_learned, _ = self._parse_mode(self.feedback_mode)
        if is_learned:
            if train_state.Q is None:
                raise ValueError("learned feedback metrics require train_state.Q")
            return self._learned_fb_weights(train_state, batch)
        if str(self.feedback_mode) == "random":
            if train_state.random_fb is None:
                raise ValueError("random feedback metrics require train_state.random_fb")
            bs = batch[0].shape[0]
            fb = [jnp.broadcast_to(w, (bs,) + w.shape) for w in train_state.random_fb]
            return self.modify_fb_weights(fb, batch)
        raise ValueError(
            "only learned* or random feedback can be scored against the analytic reference"
        )

    # ------------------------------------------------------------------ #
    # Noise, Q update, etc
    # ------------------------------------------------------------------ #
    def _make_eps(self, x, key, dt=None, tau_eps=None):
        """Per-(sample, layer) OU paths at `dt`; default preserves `model.dt`.
        """
        vf = self.model.vf
        dt = self.model.dt if dt is None else dt
        if dt <= 0:
            raise ValueError("feedback-learning dt must be positive")
        ensemble_sizes, _, _ = vf._get_hidden_sizes()
        ts = jnp.arange(0, self.model.T, dt)
        tau_eps = vf._tau_eps_per_layer(tau_eps)
        keys = jax.random.split(key, x.shape[0])
        return vmap(lambda k: make_ou_paths(k, ts, ensemble_sizes, tau_eps, dt))(keys)

    def _squash_reference(self, OL_y_pred):
        """makes the controller error vanish at the OL equilibrium, so
         control is purely noise-driven."""
        name = getattr(self.loss, "name", "")
        if name == "cross_entropy":
            return jax.nn.softmax(OL_y_pred, axis=-1)
        if name == "sigmoid_cross_entropy":
            return jax.nn.sigmoid(OL_y_pred)
        return OL_y_pred

    def _q_optimizer(self, pretraining=False):
        """Dedicated optax optimizer for the feedback weights
        """
        lr = self.q_lr
        if pretraining and self.q_lr_pretraining is not None:
            lr = self.q_lr_pretraining
        return optax.adam(lr)

    def _assembly_current_for_fb_learning(self):
        """Assembly-space current used only by the separate Q-learning phase."""
        if not self.inject_current_during_fb_learning:
            return 0.0
        if self.fb_learning_current < 0.0:
            raise ValueError("fb_learning_current must be non-negative.")
        return self.fb_learning_current

    # ------------------------------------------------------------------ #
    # Optimizer resets
    # ------------------------------------------------------------------ #
    def reset_q_optimizer(self, train_state):
        """
        Reset Q optimizer state.
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
        """
        return self.reset_q_optimizer(self.reset_optimizer(train_state))

    def _resolve_beta(self, beta=None):
        if beta is None:
            beta = self.model.vf.beta
        if self.norm_fb_weights:
            if beta:
                logger.warning(
                    "vf.beta=%g is ignored: norm_fb_weights=True renormalises Q after "
                    "every update, which discards the Q leak. Set beta=0 to silence, or "
                    "norm_fb_weights=False to use it.", beta)
            return 0.0
        return beta

    def _q_grads(self, Q, sources, beta):
        """Q gradient (descent), i.e. -dQ
        """
        vf = self.model.vf
        s_l = vf.layer_scaling()
        return [
            s_l[l] * (-sources[l] + beta * Q[l])
            for l in range(vf.nb_hidden)
        ]

    # ------------------------------------------------------------------ #
    # feedback (Q) pre-training
    # ------------------------------------------------------------------ #
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def pretrain_step(self, train_state, batch, u0):
        x = batch[0]
        vf = self.model.vf
        OL_y_pred, OL_state, _ = self.model.openloop(train_state.params, u0, x)
        y_ref = self._squash_reference(OL_y_pred)

        key = jax.random.fold_in(jax.random.PRNGKey(self.q_seed + 7), train_state.step)
        eps_paths = self._make_eps(x, key, dt=self.fb_learning_dt)
        sources = accumulate_squash_source(
            self.model, train_state.params, x, y_ref, OL_state,
            list(train_state.Q), eps_paths,
            solver=self._noisy_solver_instance(),
            assembly_current=self._assembly_current_for_fb_learning(),
            dt=self.fb_learning_dt)


        train_state = self._update_Q(train_state, sources, beta=0.0, force_update=True,
                                     pretraining=True)
        train_state = train_state.replace(step=train_state.step + 1)

        # Metrics
        fb_inh = vf.apply(train_state.params, list(train_state.Q), method=vf.fb_from_Q)
        subalign = self._fb_subspace_alignment_from_projected(
            fb_inh, train_state.params, OL_state)
        metrics = {}
        for l in range(vf.nb_hidden):
            metrics[f"fb_subalign_layer{l}"] = subalign[l]
        return train_state, metrics

    def pretrain_epoch(self, train_state, train_data, batchsize, max_batches=None,
                       monitor=None, **kwargs):
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
            if monitor is not None:
                monitor.record_batch(train_state, None, bm)
            if metrics is None:
                metrics = {k: [] for k in bm}
            for k, v in bm.items():
                metrics[k].append(v)

        metrics = {k: jnp.mean(jnp.stack(v)) for k, v in metrics.items()}
        if monitor is not None:
            monitor.record_epoch()
        return train_state, metrics

    # ------------------------------------------------------------------ #
    # Train step
    # ------------------------------------------------------------------ #
    def train_step(self, train_state, batch, u0):
        is_learned, rule = self._parse_mode(self.feedback_mode)
        if not is_learned:
            # analytic / random -> identical to BalanceControlled / FeedbackControlTrainer
            return FeedbackControlTrainer.train_step(self, train_state, batch, u0)
        if rule == "two_phase":
            return self._train_step_two_phase(train_state, batch, u0)
        if self.inject_current_during_fb_learning:
            raise ValueError(
                "inject_current_during_fb_learning is only supported for the separate "
                "Q-learning phase in learned-twophase training (and Q pretraining); "
                "it is not supported by single-phase feedback learning."
            )
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

    def _clean_ff_phase(self, train_state, batch, OL_y_pred, OL_state, OL_vf_sol):
        """Apply the clean FF half of two-phase learning with the current Q.
        """
        x, y = batch[0], batch[1]
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

    def _shared_q_oracle_from_ol_state(self, params, OL_state, ridge_rel=1e-6):
        """Normalized representable batch-mean analytic Jacobian in Q coordinates."""
        vf = self.model.vf
        jacobians = jax.vmap(
            lambda state: vf.apply(params, state, method=vf.calculate_jacobian)
        )(OL_state["vf"])
        memberships = params["constants"]["memberships"]["M_I"]
        oracle_Q = []
        for J, membership in zip(jacobians, memberships):
            M = membership.astype(J.dtype)
            jbar = jnp.mean(J, axis=0)
            gram = M.T @ M
            scale = jnp.trace(gram) / gram.shape[0]
            ridge = ridge_rel * (scale + 1e-12)
            # min_Q ||Q M^T - Jbar||_F, written as a solve rather than an inverse.
            rhs = M.T @ jbar.T
            q = jnp.linalg.solve(
                gram + ridge * jnp.eye(gram.shape[0], dtype=gram.dtype), rhs).T
            oracle_Q.append(q)
        return self._normalize_learned_Q(params, oracle_Q)

    @partial(jax.jit, static_argnums=(0,))
    def shared_q_oracle(self, train_state, batch, u0):
        """Return the per-batch zero-lag shared-Q oracle without changing state."""
        x = batch[0]
        _, OL_state, _ = self.model.openloop(train_state.params, u0, x)
        return self._shared_q_oracle_from_ol_state(train_state.params, OL_state)

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def frozen_q_train_step(self, train_state, batch, u0):
        """Update FF once with the pretrained Q fixed and no noisy Q solve."""
        x = batch[0]
        OL_y_pred, OL_state, OL_vf_sol = self.model.openloop(
            train_state.params, u0, x)
        return self._clean_ff_phase(
            train_state, batch, OL_y_pred, OL_state, OL_vf_sol)

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def oracle_shared_q_train_step(self, train_state, batch, u0):
        """Refresh Q to the batch-mean analytic oracle, then update FF once."""
        x = batch[0]
        OL_y_pred, OL_state, OL_vf_sol = self.model.openloop(
            train_state.params, u0, x)
        oracle_Q = self._shared_q_oracle_from_ol_state(train_state.params, OL_state)
        train_state = train_state.replace(Q=oracle_Q)
        return self._clean_ff_phase(
            train_state, batch, OL_y_pred, OL_state, OL_vf_sol)

    def _apply_Q_update(self, train_state, sources, beta=None, pretraining=False):
        """Apply one unconditional Q-optimizer step.
        """
        q_tx = self._q_optimizer(pretraining=pretraining)

        gQ = self._q_grads(train_state.Q, sources, self._resolve_beta(beta))
        gQ = self._norm_clip_grads(gQ)   # honor norm_grads / clip_grads for Q too
        updates, new_opt_state = q_tx.update(
            gQ, train_state.q_opt_state, train_state.Q)
        new_Q = optax.apply_updates(train_state.Q, updates)
        new_Q = self._normalize_learned_Q(train_state.params, new_Q)
        return train_state.replace(Q=new_Q, q_opt_state=new_opt_state)

    def _update_Q(self, train_state, sources, beta=None, force_update=False, pretraining=False):
        """Apply one Q-optimizer step every `q_update_every` steps.
        """
        if force_update or self.q_update_every == 1:
            return self._apply_Q_update(
                train_state, sources, beta=beta, pretraining=pretraining)

        do_update = (train_state.step % self.q_update_every) == 0
        return jax.lax.cond(
            do_update,
            lambda ts: self._apply_Q_update(
                ts, sources, beta=beta, pretraining=pretraining),
            lambda ts: ts,
            train_state,
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def _train_step_two_phase(self, train_state, batch, u0):
        """Two-phase feedback training: OL settle -> noisy squashing (learn Q) -> clean closed-loop
        with new Q -> standard BCP feedforward update."""
        x, y = batch[0], batch[1]
        OL_y_pred, OL_state, OL_vf_sol = self.model.openloop(train_state.params, u0, x)

        # ---- feedback-learning phase: accumulate source, update Q ----
        y_ref = self._squash_reference(OL_y_pred)

        def learn_Q(ts):
            key = jax.random.fold_in(jax.random.PRNGKey(self.q_seed + 1), ts.step)
            eps_paths = self._make_eps(x, key, dt=self.fb_learning_dt)
            sources = accumulate_squash_source(
                self.model, ts.params, x, y_ref, OL_state,
                list(ts.Q), eps_paths,
                solver=self._noisy_solver_instance(),
                assembly_current=self._assembly_current_for_fb_learning(),
                dt=self.fb_learning_dt)
            return self._apply_Q_update(ts, sources)

        if self.q_update_every == 1:
            train_state = learn_Q(train_state)
        else:
            do_update = (train_state.step % self.q_update_every) == 0
            train_state = jax.lax.cond(
                do_update, learn_Q, lambda ts: ts, train_state)

        # ---- clean closed-loop with the updated Q -> FF (BCP) gradient ----
        return self._clean_ff_phase(
            train_state, batch, OL_y_pred, OL_state, OL_vf_sol)

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
    def _fb_subspace_alignment_energy(Q_l, J_l, ridge_rel=1e-6):
        """Fraction of ||Q_l||_F^2 lying in row(J_l), i.e. subspace alignment but 
        with squared ||Q_l||_F
        """
        m = J_l.shape[0]
        gram = J_l @ J_l.T
        scale = jnp.trace(gram) / m
        ridge = ridge_rel * (scale + 1e-12)
        P = J_l.T @ jnp.linalg.solve(
            gram + ridge * jnp.eye(m, dtype=J_l.dtype), J_l)   # [sizes_inh, sizes_inh]
        return jnp.sum((Q_l @ P) ** 2) / (jnp.sum(Q_l ** 2) + 1e-12)

    @classmethod
    def _fb_subspace_alignment_ratio(cls, Q_l, J_l, ridge_rel=1e-6):
        """FB-subspace alignment: fraction of ||Q_l||_F lying in row(J_l). In [0, 1].
        """
        return jnp.sqrt(cls._fb_subspace_alignment_energy(Q_l, J_l, ridge_rel))

    def _fb_subspace_alignment_from_projected(self, fb_inh, params, ol_state):
        """alignment of projected Q at an open-loop state.
        """
        vf = self.model.vf

        def _per_sample(vf_state):
            return vf.apply(params, vf_state, method=vf.calculate_jacobian)

        J = jax.vmap(_per_sample)(ol_state["vf"])  # list per layer [batch, dout, sizes_inh]
        out = []
        for l in range(vf.nb_hidden):
            fb_l = fb_inh[l]
            ratios = jax.vmap(lambda Jb: self._fb_subspace_alignment_ratio(fb_l, Jb))(J[l])
            out.append(jnp.mean(ratios))
        return out

    @partial(jax.jit, static_argnums=(0,))
    def fb_subspace_alignment(self, train_state, batch):
        """Alignment of learned or fixed-random feedback with Jacobian row spaces."""
        vf = self.model.vf
        u0 = vf.get_initial_state_batchexp(batch[0])
        _, OL_state, _ = self.model.openloop(train_state.params, u0, batch[0])

        is_learned, _ = self._parse_mode(self.feedback_mode)
        if is_learned:
            if train_state.Q is None:
                raise ValueError("learned feedback alignment requires train_state.Q")
            fb_inh = vf.apply(
                train_state.params, list(train_state.Q), method=vf.fb_from_Q)
        elif str(self.feedback_mode) == "random":
            if train_state.random_fb is None:
                raise ValueError("random feedback alignment requires train_state.random_fb")
            fb_inh = list(train_state.random_fb)
        else:
            raise ValueError(
                "fb_subspace_alignment supports only learned* or random feedback modes"
            )

        return self._fb_subspace_alignment_from_projected(
            fb_inh, train_state.params, OL_state)

    @partial(jax.jit, static_argnums=(0,))
    def fb_subspace_alignment_ceiling(self, train_state, batch):
        """Per-layer upper bound on the achievable FB-subspace alignment"""
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
        """angle (deg) between the FF grads with learned Q (or random) and the FF grads the analytic Jacobian feedback would apply.
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
        is_learned, rule = self._parse_mode(self.feedback_mode)  # static -> resolved at trace time
        if rule == "two_phase" or not is_learned:
            # two-phase and fixed-random both drive a clean closed loop with a shared
            # feedback matrix; only the source of that matrix differs.
            fb_l = self._applied_fb_weights(train_state, batch)
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
    def ff_update_alignment_floor(self, train_state, batch):
        """lower bound (deg) on the achievable FF-update angle.
        """
        x, y = batch[0], batch[1]
        vf = self.model.vf
        u0 = vf.get_initial_state_batchexp(x)
        OL_y_pred, OL_state, _ = self.model.openloop(train_state.params, u0, x)
        y_targets = self.calculate_targets(OL_y_pred, y)

        # analytic reference: per-sample Jacobian feedback (exactly what BalanceControlled applies)
        fb_a = self.modify_fb_weights(
            self.model.get_fb_weights(train_state.params, OL_state), batch)
        CL_ya, CL_state_a, _ = self.model.closedloop(
            train_state.params, OL_state, x, y_targets, fb_a)
        grads_a = self.get_gradients(
            train_state, x, y, OL_y_pred, CL_ya, OL_state, CL_state_a)

        # floor: best a *shared* feedback can do -> batch-mean Jacobian, broadcast over the batch,
        # then the same clip/norm as the analytic path (identical magnitude to the learned Q).
        bs = x.shape[0]
        fb_raw = self.model.get_fb_weights(train_state.params, OL_state)  # list [bs, dout, sizes_inh]
        fb_bar = [jnp.broadcast_to(jnp.mean(w, axis=0), (bs,) + w.shape[1:]) for w in fb_raw]
        fb_bar = self.modify_fb_weights(fb_bar, batch)
        CL_yb, CL_state_b, _ = self.model.closedloop(
            train_state.params, OL_state, x, y_targets, fb_bar)
        grads_b = self.get_gradients(
            train_state, x, y, OL_y_pred, CL_yb, OL_state, CL_state_b)

        blocks = [f"hidden_{l}" for l in range(vf.nb_hidden)] + ["readout"]
        return [self._angle_deg(grads_b["params"][b], grads_a["params"][b]) for b in blocks]

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
        fb_weights = self._applied_fb_weights(train_state, batch)   # list, each [bs, dout, sizes_inh]
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
