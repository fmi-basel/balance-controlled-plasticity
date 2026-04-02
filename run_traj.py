# Balance controlled plasticity
# Run script for trajectory learning task
#
# January 2025
# Author: Julian Rossbroich
#
#  IMPORTS
# # # # # # # # # # #

# MISC
import os

# LOGGING & RUNTIME
import logging
import time
from dataclasses import dataclass

# CONFIG
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

# DATA
import pickle

# NUMERIC
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

from functools import partial

from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    Euler,
    ConstantStepSize,
)

# LOCAL

from bcp.utils.trajtask_utils import ShapeTrajectoryTask
from bcp.onlinelearning import SimplePopModel_NoHidden, OnlineLearningMode

# SETUP
# # # # # # # # # # #

# NO GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Force JAX to use CPU also on Apple Silicon
jax.config.update("jax_platform_name", "cpu")

# SET UP LOGGER
logging.basicConfig()
logger = logging.getLogger("run_traj")


# SET UP CLASSES FOR THE TASK
# # # # # # # # # # #


class SimulationRunner:
    def __init__(
        self,
        vectorfield,
        solver,
        stepsize_controller,
        dt: float,
        rec_dt: float,
        T: float,
    ) -> None:

        self.vectorfield = vectorfield
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.dt = dt
        self.T = T
        self.rec_ts = jnp.arange(0, T, rec_dt)
        self.update_wEE = bool(getattr(vectorfield, "eta_EE", 0.0) != 0.0)
        self.mode_open_loop = OnlineLearningMode.open_loop_eval()
        self.mode_closed_loop = OnlineLearningMode.closed_loop_eval()
        self.mode_train_readout = OnlineLearningMode.train_readout()
        self.mode_train_full = OnlineLearningMode.train_full(update_wEE=self.update_wEE)

        logger.debug(
            "SimulationRunner initialized with dt=%.5f, rec_dt=%.5f, T=%.3f",
            dt,
            rec_dt,
            T,
        )

    @partial(jit, static_argnums=(0, 4, 5))
    def _run_segment(
        self,
        state,
        inputs,
        targets,
        mode,
        save_final_only,
        phase_iter=0,
        W_IE_override=None,
    ):
        def f(t, state, etc):
            return self.vectorfield(
                state,
                t,
                inputs,
                targets,
                mode=mode,
                phase_iter=phase_iter,
                W_IE_override=W_IE_override,
            )

        saveat = SaveAt(t1=True) if save_final_only else SaveAt(ts=self.rec_ts)
        odeterm = ODETerm(f)
        sol = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=0,
            t1=self.T,
            dt0=self.dt,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            max_steps=None,
        )

        if save_final_only:
            final_state = jax.tree.map(lambda x: x[0], sol.ys)
        else:
            final_state = jax.tree_map(lambda x: x[-1], sol.ys)

        return final_state, sol

    def __call__(self, state, inputs):
        logger.debug("Performing open-loop simulation call...")
        final_state, sol = self.run_mode(
            state,
            inputs,
            None,
            self.mode_open_loop,
            save_final_only=False,
            phase_iter=0,
        )
        logger.debug("Open-loop simulation complete.")
        return final_state, sol

    def closedloop(self, state, inputs, targets):
        logger.debug("Performing closed-loop simulation call...")
        final_state, sol = self.run_mode(
            state,
            inputs,
            targets,
            self.mode_closed_loop,
            save_final_only=False,
            phase_iter=0,
        )
        logger.debug("Closed-loop simulation complete.")
        return final_state, sol

    def learn(self, state, inputs, targets):
        logger.debug("Performing learning iteration (full learning)...")
        final_state, sol = self.run_mode(
            state,
            inputs,
            targets,
            self.mode_train_full,
            save_final_only=True,
            phase_iter=0,
        )
        logger.debug("Learning iteration complete.")
        return final_state, sol

    def learn_readout(self, state, inputs, targets):
        logger.debug("Performing learning iteration (readout learning only)...")
        final_state, sol = self.run_mode(
            state,
            inputs,
            targets,
            self.mode_train_readout,
            save_final_only=True,
            phase_iter=0,
        )
        logger.debug("Readout-only learning iteration complete.")
        return final_state, sol

    def run_mode(
        self,
        state,
        inputs,
        targets,
        mode,
        save_final_only=True,
        phase_iter=0,
    ):
        # Performance optimization:
        # when W_IE is frozen, keep it outside the ODE state so diffrax
        # does not integrate a large zero-derivative matrix every step.
        if (not mode.update_wIE) and ("W_IE" in state):
            W_IE_override = state["W_IE"]
            state_solver = {key: value for key, value in state.items() if key != "W_IE"}
            final_state, sol = self._run_segment(
                state_solver,
                inputs,
                targets,
                mode,
                save_final_only,
                phase_iter,
                W_IE_override,
            )
            final_state = dict(final_state)
            final_state["W_IE"] = W_IE_override
            return final_state, sol

        return self._run_segment(
            state,
            inputs,
            targets,
            mode,
            save_final_only,
            phase_iter,
        )


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    iterations: int
    mode: OnlineLearningMode


# MAIN FUNCTION
# # # # # # # # # # #
@hydra.main(version_base=None, config_path="conf", config_name="run_traj")
def main(cfg: DictConfig) -> None:

    logger.info("🌱 Starting MS_TrajTask")
    logger.info(f"📂 Working directory: {os.getcwd()}")
    logger.debug("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))

    # SET PRECISION
    # # # # # # # # # # # # # # # # # # #
    jax.config.update("jax_default_matmul_precision", "float32")
    logger.debug("JAX precision set to float32.")

    # # # # # # # # # # # # # # # # # # #
    # RNG SETUP
    # # # # # # # # # # # # # # # # # # #

    if not cfg.seed:
        rng = int(time.time())
    else:
        rng = int(cfg.seed)

    logger.info(f"🔑 RNG setup with seed: {rng}")
    rng = jax.random.PRNGKey(rng)

    # COMPUTE SOME PARAMETERS FROM CFG
    # # # # # # # # # # # # # # # # # # #

    logger.debug("Computing parameters derived from configuration...")
    timesteps = int(cfg.T / cfg.dt)
    input_freq_min = 1 / (cfg.max_period * cfg.T)
    logger.debug(
        "Parameters computed. timesteps=%d, input_freq_min=%.6f",
        timesteps,
        input_freq_min,
    )

    # INSTANTIATING MODEL, DATA, TRAINER
    # # # # # # # # # # # # # # # # # # #
    logger.debug("Instantiating model, data, and runner...")
    key_inputs, key_targets, key_init, rng = jax.random.split(rng, 4)

    # Make inputs
    inputs = instantiate(cfg.inputs, dt=cfg.dt, freq_min=input_freq_min, key=key_inputs)

    # Make target trajectory
    target = instantiate(cfg.trajectory, T=cfg.T, N_points=timesteps, key=key_targets)

    # Make task
    task = ShapeTrajectoryTask(inputs, target)

    # Make vectorfield
    vectorfield = instantiate(cfg.model, rng_key=key_init)
    state = vectorfield.get_initial_state()

    # Make sim
    solver = Euler()
    step_size = ConstantStepSize()

    sim = SimulationRunner(vectorfield, solver, step_size, cfg.dt, cfg.rec_dt, cfg.T)
    logger.debug("Instantiation complete.")

    def parse_mode_preset(name: str):
        if name == "open_loop_eval":
            return OnlineLearningMode.open_loop_eval()
        if name == "closed_loop_eval":
            return OnlineLearningMode.closed_loop_eval()
        if name == "train_readout":
            return OnlineLearningMode.train_readout()
        if name == "train_full":
            return OnlineLearningMode.train_full(update_wEE=sim.update_wEE)
        if name == "train_wie_only":
            return OnlineLearningMode.train_wie_only()
        raise ValueError(f"Unsupported training phase preset: {name}")

    def build_training_phases():
        if (
            "training_phases" in cfg
            and cfg.training_phases is not None
            and len(cfg.training_phases) > 0
        ):
            phases = []
            for idx, phase_cfg in enumerate(cfg.training_phases):
                name = str(phase_cfg.get("name", f"phase_{idx + 1}"))
                iterations = int(phase_cfg.get("iterations", 0))
                if iterations <= 0:
                    raise ValueError(f"Phase '{name}' must have iterations > 0")

                if "preset" in phase_cfg and phase_cfg.preset is not None:
                    mode = parse_mode_preset(str(phase_cfg.preset))
                else:
                    mode = OnlineLearningMode(
                        closedloop=bool(phase_cfg.get("closedloop", False)),
                        update_wFF=bool(phase_cfg.get("update_wFF", False)),
                        update_wOUT=bool(phase_cfg.get("update_wOUT", False)),
                        update_wEE=bool(phase_cfg.get("update_wEE", False)),
                        update_wIE=bool(phase_cfg.get("update_wIE", False)),
                    )
                phases.append(
                    TrainingPhase(name=name, iterations=iterations, mode=mode)
                )
            return phases

        # fallback for single-phase behavior.
        if cfg.train_readout_only:
            logger.info("Training mode (legacy): readout only.")
            return [
                TrainingPhase(
                    name="readout_only",
                    iterations=int(cfg.train_iterations),
                    mode=OnlineLearningMode.train_readout(),
                )
            ]

        logger.info("Training mode (legacy): full network.")
        return [
            TrainingPhase(
                name="full",
                iterations=int(cfg.train_iterations),
                mode=OnlineLearningMode.train_full(update_wEE=sim.update_wEE),
            )
        ]

    training_phases = build_training_phases()
    total_train_iterations = int(sum(phase.iterations for phase in training_phases))

    def is_wie_pretraining_phase(mode: OnlineLearningMode) -> bool:
        return (
            mode.update_wIE
            and not mode.closedloop
            and not mode.update_wFF
            and not mode.update_wOUT
            and not mode.update_wEE
        )

    # FUNCTIONS FOR TESTING & RECORDING
    # # # # # # # # # # # # # # # # # # #

    def test(sim, state, task, dt, rec_dt, run_closedloop=True, phase_iter=0):
        logger.debug("Running test simulations...")

        # Keep evaluation side-effect free with respect to the training stream.
        # Some input generators (e.g. SinusoidalInputs) advance internal time in
        # task.simulate(); we restore it after test-time simulations.
        task_time0 = None
        if hasattr(task, "inputs") and hasattr(task.inputs, "current_time"):
            task_time0 = task.inputs.current_time

        # To make the test fair, we first run one OL simulation to get the network
        # in a state where the controller from the previous iteration is not
        # affecting the results.
        x, y, x_interp, y_interp = task.simulate()
        state_eval, _ = sim(state, x_interp)

        # now we simulate the next 5s of input
        x, y, x_interp, y_interp = task.simulate()

        CL_results = None
        if run_closedloop:
            # We first run the closed-loop test
            # but do not update the state
            _, CL_sol = sim.closedloop(state_eval, x_interp, y_interp)
            CL_results = sim.vectorfield.analyze_run(
                x,
                CL_sol,
                dt,
                rec_dt,
                y,
                closedloop=True,
                phase_iter=phase_iter,
                W_IE_override=state_eval.get("W_IE", None),
            )
            logger.debug("Closed-loop test done.")

        # Now we run the open-loop test on the same piece of input
        # without modifying the training state.
        _, OL_sol = sim(state_eval, x_interp)
        OL_results = sim.vectorfield.analyze_run(
            x,
            OL_sol,
            dt,
            rec_dt,
            y,
            closedloop=False,
            phase_iter=phase_iter,
            W_IE_override=state_eval.get("W_IE", None),
        )
        logger.debug("Open-loop test done.")

        if task_time0 is not None:
            task.inputs.current_time = task_time0

        return OL_results, CL_results

    def make_results_dict(vf, total_iterations):
        logger.debug("Making results dictionary...")
        rec_iters = np.arange(0, total_iterations, cfg.rec_every_Nth_iter)

        # Also record the last iteration
        rec_iters = np.append(rec_iters, total_iterations)

        # Optionally also record the first 50 iterations (for animated plot)
        if cfg.rec_first_50_iters:
            additional_iters = np.arange(0, 50)
            rec_iters = np.array(
                list(np.unique(np.concatenate((rec_iters, additional_iters))))
            )

        # Make result dictionary

        # If the vectorfield has no HL,
        # we only record R2 and loss
        if isinstance(vf, SimplePopModel_NoHidden):
            results = {
                "rec_iters": rec_iters,  # Iterations that are recorded
                "training_time": rec_iters * cfg.T / 60,  # Training time in minutes
                "OL_R2": [],  # Open-loop R2 (Scalar)
                "CL_R2": [],  # Closed-loop R2 (scalar)
                "OL_loss": [],  # Open-loop loss (scalar)
                "CL_loss": [],
                "mean_abs_balance_error": [],
            }  # Closed-loop loss (scalar)

            if cfg.rec_activity:
                logger.info("❗ VF activity recording is on!")
                results["OL_u"] = []
                results["CL_u"] = []

            if cfg.rec_weights:
                results["W"] = []

        else:
            results = {
                "rec_iters": rec_iters,  # Iterations that are recorded
                "training_time": rec_iters * cfg.T / 60,  # Training time in minutes
                "OL_R2": [],  # Open-loop R2 (Scalar)
                "CL_R2": [],  # Closed-loop R2 (scalar)
                "OL_loss": [],  # Open-loop loss (scalar)
                "CL_loss": [],  # Closed-loop loss (scalar)
                "OL_error": [],  # Open-loop error (array of shape [Neurons])
                "CL_error": [],  # Closed-loop error (array of shape [Neurons])
                "error_trace": [],
                "mean_abs_balance_error": [],
            }  # Error trace (array of shape [Neurons])

            if cfg.rec_activity:
                logger.info("❗ VF activity recording is on! Brace for large outputs.")
                results["OL_rE"] = []
                results["CL_rE"] = []
                results["OL_rI"] = []
                results["CL_rI"] = []
                results["I_FF_bar"] = []
                results["OL_I_IE"] = []
                results["CL_I_IE"] = []
                results["OL_uOut"] = []
                results["CL_uOut"] = []

            if cfg.rec_weights:
                results["W_FF"] = []
                results["W_OUT"] = []
                results["B"] = []
                results["g_EE_A"] = []
                if "W_IE" in state:
                    results["W_IE"] = []

        logger.debug("Results dictionary constructed.")
        return rec_iters, results

    def make_empty_results_dict(vf):
        if isinstance(vf, SimplePopModel_NoHidden):
            out = {
                "OL_R2": [],
                "CL_R2": [],
                "OL_loss": [],
                "CL_loss": [],
                "mean_abs_balance_error": [],
            }
            if cfg.rec_activity:
                out["OL_u"] = []
                out["CL_u"] = []
            if cfg.rec_weights:
                out["W"] = []
            return out

        out = {
            "OL_R2": [],
            "CL_R2": [],
            "OL_loss": [],
            "CL_loss": [],
            "OL_error": [],
            "CL_error": [],
            "error_trace": [],
            "mean_abs_balance_error": [],
        }
        if cfg.rec_activity:
            out["OL_rE"] = []
            out["CL_rE"] = []
            out["OL_rI"] = []
            out["CL_rI"] = []
            out["I_FF_bar"] = []
            out["OL_I_IE"] = []
            out["CL_I_IE"] = []
            out["OL_uOut"] = []
            out["CL_uOut"] = []
        if cfg.rec_weights:
            out["W_FF"] = []
            out["W_OUT"] = []
            out["B"] = []
            out["g_EE_A"] = []
            if "W_IE" in state:
                out["W_IE"] = []
        return out

    def record_results(dict, OL_results, CL_results, state):
        logger.debug("Recording results from current iteration...")
        has_closedloop_results = CL_results is not None
        dict["OL_R2"].append(OL_results["R2"])
        dict["CL_R2"].append(CL_results["R2"] if has_closedloop_results else np.nan)
        dict["OL_loss"].append(OL_results["Loss"])
        dict["CL_loss"].append(CL_results["Loss"] if has_closedloop_results else np.nan)

        # only record everything else if the vector field
        # has a hidden layer (otherwise the results are not there)
        if not isinstance(vectorfield, SimplePopModel_NoHidden):
            # compute mean over time of error in hidden layer
            dict["OL_error"].append(OL_results["error_hidden"].sum(0))
            mean_abs_balance_error = float(
                np.mean(np.abs(np.asarray(OL_results["error_hidden"])))
            )
            dict["mean_abs_balance_error"].append(mean_abs_balance_error)
            dict["error_trace"].append(mean_abs_balance_error)
            if has_closedloop_results:
                dict["CL_error"].append(CL_results["error_hidden"].sum(0))
            else:
                dict["CL_error"].append(
                    np.full_like(np.asarray(OL_results["error_hidden"].sum(0)), np.nan)
                )

            if cfg.rec_activity:
                dict["OL_rE"].append(OL_results["rE"])
                dict["CL_rE"].append(
                    CL_results["rE"]
                    if has_closedloop_results
                    else np.full_like(np.asarray(OL_results["rE"]), np.nan)
                )
                dict["OL_rI"].append(OL_results["rI"])
                dict["CL_rI"].append(
                    CL_results["rI"]
                    if has_closedloop_results
                    else np.full_like(np.asarray(OL_results["rI"]), np.nan)
                )
                dict["I_FF_bar"].append(OL_results["I_FF_bar"])
                dict["OL_I_IE"].append(OL_results["I_IE"])
                dict["CL_I_IE"].append(
                    CL_results["I_IE"]
                    if has_closedloop_results
                    else np.full_like(np.asarray(OL_results["I_IE"]), np.nan)
                )
                dict["OL_uOut"].append(OL_results["uOut"])
                dict["CL_uOut"].append(
                    CL_results["uOut"]
                    if has_closedloop_results
                    else np.full_like(np.asarray(OL_results["uOut"]), np.nan)
                )

            if cfg.rec_weights:
                dict["W_FF"].append(state["W_FF"])
                dict["W_OUT"].append(state["W_OUT"])
                dict["B"].append(state["B"])
                dict["g_EE_A"].append(state["g_EE_A"])
                if "W_IE" in dict:
                    dict["W_IE"].append(state["W_IE"])

        else:
            dict["mean_abs_balance_error"].append(np.nan)
            if cfg.rec_activity:
                dict["OL_u"].append(OL_results["u"])
                dict["CL_u"].append(
                    CL_results["u"]
                    if has_closedloop_results
                    else np.full_like(np.asarray(OL_results["u"]), np.nan)
                )

            if cfg.rec_weights:
                dict["W"].append(state["W"])

        logger.debug("Results recorded successfully.")
        return dict

    # Make result dictionary
    logger.info("📊 Setting up results dictionary...")
    rec_iters, results = make_results_dict(vectorfield, total_train_iterations)
    rec_iters_set = set(int(v) for v in rec_iters)
    phase_results = {}
    phase_rec_iters = {}
    phase_final_states = {}
    phase_keys = []
    for idx, phase in enumerate(training_phases):
        phase_key = f"{idx:02d}_{phase.name}"
        phase_keys.append(phase_key)
        phase_results[phase_key] = make_empty_results_dict(vectorfield)
        phase_rec_iters[phase_key] = []

    # Record iteration 0 (before training)
    logger.info("📝 Recording performance before training...")
    init_run_closedloop = True
    if len(training_phases) > 0:
        first_mode = training_phases[0].mode
        init_run_closedloop = not (first_mode.update_wIE and not first_mode.closedloop)
    start_time = time.perf_counter()
    if len(training_phases) > 0 and is_wie_pretraining_phase(training_phases[0].mode):
        init_task_time0 = None
        if hasattr(task, "inputs") and hasattr(task.inputs, "current_time"):
            init_task_time0 = task.inputs.current_time
        x_init, y_init, x_init_interp, _ = task.simulate()
        state_init_eval, init_sol = sim(state, x_init_interp)
        OL_results = sim.vectorfield.analyze_run(
            x_init,
            init_sol,
            cfg.dt,
            cfg.rec_dt,
            y_init,
            closedloop=False,
            phase_iter=0,
            W_IE_override=state_init_eval.get("W_IE", None),
        )
        CL_results = None
        if init_task_time0 is not None:
            task.inputs.current_time = init_task_time0
    else:
        OL_results, CL_results = test(
            sim,
            state,
            task,
            cfg.dt,
            cfg.rec_dt,
            run_closedloop=init_run_closedloop,
        )
    results = record_results(results, OL_results, CL_results, state)
    if len(phase_keys) > 0:
        phase_results[phase_keys[0]] = record_results(
            phase_results[phase_keys[0]], OL_results, CL_results, state
        )
        phase_rec_iters[phase_keys[0]].append(0)
    elapsed_time = time.perf_counter() - start_time

    if CL_results is not None:
        logger.info(
            f"[Init] 🔍 Open-loop R²: {OL_results['R2']:.4f} | 🔒 Closed-loop R²: {CL_results['R2']:.4f} | ⚠️ Loss: {OL_results['Loss']:.6f} | 🕒 Time: {elapsed_time:.2f}s"
        )
    else:
        init_mean_abs_balance_error = float(
            np.mean(np.abs(np.asarray(OL_results["error_hidden"])))
        )
        logger.info(
            f"[Init] 🔍 Open-loop mean |Δ|: {init_mean_abs_balance_error:.6f} | ⚠️ Loss: {OL_results['Loss']:.6f} | 🕒 Time: {elapsed_time:.2f}s"
        )

    logger.info(
        "🚀 Training for %d iterations across %d phase(s)...",
        total_train_iterations,
        len(training_phases),
    )
    global_iter = 0
    for phase_idx, phase in enumerate(training_phases):
        phase_key = phase_keys[phase_idx]
        phase_is_wie_pretrain = is_wie_pretraining_phase(phase.mode)
        logger.info(
            "🔁 Phase '%s' | iters=%d | mode(cl=%s, wFF=%s, wOUT=%s, wEE=%s, wIE=%s)",
            phase.name,
            phase.iterations,
            phase.mode.closedloop,
            phase.mode.update_wFF,
            phase.mode.update_wOUT,
            phase.mode.update_wEE,
            phase.mode.update_wIE,
        )

        for phase_iter in range(1, phase.iterations + 1):
            x, y, x_interp, y_interp = task.simulate()
            iter_idx = phase_iter - 1
            will_record = (global_iter + 1) in rec_iters_set

            start_time = time.perf_counter()
            state, sol = sim.run_mode(
                state,
                x_interp,
                y_interp,
                phase.mode,
                save_final_only=not (phase_is_wie_pretrain and will_record),
                phase_iter=iter_idx,
            )
            if hasattr(vectorfield, "project_state") and (
                phase.mode.update_wEE or phase.mode.update_wIE
            ):
                state = vectorfield.project_state(state, phase_iter=iter_idx)
            jax.tree_util.tree_map(
                lambda x: (
                    x.block_until_ready() if hasattr(x, "block_until_ready") else x
                ),
                state,
            )
            jax.tree_util.tree_map(
                lambda x: (
                    x.block_until_ready() if hasattr(x, "block_until_ready") else x
                ),
                sol,
            )
            iteration_time = time.perf_counter() - start_time

            global_iter += 1
            if global_iter in rec_iters_set:
                run_closedloop_eval = not (
                    phase.mode.update_wIE and not phase.mode.closedloop
                )
                if phase_is_wie_pretrain:
                    OL_results = sim.vectorfield.analyze_run(
                        x, sol, cfg.dt, cfg.rec_dt, y, closedloop=False
                    )
                    CL_results = None
                else:
                    OL_results, CL_results = test(
                        sim,
                        state,
                        task,
                        cfg.dt,
                        cfg.rec_dt,
                        run_closedloop=run_closedloop_eval,
                        phase_iter=iter_idx,
                    )
                results = record_results(results, OL_results, CL_results, state)
                phase_results[phase_key] = record_results(
                    phase_results[phase_key], OL_results, CL_results, state
                )
                phase_rec_iters[phase_key].append(phase_iter)
                iter_time_ms = iteration_time * 1000.0
                if phase_is_wie_pretrain:
                    mean_abs_balance_error = float(
                        np.mean(np.abs(np.asarray(OL_results["error_hidden"])))
                    )
                    logger.info(
                        "📌 Iter %04d/%04d | Phase=%s [%d/%d] | 🔍 Open-loop mean |Δ|: %.6f | 📉 Loss: %.6f | 🕒 Time per iter: %.2fms",
                        global_iter,
                        total_train_iterations,
                        phase.name,
                        phase_iter,
                        phase.iterations,
                        mean_abs_balance_error,
                        OL_results["Loss"],
                        iter_time_ms,
                    )
                elif CL_results is not None:
                    logger.info(
                        "📌 Iter %04d/%04d | Phase=%s [%d/%d] | 🔍 Open-loop R²: %.4f | 🔒 Closed-loop R²: %.4f | 📉 Loss: %.6f | 🕒 Time per iter: %.2fms",
                        global_iter,
                        total_train_iterations,
                        phase.name,
                        phase_iter,
                        phase.iterations,
                        OL_results["R2"],
                        CL_results["R2"],
                        OL_results["Loss"],
                        iter_time_ms,
                    )
                else:
                    logger.info(
                        "📌 Iter %04d/%04d | Phase=%s [%d/%d] | 🔍 Open-loop R²: %.4f | 📉 Loss: %.6f | 🕒 Time per iter: %.2fms",
                        global_iter,
                        total_train_iterations,
                        phase.name,
                        phase_iter,
                        phase.iterations,
                        OL_results["R2"],
                        OL_results["Loss"],
                        iter_time_ms,
                    )
        phase_final_states[phase_key] = {k: v for k, v in state.items()}

    logger.info("✅ Finished training...")

    logger.info("🔄 Converting results to Numpy arrays...")
    for key in results.keys():
        results[key] = np.array(results[key])

    if len(training_phases) > 1:
        for phase_idx, phase_key in enumerate(phase_keys):
            for key in list(phase_results[phase_key].keys()):
                phase_results[phase_key][key] = np.array(phase_results[phase_key][key])
            rec_iters_phase = np.array(phase_rec_iters[phase_key], dtype=int)
            phase_results[phase_key]["rec_iters"] = rec_iters_phase
            phase_results[phase_key]["training_time"] = rec_iters_phase * cfg.T / 60
            phase_results[phase_key]["phase_name"] = training_phases[phase_idx].name

    logger.info("💾 Saving results to disk...")
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    if len(training_phases) > 1:
        logger.info("💾 Saving per-phase results to disk...")
        with open("phase_results.pkl", "wb") as f:
            pickle.dump(phase_results, f)

    logger.info("💾 Saving final state to disk...")
    with open("final_state.pkl", "wb") as f:
        pickle.dump(state, f)

    if len(training_phases) > 1:
        logger.info("💾 Saving per-phase final states to disk...")
        with open("phase_final_states.pkl", "wb") as f:
            pickle.dump(phase_final_states, f)

    logger.info("💾 Saving vectorfield to disk...")
    with open("vf.pkl", "wb") as f:
        pickle.dump(vectorfield, f)


if __name__ == "__main__":
    main()
