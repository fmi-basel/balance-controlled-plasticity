# Balance controlled plasticity
# Run script for M1 lever press learning task
#
# Replicates the experiment from the paper:
# Ren, C. et al.
# Global and subtype-specific modulation of cortical inhibitory neurons regulated by acetylcholine during motor learning.
# Neuron 110, 2334-2350.e8 (2022).
#
# April 2025
# Author: Julian Rossbroich

#  IMPORTS
# # # # # # # # # # #

# MISC
import gc
import os
from inspect import unwrap

# LOGGING & RUNTIME
import logging
import time

# CONFIG
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

# DATA
import pickle

# NUMERIC
import numpy as np
import equinox as eqx
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
from bcp.data.experiments import RenExperiment

# The runner methods already JIT-compile the solve. Avoid Diffrax's nested JIT
diffeqsolve = unwrap(diffeqsolve)

# SETUP
# # # # # # # # # # #

# NO GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Force JAX to use CPU also on Apple Silicon
jax.config.update("jax_platform_name", "cpu")

# SET UP LOGGER
logging.basicConfig()
logger = logging.getLogger("run_motor")


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
        self.rec_dt = rec_dt
        self.T = T
        self.rec_ts = jnp.arange(0, T, rec_dt)

        logger.debug(
            "SimulationRunner initialized with dt=%.5f, rec_dt=%.5f, T=%.3f",
            dt,
            rec_dt,
            T,
        )

    @partial(jit, static_argnums=(0,))
    def __call__(self, state, inputs):
        logger.debug("Performing open-loop simulation call...")

        def f(t, state, etc):
            return self.vectorfield(state, t, inputs)

        odeterm = ODETerm(f)
        sol = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=0,
            t1=self.T,
            dt0=self.dt,
            saveat=SaveAt(ts=self.rec_ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=None,
        )

        final_state = jax.tree.map(lambda x: x[-1], sol.ys)
        logger.debug("Open-loop simulation complete.")
        return final_state, sol

    @partial(jit, static_argnums=(0,))
    def closedloop(self, state, inputs, targets):
        logger.debug("Performing closed-loop simulation call...")

        def f(t, state, etc):
            return self.vectorfield(state, t, inputs, targets, closedloop=True)

        odeterm = ODETerm(f)
        sol = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=0,
            t1=self.T,
            dt0=self.dt,
            saveat=SaveAt(ts=self.rec_ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=None,
        )

        final_state = jax.tree.map(lambda x: x[-1], sol.ys)
        logger.debug("Closed-loop simulation complete.")
        return final_state, sol

    @partial(jit, static_argnums=(0,))
    def learn(self, state, inputs, targets):
        logger.debug("Performing learning iteration (full learning)...")

        def f(t, state, etc):
            return self.vectorfield(
                state, t, inputs, targets, closedloop=True, update_wFF=True, update_wOUT=True
            )

        odeterm = ODETerm(f)
        sol = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=0,
            t1=self.T,
            dt0=self.dt,
            saveat=SaveAt(ts=self.rec_ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=None,
        )

        final_state = jax.tree.map(lambda x: x[-1], sol.ys)
        logger.debug("Learning iteration complete.")
        return final_state, sol


# MAIN FUNCTION
# # # # # # # # # # #
def _run(cfg: DictConfig) -> None:
    logger.info("🌱 Starting MS_Ren")
    logger.info(f"📂 Working directory: {os.getcwd()}")
    logger.debug("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))

    # SET PRECISION
    # # # # # # # # # # # # # # # # # # #
    jax.config.update("jax_default_matmul_precision", "float32")
    logger.debug("JAX precision set to float32.")

    # # # # # # # # # # # # # # # # # # #
    # RNG SETUP
    # # # # # # # # # # # # # # # # # # #

    jax.config.update("jax_threefry_partitionable", cfg.jax_threefry_partitionable)
    logger.info(f"🔑 jax_threefry_partitionable={cfg.jax_threefry_partitionable}")

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
    ts = jnp.linspace(0, cfg.T, timesteps)
    rec_ts = jnp.arange(0, cfg.T, cfg.rec_dt)

    # INSTANTIATING MODEL, DATA, TRAINER
    # # # # # # # # # # # # # # # # # # #
    logger.debug("Instantiating model, data, and runner...")
    key_task, key_init, rng = jax.random.split(rng, 3)

    # Make experiment
    task = RenExperiment(instantiate(cfg.task, key=key_task))

    # Make vectorfield
    vectorfield = instantiate(cfg.model, rng_key=key_init)
    state = vectorfield.get_initial_state()

    # Make sim
    solver = Euler()
    step_size = ConstantStepSize()

    sim = SimulationRunner(vectorfield, solver, step_size, cfg.dt, cfg.rec_dt, cfg.T)
    logger.debug("Instantiation complete.")

    # Shifted and scaled feedforward weights
    state["W_FF"] = state["W_FF"] * 7 + 1 / cfg.task.N
    
    W_IE_frozen = state.pop("W_IE", None)

    # FUNCTIONS FOR TESTING & RECORDING
    # # # # # # # # # # # # # # # # # # #

    def test(sim, state, task, dt, rec_dt, key):
        logger.debug("Running test simulations...")

        key1, key2 = jax.random.split(key)

        # For testing, we present CS only without US input or target
        x, y, x_interp, y_interp = task.simulate(key1)
        state, OL_sol = sim(state, x_interp)
        OL_results = sim.vectorfield.analyze_run(
            x, OL_sol, dt, rec_dt, y, closedloop=False
        )

        # Then we present CS and US input with target
        x, y, x_interp, y_interp = task.simulate(key2)
        state, CL_sol = sim.closedloop(state, x_interp, y_interp)
        CL_results = sim.vectorfield.analyze_run(
            x, CL_sol, dt, rec_dt, y, closedloop=True
        )

        return state, OL_results, CL_results

    # recorded key -> (run, analyze_run output)
    ACTIVITY_KEYS = {
        "OL_rE": ("OL", "rE"),
        "CL_rE": ("CL", "rE"),
        "OL_rI": ("OL", "rI"),
        "CL_rI": ("CL", "rI"),
        "I_FF_bar": ("OL", "I_FF_bar"),
        "OL_I_IE": ("OL", "I_IE"),
        "CL_I_IE": ("CL", "I_IE"),
        "OL_uOut": ("OL", "uOut"),
        "CL_uOut": ("CL", "uOut"),
        "fb": ("CL", "fb"),
    }
    activity_keys = list(ACTIVITY_KEYS) if cfg.rec_activity_keys is None else list(cfg.rec_activity_keys)

    def make_results_dict(vf):
        logger.debug("Making results dictionary...")
        rec_iters = np.arange(0, cfg.train_iterations, cfg.rec_every_Nth_iter)

        # Also record the last iteration
        rec_iters = np.append(rec_iters, cfg.train_iterations)

        # Make result dictionary
        results = {
            "rec_iters": rec_iters,  # Iterations that are recorded
            "OL_R2": None,  # Open-loop R2 (Scalar)
            "CL_R2": None,  # Closed-loop R2 (scalar)
            "OL_loss": None,  # Open-loop loss (scalar)
            "CL_loss": None,  # Closed-loop loss (scalar)
            "OL_error": None,  # Open-loop error (array of shape [Neurons])
            "CL_error": None,  # Closed-loop error (array of shape [Neurons])
        }

        if cfg.rec_activity:
            logger.info("❗ VF activity recording is on! Brace for large outputs.")
            for key in activity_keys:
                results[key] = None

        if cfg.rec_weights:
            results["W_FF"] = None
            results["W_OUT"] = None
            results["B"] = None

        logger.debug("Results dictionary constructed.")
        return rec_iters, results

    def record_results(results, OL_results, CL_results, state, record_idx):
        logger.debug("Recording results from current iteration...")
        values = {
            "OL_R2": OL_results["R2"],
            "CL_R2": CL_results["R2"],
            "OL_loss": OL_results["Loss"],
            "CL_loss": CL_results["Loss"],
            # Compute sum over time of error in hidden layer.
            "OL_error": OL_results["error_hidden"].sum(0),
            "CL_error": CL_results["error_hidden"].sum(0),
        }

        if cfg.rec_activity:
            runs = {"OL": OL_results, "CL": CL_results}
            for key in activity_keys:
                run, name = ACTIVITY_KEYS[key]
                values[key] = runs[run][name]

        if cfg.rec_weights:
            for key in ("W_FF", "W_OUT", "B"):
                values[key] = state[key]

        # Copy directly into the final host arrays. Keeping JAX-array lists and
        # stacking them at the end holds a second copy of the activity history.
        for key, value in values.items():
            value = np.asarray(value)
            if results[key] is None:
                results[key] = np.empty(
                    (len(rec_iters), *value.shape), dtype=value.dtype
                )
            results[key][record_idx] = value

        logger.debug("Results recorded successfully.")
        return results

    # Make result dictionary
    logger.info("📊 Setting up results dictionary...")
    rec_iters, results = make_results_dict(vectorfield)

    key_iter, rng = jax.random.split(rng, 2)

    # Record iteration 0 (before training)
    logger.info("📝 Recording performance before training...")
    start_time = time.perf_counter()
    state, OL_results, CL_results = test(sim, state, task, cfg.dt, cfg.rec_dt, key_iter)
    results = record_results(results, OL_results, CL_results, state, 0)
    record_idx = 1
    lever_minimums = np.empty(cfg.train_iterations, dtype=state["uOut"].dtype)
    elapsed_time = time.perf_counter() - start_time

    logger.info(
        f"[Init] 🔍 Open-loop R2: {np.max(OL_results['R2']):.4f} | 🔒 Closed-loop R2: {np.max(CL_results['R2']):.4f} | 🕒 Time: {elapsed_time:.2f}s"
    )
    del OL_results, CL_results

    logger.info(f"🚀 Training for {cfg.train_iterations} iterations...")
    for idx, iter in enumerate(np.arange(1, cfg.train_iterations + 1)):
        key_iter, rng = jax.random.split(rng)
        x, y, x_interp, y_interp = task.simulate(key_iter)

        start_time = time.perf_counter()
        state, sol = sim.learn(state, x_interp, y_interp)
        iteration_time = time.perf_counter() - start_time

        # Always record minimum output of the trial to calculate
        # whether the lever trace crossed the threshold
        # Get the minimum output of the trial
        lever_trace = jnp.cumsum(sol.ys["uOut"][:, 0]) * cfg.dt
        lever_min = jnp.min(lever_trace)
        lever_minimums[idx] = np.asarray(lever_min)
        del sol, lever_trace, lever_min

        if iter in rec_iters:
            key_test, rng = jax.random.split(rng)
            state, OL_results, CL_results = test(
                sim, state, task, cfg.dt, cfg.rec_dt, key_test
            )
            results = record_results(
                results, OL_results, CL_results, state, record_idx
            )
            record_idx += 1
            logger.info(
                f"📌 Iter {iter:04d} | Open-loop R2: {np.max(OL_results['R2']):.4f} | 🔒 Closed-loop R2: {np.max(CL_results['R2']):.4f} | 🕒 Time: {iteration_time:.2f}s"
            )
            del OL_results, CL_results

    logger.info("✅ Finished training...")
    results["lever_minimums"] = lever_minimums

    logger.info("🔄 Converting results to Numpy arrays...")
    for key in results.keys():
        results[key] = np.asarray(results[key])

    logger.info("💾 Saving results to disk...")
    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    logger.info("💾 Saving final state to disk...")
    if W_IE_frozen is not None:
        state["W_IE"] = W_IE_frozen
    with open("final_state.pkl", "wb") as f:
        pickle.dump(state, f)

    logger.info("💾 Saving vectorfield to disk...")
    with open("vf.pkl", "wb") as f:
        pickle.dump(vectorfield, f)


@hydra.main(version_base=None, config_path="conf", config_name="run_motor")
def main(cfg: DictConfig) -> None:
    try:
        _run(cfg)
    finally:
        # Hydra's basic launcher runs all seeds in one process. Drop compiled
        # functions and their static model instances after every seed, also when
        # a run fails. Clear Equinox before JAX, as recommended by Equinox.
        eqx.clear_caches()
        jax.clear_caches()
        gc.collect()


if __name__ == "__main__":
    main()
