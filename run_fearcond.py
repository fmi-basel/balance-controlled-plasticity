# Balance controlled plasticity
# Run script for Fear-conditioning task.
#
# Replicates the experiment from the paper:
# Krabbe, S. et al. 
# Adaptive disinhibitory gating by VIP interneurons permits associative learning. 
# Nature neuroscience 22, 1834–1843 (2019).
#
# 
# April 2025
# Author: Julian Rossbroich

#  IMPORTS
# # # # # # # # # # # 

# MISC
import os

# LOGGING & RUNTIME
import logging
import time

# CONFIG
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

# DATA
import pickle
from collections import namedtuple


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


# PLOTTING
import matplotlib as mpl 
import matplotlib.pyplot as plt
mpl.use('Agg') # non-interactive mpl backend
mpl.style.use('spiffy')

# LOCAL
from bcp.data.experiments import KrabbeExperiment

# SETUP
# # # # # # # # # # #

# NO GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Force JAX to use CPU also on Apple Silicon
jax.config.update("jax_platform_name", "cpu")

# SET UP LOGGER
logging.basicConfig()
logger = logging.getLogger("run_fearcond")


# SET UP CLASSES FOR THE TASK
# # # # # # # # # # #

# Define a namedtuple to replace the custom class
CustomDiffraxSol = namedtuple("CustomDiffraxSol", ["t0", "t1", "ys", "ts"])


def merge_solutions(s1, s2, s3):
    """
    Concatenate diffrax solutions
    """
    t0 = s1.t0
    t1 = s3.t1
    ys = jax.tree_map(lambda a, b, c: jnp.concatenate([a, b, c], axis=0), s1.ys, s2.ys, s3.ys)
    ts = jax.tree_map(lambda a, b, c: jnp.concatenate([a, b, c], axis=0), s1.ts, s2.ts, s3.ts)

    return CustomDiffraxSol(t0, t1, ys, ts)

    
class SimulationRunner:
    
    def __init__(self, 
                 vectorfield, 
                 solver, 
                 stepsize_controller,
                 dt: float,
                 rec_dt: float,
                 T: float,
                 T_CS: float,
                 T_US: float,
                 T_ITI: float
                 ) -> None:

        self.vectorfield = vectorfield
        self.solver = solver
        self.stepsize_controller = stepsize_controller
        self.dt = dt
        self.rec_dt = rec_dt
        self.T = T
        self.T_CS = T_CS
        self.T_US = T_US
        self.T_ITI = T_ITI
        self.rec_ts = jnp.arange(0, T, rec_dt)
        
        logger.debug("SimulationRunner initialized with dt=%.5f, rec_dt=%.5f, T=%.3f", dt, rec_dt, T)
        
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
            max_steps=None
        )
        
        final_state = jax.tree_map(lambda x: x[-1], sol.ys)
        logger.debug("Open-loop simulation complete.")
        return final_state, sol
    
    @partial(jit, static_argnums=(0,))
    def closedloop(self, state, inputs, targets):
        logger.debug("Performing closed-loop simulation call...")        
        
        # PHASE 1: CS presentation 
        # Network behaves as in open-loop (Feedback control is off)
        def f(t, state, etc):
            return self.vectorfield(state, t, inputs)
        
        saveat_ts = jnp.arange(0, self.T_CS, self.rec_dt)

        odeterm = ODETerm(f)
        sol_CS = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=0,
            t1=self.T_CS,
            dt0=self.dt,
            saveat=SaveAt(ts=saveat_ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=None
        )
        state = jax.tree_map(lambda x: x[-1], sol_CS.ys)
        
        # PHASE 2: US presentation
        # Network behaves as in closed-loop (Feedback control is on)
        def f(t, state, etc):
            return self.vectorfield(state, t, inputs, targets, True, False, False)
        
        saveat_ts = jnp.arange(self.T_CS, self.T_CS + self.T_US, self.rec_dt)
        
        odeterm = ODETerm(f)
        sol_US = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=self.T_CS,
            t1=self.T_CS + self.T_US,
            dt0=self.dt,
            saveat=SaveAt(ts=saveat_ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=None
        )
        state = jax.tree_map(lambda x: x[-1], sol_US.ys)
        
        # PHASE 3: ITI (no CS or US)
        # Network behaves as in open-loop (Feedback control is off)
        def f(t, state, etc):
            return self.vectorfield(state, t, inputs, None, False, True, False)
        
        saveat_ts = jnp.arange(self.T_CS + self.T_US, self.T, self.rec_dt)
        
        odeterm = ODETerm(f)
        sol_ITI = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=self.T_CS + self.T_US,
            t1=self.T,
            dt0=self.dt,
            saveat=SaveAt(ts=saveat_ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=None
        )
        
        # Concatenate solutions
        sol = merge_solutions(sol_CS, sol_US, sol_ITI)
        
        final_state = jax.tree_map(lambda x: x[-1], sol_ITI.ys)
        logger.debug("Closed-loop simulation complete.")
        return final_state, sol
    
    @partial(jit, static_argnums=(0,))
    def learn(self, state, inputs, targets):
        logger.debug("Performing learning iteration (hidden only)...")

        # PHASE 1: CS presentation 
        # Network behaves as in open-loop (Feedback control is off)
        def f(t, state, etc):
            return self.vectorfield(state, t, inputs, None, False, True, False)
        
        saveat_ts = jnp.arange(0, self.T_CS, self.rec_dt)

        odeterm = ODETerm(f)
        sol_CS = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=0,
            t1=self.T_CS,
            dt0=self.dt,
            saveat=SaveAt(ts=saveat_ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=None
        )
        state = jax.tree_map(lambda x: x[-1], sol_CS.ys)
        
        # PHASE 2: US presentation
        # Network behaves as in closed-loop (Feedback control is on)
        def f(t, state, etc):
            return self.vectorfield(state, t, inputs, targets, True, True, False)
        
        saveat_ts = jnp.arange(self.T_CS, self.T_CS + self.T_US, self.rec_dt)
        
        odeterm = ODETerm(f)
        sol_US = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=self.T_CS,
            t1=self.T_CS + self.T_US,
            dt0=self.dt,
            saveat=SaveAt(ts=saveat_ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=None
        )
        state = jax.tree_map(lambda x: x[-1], sol_US.ys)
        
        # PHASE 3: ITI (no CS or US)
        # Network behaves as in open-loop (Feedback control is off)
        def f(t, state, etc):
            return self.vectorfield(state, t, inputs, None, False, True, False)
        
        saveat_ts = jnp.arange(self.T_CS + self.T_US, self.T, self.rec_dt)
        
        odeterm = ODETerm(f)
        sol_ITI = diffeqsolve(
            odeterm,
            self.solver,
            y0=state,
            t0=self.T_CS + self.T_US,
            t1=self.T,
            dt0=self.dt,
            saveat=SaveAt(ts=saveat_ts),
            stepsize_controller=self.stepsize_controller,
            max_steps=None
        )
        
        # Concatenate solutions
        sol = merge_solutions(sol_CS, sol_US, sol_ITI)
        
        final_state = jax.tree_map(lambda x: x[-1], sol_ITI.ys)
        logger.debug("Learning iteration complete.")
        return final_state, sol


# MAIN FUNCTION
# # # # # # # # # # #
@hydra.main(version_base=None, config_path="conf", config_name="run_fearcond")
def main(cfg: DictConfig) -> None:
    
    logger.info("🌱 Starting MS_Krabbe")
    logger.info(f"📂 Working directory: {os.getcwd()}")
    logger.debug("Loaded configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    # SET PRECISION
    # # # # # # # # # # # # # # # # # # #
    jax.config.update('jax_default_matmul_precision', 'float32')
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
    ts = jnp.linspace(0, cfg.T, timesteps)
    rec_ts = jnp.arange(0, cfg.T, cfg.rec_dt)

    # INSTANTIATING MODEL, DATA, TRAINER
    # # # # # # # # # # # # # # # # # # #
    logger.debug("Instantiating model, data, and runner...")
    key_task, key_init, rng = jax.random.split(rng, 3)
    
    # Make experiment
    task = KrabbeExperiment(instantiate(cfg.task, key=key_task))
    
    # Make vectorfield
    vectorfield = instantiate(cfg.model, rng_key=key_init)
    state = vectorfield.get_initial_state()
    
    # Make sim
    solver = Euler()
    step_size = ConstantStepSize()
        
    sim = SimulationRunner(vectorfield, solver, step_size, cfg.dt, cfg.rec_dt, cfg.T, cfg.task.T_CS, cfg.task.T_US, cfg.task.T_ITI)
    logger.debug("Instantiation complete.")
    
    # Choose learning mode (Fixed to hidden only learning)
    learn_func = sim.learn
    
    # MODIFY INITIAL STATE
    # # # # # # # # # # # # # # # # # # #

    # Upscale output weights
    state['W_OUT'] *= 10
    
    # Make sure outputs sum to 0
    state["W_OUT"] = state["W_OUT"] - jnp.mean(state["W_OUT"])
    
    # Shifted and scaled feedforward weights
    state['W_FF'] = state['W_FF'] * 1 + 1 / cfg.task.N
    
    # OUTPUT NONLINEARITY
    # # # # # # # # # # # # # # # # # # #
    out_nl = vectorfield.controller.loss.sigmoid

    # FUNCTIONS FOR TESTING & RECORDING
    # # # # # # # # # # # # # # # # # # #
        
    def test(sim, state, task, dt, rec_dt, key):
        logger.debug("Running test simulations...")
        
        key1, key2 = jax.random.split(key)
        
        # For testing, we present CS only without US input or target
        x, y, x_interp, y_interp = task.simulate_test(key1)
        _, OL_sol = sim(state, x_interp)
        OL_results = sim.vectorfield.analyze_run(x, OL_sol, dt, rec_dt, y, closedloop=False)
        OL_results['freeze'] = out_nl(OL_results['uOut'])
        
        # Then we present CS and US input with target
        #x, y, x_interp, y_interp = task.simulate(key2)
        _, CL_sol = sim.closedloop(state, x_interp, y_interp)
        freezing_CL = out_nl(CL_sol.ys['uOut'][:,:,0])
        
        # the analyze_run function does not support closed-loop only during US presentation
        # so we get around this by setting y_target to the output of the closed-loop simulation
        # everywhere except during the US presentation
        y_for_analysis = y
        y_for_analysis.at[:task.task.timesteps_CS,:].set(freezing_CL[:task.task.timesteps_CS,:])
        y_for_analysis.at[task.task.timesteps_CS+task.task.timesteps_US:,:].set(freezing_CL[task.task.timesteps_CS+task.task.timesteps_US:,:])
                
        CL_results = sim.vectorfield.analyze_run(x, CL_sol, dt, rec_dt, y_for_analysis, closedloop=True)
        CL_results['freeze'] = out_nl(CL_results['uOut'])
        
        return state, OL_results, CL_results
    
    def make_results_dict(vf):
        logger.debug("Making results dictionary...")
        rec_iters = np.arange(0, cfg.train_iterations+1)
        
        # Make result dictionary
        results = {'rec_iters': rec_iters,                   # Iterations that are recorded
                'OL_R2': [],                                 # Open-loop R2 (Scalar)
                'CL_R2': [],                                 # Closed-loop R2 (scalar)
                'OL_loss': [],                               # Open-loop loss (scalar)
                'CL_loss': [],                               # Closed-loop loss (scalar)              
                'OL_error': [],                              # Open-loop error (array of shape [Neurons])
                'CL_error': []}                              # Closed-loop error (array of shape [Neurons])
        
        if cfg.rec_activity:
            logger.info("❗ VF activity recording is on! Brace for large outputs.")
            results['OL_rE'] = []
            results['CL_rE'] = []
            results['OL_rI'] = []
            results['CL_rI'] = []
            results['I_FF_bar'] = []
            results['OL_I_IE'] = []
            results['CL_I_IE'] = [] 
            results['OL_uOut'] = []
            results['CL_uOut'] = []
            results['OL_freeze'] = []
            results['CL_freeze'] = []
            results['fb'] = []
            
        if cfg.rec_weights:
            results['W_FF'] = []
            results['W_OUT'] = []
            results['B'] = []
        
        logger.debug("Results dictionary constructed.")
        return rec_iters, results
    

    def record_results(dict, OL_results, CL_results, state):
        logger.debug("Recording results from current iteration...")
        dict['OL_R2'].append(OL_results['R2'])
        dict['CL_R2'].append(CL_results['R2'])
        dict['OL_loss'].append(OL_results['Loss'])
        dict['CL_loss'].append(CL_results['Loss'])
        
        # compute sum over time of error in hidden layer
        dict['OL_error'].append(OL_results['error_hidden'].sum(0))
        dict['CL_error'].append(CL_results['error_hidden'].sum(0))
            
        if cfg.rec_activity:
            dict['OL_rE'].append(OL_results['rE'])
            dict['CL_rE'].append(CL_results['rE'])
            dict['OL_rI'].append(OL_results['rI'])
            dict['CL_rI'].append(CL_results['rI'])
            dict['I_FF_bar'].append(OL_results['I_FF_bar'])
            dict['OL_I_IE'].append(OL_results['I_IE'])
            dict['CL_I_IE'].append(CL_results['I_IE'])
            dict['OL_uOut'].append(OL_results['uOut'])
            dict['CL_uOut'].append(CL_results['uOut'])
            dict['OL_freeze'].append(out_nl(OL_results['uOut']))
            dict['CL_freeze'].append(out_nl(CL_results['uOut']))
            dict['fb'].append(CL_results['fb'])
        
        if cfg.rec_weights:
            dict['W_FF'].append(state['W_FF'])
            dict['W_OUT'].append(state['W_OUT'])
            dict['B'].append(state['B'])
                
        logger.debug("Results recorded successfully.")
        return dict
    
    # Make result dictionary
    logger.info("📊 Setting up results dictionary...")
    rec_iters, results = make_results_dict(vectorfield)

    key_iter, rng = jax.random.split(rng, 2)
    
    # Record iteration 0 (before training)
    logger.info("📝 Recording performance before training...")
    start_time = time.perf_counter()
    state, OL_results, CL_results = test(sim, state, task, cfg.dt, cfg.rec_dt, key_iter)
    results = record_results(results, OL_results, CL_results, state)
    elapsed_time = time.perf_counter() - start_time
    
    # plot OL and CL outputs
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(rec_ts, out_nl(OL_results['uOut']), label='Open-loop')
    ax.plot(rec_ts, out_nl(CL_results['uOut']), label='Closed-loop')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Output')
    ax.legend()
    plt.savefig(f'outputs_init.png')
    plt.close()

    logger.info(f"[Init] 🔍 Open-loop freeze: {np.max(OL_results['freeze']):.4f} | 🔒 Closed-loop freeze: {np.max(CL_results['freeze']):.4f} | 🕒 Time: {elapsed_time:.2f}s")

    logger.info(f"🚀 Training for {cfg.train_iterations} iterations...")
    for idx, iter in enumerate(np.arange(1, cfg.train_iterations + 1)):
        
        key_iter, rng = jax.random.split(rng)
        x, y, x_interp, y_interp = task.simulate(key_iter)
        
        start_time = time.perf_counter()
        state, sol = learn_func(state, x_interp, y_interp)
        iteration_time = time.perf_counter() - start_time
        
        if iter in rec_iters:
            key_test, rng = jax.random.split(rng)
            _, OL_results, CL_results = test(sim, state, task, cfg.dt, cfg.rec_dt, key_test)
            results = record_results(results, OL_results, CL_results, state)            
            logger.info(f"📌 Iter {iter:04d} | Open-loop freeze: {np.max(OL_results['freeze']):.4f} | 🔒 Closed-loop freeze: {np.max(CL_results['freeze']):.4f} | 🕒 Time: {iteration_time:.2f}s")
            
            # plot OL and CL outputs
            fig, ax = plt.subplots(1, 1, figsize=(6, 3))
            ax.plot(rec_ts, out_nl(OL_results['uOut']), label='Open-loop')
            ax.plot(rec_ts, out_nl(CL_results['uOut']), label='Closed-loop')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Output')
            ax.legend()
            plt.savefig(f'outputs_{iter:04d}.png')
            plt.close()
            
    logger.info("✅ Finished training...")

    logger.info("🔄 Converting results to Numpy arrays...")
    for key in results.keys():
        results[key] = np.array(results[key])

    logger.info("💾 Saving results to disk...")
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    logger.info("💾 Saving final state to disk...")
    with open('final_state.pkl', 'wb') as f:
        pickle.dump(state, f)
        
    logger.info("💾 Saving vectorfield to disk...")
    with open('vf.pkl', 'wb') as f:
        pickle.dump(vectorfield, f) 
        
if __name__ == "__main__":
    main()
    