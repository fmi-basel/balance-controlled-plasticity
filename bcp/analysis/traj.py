# Rebuild the task and vectorfield of a run_traj.py run from its saved config.

from pathlib import Path

import jax
from hydra.utils import instantiate

from bcp.utils.trajtask_utils import ShapeTrajectoryTask


def rebuild_traj_run(run, shapes_dir, jax_threefry_partitionable=False):
    """Return (cfg, task, vectorfield) of a simulation run, as at the start of training.

    Uses the same key splits as run_traj.py and sets `jax_threefry_partitionable`
    to the value the run used. Runs whose config predates that key fall back to the
    argument: False for jax < 0.5 (runs before 2026-07-08), True for later runs.
    """
    cfg = run.config
    cfg.trajectory.path = str(Path(shapes_dir))

    partitionable = cfg.get("jax_threefry_partitionable", jax_threefry_partitionable)
    jax.config.update("jax_threefry_partitionable", bool(partitionable))

    timesteps = int(cfg.T / cfg.dt)
    key_inputs, key_targets, key_init, _ = jax.random.split(
        jax.random.PRNGKey(int(cfg.seed)), 4
    )
    inputs = instantiate(
        cfg.inputs, dt=cfg.dt, freq_min=1 / (cfg.max_period * cfg.T), key=key_inputs
    )
    target = instantiate(cfg.trajectory, T=cfg.T, N_points=timesteps, key=key_targets)
    vectorfield = instantiate(cfg.model, rng_key=key_init)

    return cfg, ShapeTrajectoryTask(inputs, target), vectorfield
