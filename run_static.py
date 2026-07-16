# Balance controlled plasticity
# Run script for training & testing models on static datasets
# (e.g. MNIST, F-MNIST)
#
# January 2025
# Author: Julian Rossbroich


# # # # # # # # # # # # # # # # # # #
# IMPORTS
# # # # # # # # # # # # # # # # # # #

# MISC
import os

# LOGGING & RUNTIME
import logging
import time
import json
import numpy as np

# CONFIG
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.utils import instantiate
from hydra.utils import get_original_cwd

# DATA
from flax.training import orbax_utils
import orbax.checkpoint
import shutil

# NUMERIC
import jax
import jax.numpy as jnp

# PROJECT-SPECIFIC
import bcp.utils.runtime_utils as rtutils


# # # # # # # # # # # # # # # # # # #
# SETUP
# # # # # # # # # # # # # # # # # # #


os.environ["HYDRA_FULL_ERROR"] = "1"

# SET UP LOGGER
logging.basicConfig()
logger = logging.getLogger("run_static")

# prevent `nvlink` errors due to CUDA weirdness
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

# remove Jax spam in logger
logging.getLogger("jax._src.lib.xla_bridge").addFilter(lambda _: False)

# remove tensorflow spam in logger
logging.getLogger("absl").setLevel(logging.ERROR)

# # # # # # # # # # # # # # # # # # #
# MAIN FUNCTION
# # # # # # # # # # # # # # # # # # #

OmegaConf.register_new_resolver("orig_cwd", lambda: get_original_cwd())


def _feedback_mode_flags(trainer):
    """Return mode flags used to select fixed feedback diagnostics."""
    mode = str(getattr(trainer, "feedback_mode", "analytic"))
    is_learned = mode.startswith("learned")
    supports_alignment = (
        (is_learned or mode == "random")
        and hasattr(trainer, "fb_subspace_alignment")
    )
    return mode, is_learned, supports_alignment


def _materialize_feedback_metric_batch(train_data):
    """Copy one batch from a separate iterator for reuse throughout a run."""
    return tuple(np.asarray(value).copy() for value in next(iter(train_data)))


def _mean(values):
    return float(sum(values) / len(values))


def _feedback_alignment_snapshot(trainer, train_state, metric_batch, supports_alignment=True):
    """feedback alignment metric snapshot
    """
    snapshot = {}

    if supports_alignment:
        alignment = [
            float(value)
            for value in trainer.fb_subspace_alignment(train_state, metric_batch)
        ]
        snapshot.update(
            alignment=alignment,
            alignment_mean=_mean(alignment),
            ceiling=[],
            ratio=[],
            ratio_mean=0.0,
        )

        if hasattr(trainer, "fb_subspace_alignment_ceiling"):
            ceiling = [
                float(value)
                for value in trainer.fb_subspace_alignment_ceiling(
                    train_state, metric_batch
                )
            ]
            ratio = [value / limit for value, limit in zip(alignment, ceiling)]
            snapshot.update(
                ceiling=ceiling,
                ratio=ratio,
                ratio_mean=_mean(ratio),
            )

        # Relative feedback strength ||Qu||/||Wr|| (Fig 3C) — informs norm_val.
        snapshot["fb_strength"] = [
            float(value)
            for value in trainer.feedback_strength_ratio(train_state, metric_batch)
        ]

        # Functional FF-update angle (deg) vs the analytic-feedback grads.
        if hasattr(trainer, "ff_update_alignment"):
            angles = [
                float(value)
                for value in trainer.ff_update_alignment(train_state, metric_batch)
            ]
            snapshot.update(ff_angle=angles, ff_angle_mean=_mean(angles))

    # Lower bound on the FF-update angle any shared feedback can reach.
    if hasattr(trainer, "ff_update_alignment_floor"):
        floor = [
            float(value)
            for value in trainer.ff_update_alignment_floor(train_state, metric_batch)
        ]
        snapshot.update(ff_angle_floor=floor, ff_angle_floor_mean=_mean(floor))

    return snapshot


def _log_feedback_snapshot(snapshot, labels, phase=""):
    """Log one snapshot's per-layer diagnostics."""
    def _per_layer(values, fmt="{:.3f}", names=None):
        names = names if names is not None else [f"L{l}" for l in range(len(values))]
        return " ".join(
            f"{n}=" + fmt.format(v) for n, v in zip(names, values)
        )

    if "alignment" in snapshot:
        logger.info(
            f"{phase}FB subspace-alignment: mean {snapshot['alignment_mean']:.3f}  "
            f"[{_per_layer(snapshot['alignment'])}]"
        )
        if snapshot["ceiling"]:
            logger.info(
                f"{phase}FB subspace-alignment ceiling: [{_per_layer(snapshot['ceiling'])}]"
            )
            logger.info(
                f"{phase}FB subspace-alignment / ceiling: mean {snapshot['ratio_mean']:.3f}  "
                f"[{_per_layer(snapshot['ratio'])}]"
            )
    if "fb_strength" in snapshot:
        logger.info(f"{phase}FB ratio_fb/ff: [{_per_layer(snapshot['fb_strength'])}]")
    if "ff_angle" in snapshot:
        logger.info(
            f"{phase}FB FF-update angle (deg): mean {snapshot['ff_angle_mean']:.1f}  "
            f"[{_per_layer(snapshot['ff_angle'], '{:.1f}', labels)}]"
        )
    if "ff_angle_floor" in snapshot:
        logger.info(
            f"{phase}FB FF-update angle floor (deg): "
            f"[{_per_layer(snapshot['ff_angle_floor'], '{:.1f}', labels)}]"
        )


def _record_feedback_snapshot(results, snapshot, labels):
    """Append fb diagnostics to the training-history lists in `results`.
    """
    def _append(key, value):
        results.setdefault(key, []).append(value)

    if "alignment" in snapshot:
        _append("train_CL_fb_subalign_mean", snapshot["alignment_mean"])
        for l, value in enumerate(snapshot["alignment"]):
            _append(f"train_CL_fb_subalign_layer{l}", value)

        if snapshot["ceiling"]:
            for l, value in enumerate(snapshot["ceiling"]):
                _append(f"train_CL_fb_subalign_ceil_layer{l}", value)
            _append("train_CL_fb_subalign_ratio_mean", snapshot["ratio_mean"])
            for l, value in enumerate(snapshot["ratio"]):
                _append(f"train_CL_fb_subalign_ratio_layer{l}", value)

    if "fb_strength" in snapshot:
        for l, value in enumerate(snapshot["fb_strength"]):
            _append(f"train_CL_fb_ratio_layer{l}", value)

    if "ff_angle" in snapshot:
        _append("train_CL_fb_ffangle_mean", snapshot["ff_angle_mean"])
        for label, value in zip(labels, snapshot["ff_angle"]):
            _append(f"train_CL_fb_ffangle_{label}", value)

    if "ff_angle_floor" in snapshot:
        _append("train_CL_fb_ffangle_floor_mean", snapshot["ff_angle_floor_mean"])
        for label, value in zip(labels, snapshot["ff_angle_floor"]):
            _append(f"train_CL_fb_ffangle_floor_{label}", value)


def _json_scalar_metrics(metrics):
    """Convert a scalar metric mapping from JAX/NumPy values to Python floats."""
    return {key: float(value) for key, value in metrics.items()}


def _pretrain_feedback_record(epoch, train_metrics, snapshot):
    """Build one JSON-ready feedback-pretraining history entry."""
    return {
        "epoch": int(epoch),
        "train_metrics": _json_scalar_metrics(train_metrics),
        **snapshot,
    }


@hydra.main(version_base=None, config_path="conf", config_name="run_static")
def main(cfg: DictConfig) -> None:
    """
    Main function to train and test models.
    """

    logger.info("Starting a new simulation...")

    # START TIMER
    start_time = time.time()
    timestr = time.strftime("%Y%m%d-%H%M%S")

    # DEVICE SETUP
    # # # # # # # # # # # # # # # # # # #

    # limit jax memory usage
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

    jaxdevice = rtutils.select_device(cfg.device, cfg.gpu_id)

    # Additionally set CUDA_VISIBLE_DEVICES to "" if gpu == False
    # This is a hacky way from preventing jax to use the GPU for jit-compilation with XLA
    if cfg.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # SET PRECISION
    # # # # # # # # # # # # # # # # # # #
    jax.config.update("jax_default_matmul_precision", "float32")

    if cfg.precision == "float16":
        dtype = jnp.float16
    elif cfg.precision == "float32":
        dtype = jnp.float32
    elif cfg.precision == "float64":
        dtype = jnp.float64
        jax.config.update("jax_enable_x64", True)
    else:
        dtype = jnp.bfloat16

    logger.info("Precision: {}".format(dtype))

    # # # # # # # # # # # # # # # # # # #
    # RNG SETUP
    # # # # # # # # # # # # # # # # # # #

    if not cfg.seed:
        rng = int(time.time())
    else:
        rng = int(cfg.seed)

    logger.info("RNG SETUP: using random seed {}".format(rng))

    rng = jax.random.PRNGKey(rng)

    # Derive every RNG stream from the root seed.
    derived_seeds = None
    if cfg.get("derive_seeds", False):
        derived_seeds = rtutils.derive_seeds(int(cfg.seed))
        cfg.dataset.teacher_seed = derived_seeds["teacher"]
        cfg.dataset.data_seed = derived_seeds["data"]
        cfg.model.vf.RNG_Key = derived_seeds["vf"]
        cfg.trainer.q_seed = derived_seeds["q"]
        rng = derived_seeds["init_key"]
        logger.info(
            "RNG SETUP: derived seeds from seed={}: teacher={} data={} vf={} q={}".format(
                cfg.seed,
                derived_seeds["teacher"],
                derived_seeds["data"],
                derived_seeds["vf"],
                derived_seeds["q"],
            )
        )

    # INSTANTIATING MODEL, DATA, TRAINER
    # # # # # # # # # # # # # # # # # # #

    logger.debug("Instantiating dataset...")
    dataset = instantiate(cfg.dataset)

    logger.debug("Instantiating model...")
    model = instantiate(cfg.model, dtype=dtype, vf={"dtype": dtype})

    logger.debug("Instantiating optimizer...")
    optimizer = instantiate(cfg.optimizer)

    logger.debug("Instantiating trainer...")
    trainer = instantiate(cfg.trainer, model, optimizer)

    # # # # # # # # # # # # # # # # # # #
    # GET TRAINING STATE
    # # # # # # # # # # # # # # # # # # #

    # TODO: Add option to load from checkpoint
    train_state = trainer.get_initial_train_state(dataset, rng)

    # # # # # # # # # # # # # # # # # # #
    # MONITORS / TRACKERS
    # # # # # # # # # # # # # # # # # # #

    if cfg.tracking:
        logger.debug("Instantiating tracker...")
        tracker = instantiate(cfg.tracker)

    if cfg.batchwise_tracker:
        from bcp.tracker import BatchWiseSolTracker

        logger.debug("Instantiating debug monitor...")
        batchwise_tracker = BatchWiseSolTracker("batchwise_tracker")

    else:
        batchwise_tracker = None

    # Results dictionary
    results = dict(datetime=timestr, pretrain_fb=[])

    # .hydra/config.yaml is snapshotted before the seeds are derived, so keep them here.
    if derived_seeds is not None:
        results["seeds"] = {
            key: value for key, value in derived_seeds.items() if key != "init_key"
        }

    # # # # # # # # # # # # # # # # # # #
    # LOADING TRAIN / TEST DATA
    # # # # # # # # # # # # # # # # # # #

    dataset = instantiate(cfg.dataset)

    if cfg.OL_eval_on_train:
        train_data, train_data_OL_eval = dataset.get_train_data(
            cfg.batchsize, flatten=model.vf.flatten_input, OL_eval_subset=True
        )
    else:
        train_data = dataset.get_train_data(
            cfg.batchsize, flatten=model.vf.flatten_input
        )

    if dataset.has_valid_data:
        valid_data = dataset.get_valid_data(
            cfg.batchsize, flatten=model.vf.flatten_input
        )

    test_data = dataset.get_test_data(cfg.batchsize, flatten=model.vf.flatten_input)

    _, is_learned_feedback, supports_fb_alignment = (
        _feedback_mode_flags(trainer)
    )
    # The FF-update-angle floor needs no feedback matrix of its own, so it is available
    # even for analytic feedback — keep the metric batch whenever anything is recordable.
    supports_fb_metrics = supports_fb_alignment or hasattr(
        trainer, "ff_update_alignment_floor"
    )
    feedback_metric_batch = (
        _materialize_feedback_metric_batch(train_data)
        if supports_fb_metrics
        else None
    )
    fb_metric_labels = [
        f"L{l}" for l in range(getattr(model.vf, "nb_hidden", 0))
    ] + ["readout"]

    # # # # # # # # # # # # # # # # # # #
    # TRAINING LOOP
    # # # # # # # # # # # # # # # # # # #

    # Initial tracking
    tracker.update(train_state)

    # # # # # # # # # # # # # # # # # # #
    # FEEDBACK (Q) PRE-TRAINING  (FF frozen)
    # # # # # # # # # # # # # # # # # # #

    epochs_pretrain_fb = cfg.get("epochs_pretrain_fb", 0)
    if epochs_pretrain_fb > 0:
        if not is_learned_feedback:
            raise ValueError(
                "epochs_pretrain_fb > 0 requires trainer.feedback_mode=learned* "
                "(e.g. `trainer.feedback_mode=learned model/vf=assemblies-learnedfb`)."
            )
        if not hasattr(trainer, "pretrain_epoch"):
            raise ValueError(
                "The configured trainer does not support feedback pre-training."
            )
        logger.info(
            f"Pre-training feedback weights (Q) for {epochs_pretrain_fb} epochs [FF frozen]..."
        )
        for pre_epoch in range(1, epochs_pretrain_fb + 1):
            train_state, pretrain_metrics = trainer.pretrain_epoch(
                train_state, train_data, cfg.batchsize
            )
            snapshot = _feedback_alignment_snapshot(
                trainer, train_state, feedback_metric_batch, supports_fb_alignment
            )
            logger.info(
                f"  [Q pretrain] epoch {pre_epoch}/{epochs_pretrain_fb}"
            )
            _log_feedback_snapshot(snapshot, fb_metric_labels, phase="  [Q pretrain] ")
            results["pretrain_fb"].append(
                _pretrain_feedback_record(
                    pre_epoch, pretrain_metrics, snapshot
                )
            )
        tracker.update(train_state)
        
        # Reset optimizer state after pretraining
        train_state = trainer.reset_optimizers(train_state)

    logger.info("Starting training...")
    for epoch in range(1, cfg.epochs + 1):
        logger.info(f"Epoch {epoch}/{cfg.epochs}")
        logger.info("")

        logger.info("Training...")
        train_state, train_metrics_CL = trainer.train_epoch(
            train_state, train_data, cfg.batchsize, monitor=batchwise_tracker
        )

        # Evaluation on train (OL), valid & test sets
        if dataset.has_valid_data:
            logger.info("Evaluating on validation data...")
            _, valid_metrics = trainer.eval(train_state, valid_data, cfg.batchsize)

        if cfg.OL_eval_on_train:
            logger.info("Evaluating on train data (OL)...")
            _, train_metrics_OL = trainer.eval(
                train_state, train_data_OL_eval, cfg.batchsize
            )

            logger.info("Evaluating on test data...")
        _, test_metrics = trainer.eval(train_state, test_data, cfg.batchsize)

        # Append metrics to results dictionary
        for k, v in train_metrics_CL.items():
            newk = "train_CL_" + k
            if newk not in results:
                results[newk] = []
            results[newk].append(float(v))
        for k, v in test_metrics.items():
            newk = "test_" + k
            if newk not in results:
                results[newk] = []
            results[newk].append(float(v))
        if dataset.has_valid_data:
            for k, v in valid_metrics.items():
                newk = "valid_" + k
                if newk not in results:
                    results[newk] = []
                results[newk].append(float(v))
        if cfg.OL_eval_on_train:
            for k, v in train_metrics_OL.items():
                newk = "train_OL_" + k
                if newk not in results:
                    results[newk] = []
                results[newk].append(float(v))

        # Log metrics
        logger.info(f"Train loss (CL): {train_metrics_CL.pop('loss'):.3f}")

        if cfg.OL_eval_on_train:
            logger.info(f"Train loss (OL): {train_metrics_OL.pop('loss'):.3f}")

        if dataset.has_valid_data:
            logger.info(f"Valid loss: {valid_metrics.pop('loss'):.3f}")

        logger.info(f"Test loss: {test_metrics.pop('loss'):.3f}")

        # Additional classification metrics
        if cfg.dataset.task == "classification":
            logger.info(
                f"Train accuracy (CL): {train_metrics_CL.pop('accuracy'):.1f} %"
            )

            if cfg.OL_eval_on_train:
                logger.info(
                    f"Train accuracy (OL): {train_metrics_OL.pop('accuracy'):.1f} %"
                )

            if dataset.has_valid_data:
                logger.info(f"Valid accuracy: {valid_metrics.pop('accuracy'):.1f} %")

            logger.info(f"Test accuracy: {test_metrics.pop('accuracy'):.1f} %")

        # Feedback diagnostics: alignment/ceiling/strength/FF-angle for learned or
        # fixed-random feedback, plus the FF-angle floor for every mode.
        if feedback_metric_batch is not None:
            snapshot = _feedback_alignment_snapshot(
                trainer, train_state, feedback_metric_batch, supports_fb_alignment
            )
            _log_feedback_snapshot(snapshot, fb_metric_labels)
            _record_feedback_snapshot(results, snapshot, fb_metric_labels)

        logger.info("")

        # Additional verbose (closed-loop) metrics
        if cfg.verbose:
            for k, v in train_metrics_CL.items():
                logger.info(f"Train {k}: {v:.3f}")

        logger.info(" - - - - - - - - - - - - - - -")

        # Update tracker
        tracker.update(train_state)

        # Check if loss is NaN, if so, end training
        if jnp.isnan(results["valid_loss"][-1]):
            logger.info("Loss is NaN. Ending training.")
            break

    # SAVING RESULTS
    # # # # # # # # # # # # # # # # # # #

    # WALL CLOCK TIME
    end_time = time.time()
    results["wall_time"] = end_time - start_time

    # SAVE RESULTS DICTIONARY (always overwrites)
    logger.info("Saving result dictionary …")
    with open("results.json", "w") as fh:
        json.dump(results, fh, indent=4)

    # SAVE TRACKER (always overwrites)
    logger.info("Saving tracker …")
    tracker.save("tracker.pkl")

    # SAVE CHECKPOINT (Manual overwrite)
    logger.info("Saving train state …")
    ckpt_dir = os.path.abspath("train_state")

    # If it exists, delete it
    if os.path.isdir(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(train_state)
    orbax_checkpointer.save(ckpt_dir, train_state, save_args=save_args)

    # RETURN LOSS FOR OPTUNA
    # # # # # # # # # # # # # # # # # # # #

    # return validation loss if available, otherwise train loss (OL), otherwise train loss (CL)
    if dataset.has_valid_data:
        optuna_loss = results["valid_loss"][-1]
    elif cfg.OL_eval_on_train:
        optuna_loss = results["train_OL_loss"][-1]
    else:
        optuna_loss = results["train_CL_loss"][-1]

    # Ccheck that output is not NaN
    if jnp.isnan(optuna_loss):
        return jnp.inf
    else:
        return optuna_loss


if __name__ == "__main__":
    main()
