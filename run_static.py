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
    results = dict(datetime=timestr)

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
        is_learned = str(getattr(trainer, "feedback_mode", "analytic")).startswith(
            "learned"
        )
        if not is_learned:
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
            train_state, pre_metrics = trainer.pretrain_epoch(
                train_state, train_data, cfg.batchsize
            )
            con1 = " ".join(
                f"L{l}={float(pre_metrics[f'con1_layer{l}']):.3f}"
                for l in range(model.vf.nb_hidden)
            )
            salign = " ".join(
                f"L{l}={float(pre_metrics[f'salign_layer{l}']):+.3f}"
                for l in range(model.vf.nb_hidden)
            )
            logger.info(
                f"  [Q pretrain] epoch {pre_epoch}/{epochs_pretrain_fb}  "
                f"Condition-1 {con1}  (cos_F {salign})"
            )
        tracker.update(train_state)

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

        # Learned-feedback metrics: Strong-DFC Condition 1 (primary), relative feedback
        # strength (Fig 3C), and the cosine alignment (secondary diagnostic).
        if str(getattr(trainer, "feedback_mode", "analytic")).startswith(
            "learned"
        ) and hasattr(trainer, "condition1"):
            metric_batch = next(iter(train_data))

            # Condition 1 (primary): compliance ratio in [0, 1], 1 = Q lies in row(J).
            con1 = [float(c) for c in trainer.condition1(train_state, metric_batch)]
            con1_mean = float(sum(con1) / len(con1))
            results.setdefault("train_CL_fb_con1_mean", []).append(con1_mean)
            for l, c in enumerate(con1):
                results.setdefault(f"train_CL_fb_con1_layer{l}", []).append(c)
            con1_pl = " ".join(f"L{l}={c:.3f}" for l, c in enumerate(con1))
            logger.info(f"FB Condition-1: mean {con1_mean:.3f}  [{con1_pl}]")

            # Relative feedback strength ||Qu||/||Wr|| (Fig 3C) — informs norm_val.
            fbff = [float(r) for r in trainer.feedback_strength_ratio(train_state, metric_batch)]
            for l, r in enumerate(fbff):
                results.setdefault(f"train_CL_fb_ratio_layer{l}", []).append(r)
            fbff_pl = " ".join(f"L{l}={r:.3f}" for l, r in enumerate(fbff))
            logger.info(f"FB ratio_fb/ff: [{fbff_pl}]")

            # Cosine alignment (secondary; under-reports since Q_ss = Jᵀ M⁻¹).
            aligns = [float(a) for a in trainer.feedback_alignment(train_state, metric_batch)]
            align_mean = float(sum(aligns) / len(aligns))
            results.setdefault("train_CL_fb_align_mean", []).append(align_mean)
            for l, a in enumerate(aligns):
                results.setdefault(f"train_CL_fb_align_layer{l}", []).append(a)
            perlayer = " ".join(f"L{l}={a:+.3f}" for l, a in enumerate(aligns))
            logger.info(f"FB alignment (cos_F): mean {align_mean:+.3f}  [{perlayer}]")

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
