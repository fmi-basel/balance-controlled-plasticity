"""
Module for loading legacy static runs for current analysis.

Some simulations were run with an older version of the codebase
that used the ``incontrolflax`` package (now renamed to ``bcp``). 
This module provides a loader that can adapt the legacy config and checkpoint
to the current codebase.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, open_dict

from .run import HydraRunOutput


_LEGACY_TARGETS = {
    "incontrolflax.data.FashionMNISTDataset": "bcp.data.FashionMNISTDataset",
    "incontrolflax.core.losses.SoftmaxCrossEntropy": (
        "bcp.core.losses.SoftmaxCrossEntropy"
    ),
    "incontrolflax.core.Model": "bcp.core.Model",
    "incontrolflax.core.activation.ReLU": "bcp.core.activation.ReLU",
    "incontrolflax.core.LeakyPIController": "bcp.core.LeakyPIController",
    "incontrolflax.models.lowrank.vf.LowRankExcInhVectorField": (
        "bcp.models.assemblies.vf.ExcInhAssemblyVectorField"
    ),
}

_LEGACY_VECTOR_FIELD_TARGET = (
    "incontrolflax.models.lowrank.vf.LowRankExcInhVectorField"
)
_CURRENT_VECTOR_FIELD_TARGET = (
    "bcp.models.assemblies.vf.ExcInhAssemblyVectorField"
)


class LegacyStaticCompatibilityError(ValueError):
    """Raised when a legacy run cannot be safely loaded for current analysis."""


@dataclass(frozen=True)
class LegacyStaticAnalysisArtifacts:
    """Current analysis objects backed by variables from a legacy checkpoint."""

    config: DictConfig
    dataset: Any
    model: Any
    params: Any


def _rewrite_known_targets(node: Any, path: str) -> None:
    """Rewrite supported ``incontrolflax`` targets below one config subtree."""

    if isinstance(node, DictConfig):
        if "_target_" in node:
            target = str(node["_target_"])
            if target.startswith("incontrolflax."):
                try:
                    replacement = _LEGACY_TARGETS[target]
                except KeyError as exc:
                    raise LegacyStaticCompatibilityError(
                        f"Unsupported legacy Hydra target at {path}: {target!r}. "
                        "The analysis loader only maps targets whose current "
                        "semantics have been verified."
                    ) from exc
                with open_dict(node):
                    node["_target_"] = replacement

        for key, value in node.items():
            _rewrite_known_targets(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for index, value in enumerate(node):
            _rewrite_known_targets(value, f"{path}[{index}]")


def _adapt_legacy_static_config(
    config: DictConfig,
    *,
    dataset_path: str | Path,
) -> DictConfig:
    """Return an analysis-only current config without mutating ``config``."""

    runtime_config = copy.deepcopy(config)

    try:
        dataset_config = runtime_config.dataset
        model_config = runtime_config.model
        vector_field_config = model_config.vf
    except (AttributeError, KeyError) as exc:
        raise LegacyStaticCompatibilityError(
            "Legacy static config must contain dataset, model, and model.vf sections."
        ) from exc

    dataset_path = Path(dataset_path).expanduser().resolve()
    if not dataset_path.is_dir():
        raise LegacyStaticCompatibilityError(
            f"Fashion-MNIST data directory does not exist: {dataset_path}"
        )

    original_vf_target = str(vector_field_config.get("_target_", ""))
    if original_vf_target not in {
        _LEGACY_VECTOR_FIELD_TARGET,
        _CURRENT_VECTOR_FIELD_TARGET,
    }:
        raise LegacyStaticCompatibilityError(
            "Unsupported vector-field target for legacy static analysis: "
            f"{original_vf_target!r}"
        )

    with open_dict(dataset_config):
        dataset_config.path = str(dataset_path)

    _rewrite_known_targets(dataset_config, "dataset")
    _rewrite_known_targets(model_config, "model")

    with open_dict(vector_field_config):
        if "perc_overlap" in vector_field_config:
            legacy_overlap = float(vector_field_config["perc_overlap"])
            converted_overlap = legacy_overlap / 100.0
            if "overlap" in vector_field_config and not jnp.isclose(
                float(vector_field_config["overlap"]), converted_overlap
            ):
                raise LegacyStaticCompatibilityError(
                    "Config contains inconsistent perc_overlap and overlap values."
                )
            vector_field_config["overlap"] = converted_overlap
            del vector_field_config["perc_overlap"]
        elif "overlap" not in vector_field_config:
            raise LegacyStaticCompatibilityError(
                "Legacy vector-field config is missing perc_overlap."
            )

        # These two historical switches were removed.  Their realized values
        # are carried by the checkpoint's saved membership matrices.
        vector_field_config.pop("binary_membership", None)
        vector_field_config.pop("normalize_membership", None)

        vector_field_config.setdefault("g_EI", "default")
        vector_field_config.setdefault("g_XI", 1.0)
        vector_field_config.setdefault("g_EE", 0.0)
        vector_field_config.setdefault("g_II", 0.0)
        vector_field_config.setdefault("inh_deriv_in_jac", True)

    return runtime_config


def _checkpoint_params(checkpoint: Any) -> Any:
    """Validate and extract variables needed for faithful open-loop analysis."""

    if not isinstance(checkpoint, Mapping):
        raise LegacyStaticCompatibilityError(
            "Legacy checkpoint did not restore as a mapping."
        )

    try:
        params = checkpoint["params"]
        trainable_params = params["params"]
        constants = params["constants"]
        memberships = constants["memberships"]
        recurrent_weights = constants["recurrent_weights"]
        memberships["M_E"]
        memberships["M_I"]
        recurrent_weights["W_EI"]
        recurrent_weights["W_IE"]
    except (KeyError, TypeError) as exc:
        raise LegacyStaticCompatibilityError(
            "Legacy checkpoint is missing saved trainable parameters, membership "
            "matrices, or recurrent weights; refusing to regenerate them with the "
            "current implementation."
        ) from exc

    if not isinstance(trainable_params, Mapping):
        raise LegacyStaticCompatibilityError(
            "Legacy checkpoint's params/params entry is not a parameter mapping."
        )

    return params


def load_legacy_static_analysis(
    run: HydraRunOutput,
    *,
    dataset_path: str | Path,
    dtype: Any = jnp.float32,
) -> LegacyStaticAnalysisArtifacts:
    """Load a historical static run for current open-loop analysis.

    The returned model is instantiated from an in-memory migrated config, while
    ``params`` comes directly from the checkpoint.  No optimizer, trainer, or
    model initialization is performed.
    """

    runtime_config = _adapt_legacy_static_config(
        run.config,
        dataset_path=dataset_path,
    )

    dataset = instantiate(runtime_config.dataset)
    model = instantiate(
        runtime_config.model,
        dtype=dtype,
        vf={"dtype": dtype},
    )

    try:
        checkpoint = run.load_trainstate("train_state")
    except Exception as exc:
        raise LegacyStaticCompatibilityError(
            f"Could not restore legacy train_state checkpoint from {run.path}"
        ) from exc

    params = _checkpoint_params(checkpoint)
    return LegacyStaticAnalysisArtifacts(
        config=runtime_config,
        dataset=dataset,
        model=model,
        params=params,
    )
