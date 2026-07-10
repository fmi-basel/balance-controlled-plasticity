# Nonlinear student-teacher regression task
# Julian Rossbroich
# 2026


import jax
import jax.numpy as jnp
import numpy as np

from .dataloader import DatasetLoader

from typing import Optional

# logging
import logging

logger = logging.getLogger(__name__)


_ACTIVATIONS = {
    "tanh": jnp.tanh,
    "relu": jax.nn.relu,
    "sigmoid": jax.nn.sigmoid,
    "identity": lambda x: x,
    "linear": lambda x: x,
    "none": lambda x: x,
}


def _build_teacher(key, dims, gain):
    """Random MLP weights, one matrix per layer, LeCun-scaled by fan-in."""
    keys = jax.random.split(key, len(dims) - 1)
    weights = []
    for k, (din, dout) in zip(keys, zip(dims[:-1], dims[1:])):
        W = jax.random.normal(k, (din, dout)) * (gain / jnp.sqrt(din))
        weights.append(W)
    return weights


def _teacher_forward(weights, x, activation):
    """Nonlinear hidden layers, linear readout."""
    h = x
    for i, W in enumerate(weights):
        h = h @ W
        if i < len(weights) - 1:
            h = activation(h)
    return h


class StudentTeacherDataset(DatasetLoader):
    """Synthetic nonlinear student-teacher regression dataset.

    The teacher is a fixed random MLP with dims [dim_input, *teacher_hidden,
    dim_output], nonlinear (default tanh) hidden layers and a linear readout.
    Targets can be standardized (zero-mean/unit-var over the training
    set) so MSE loss and thus controller operate in an O(1) range.
    """

    task: str = "regression"
    name: str = "StudentTeacher"
    has_valid_data: bool = True

    def __init__(
        self,
        dim_input: int = 30,
        dim_output: int = 5,
        teacher_hidden=(10, 10, 10),
        n_train: int = 500,
        n_valid: int = 500,
        n_test: int = 1000,
        teacher_activation: str = "tanh",
        teacher_gain: float = 1.0,
        standardize_targets: bool = True,
        teacher_seed: int = 0,
        data_seed: int = 1,
        task: str = "regression",
        name: str = "StudentTeacher",
        OL_eval_subset_split: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            dim_input=dim_input, dim_output=dim_output, task=task, name=name
        )

        self.has_valid_data = True
        self.OL_eval_subset_split = OL_eval_subset_split
        self.n_train = int(n_train)
        self.n_valid = int(n_valid)
        self.n_test = int(n_test)
        self.standardize_targets = bool(standardize_targets)

        # Resolve the teacher activation (accept OmegaConf strings).
        act_name = str(teacher_activation).lower()
        if act_name not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown teacher_activation '{teacher_activation}'. "
                f"Options: {sorted(_ACTIVATIONS)}"
            )
        activation = _ACTIVATIONS[act_name]

        # Build the fixed teacher (deterministic in teacher_seed).
        dims = [int(dim_input), *[int(h) for h in teacher_hidden], int(dim_output)]
        self.teacher_dims = dims
        weights = _build_teacher(jax.random.PRNGKey(int(teacher_seed)), dims, float(teacher_gain))

        # Sample inputs and teacher targets (deterministic in data_seed).
        rng = jax.random.PRNGKey(int(data_seed))
        rng_tr, rng_va, rng_te = jax.random.split(rng, 3)

        def make_split(r, n):
            x = jax.random.normal(r, (n, int(dim_input)))
            y = _teacher_forward(weights, x, activation)
            return x, y

        x_train, y_train = make_split(rng_tr, self.n_train)
        x_valid, y_valid = make_split(rng_va, self.n_valid)
        x_test, y_test = make_split(rng_te, self.n_test)

        # Standardize targets using TRAIN statistics only (keeps the target
        # function fixed; applied identically to every split).
        if self.standardize_targets:
            self.target_mean = jnp.mean(y_train, axis=0)
            self.target_std = jnp.std(y_train, axis=0) + 1e-8
            y_train = (y_train - self.target_mean) / self.target_std
            y_valid = (y_valid - self.target_mean) / self.target_std
            y_test = (y_test - self.target_mean) / self.target_std
        else:
            self.target_mean = jnp.zeros((int(dim_output),))
            self.target_std = jnp.ones((int(dim_output),))

        # Store as numpy float32 (matches the tfds.as_numpy pipeline).
        self.x_train = np.asarray(x_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.float32)
        self.x_valid = np.asarray(x_valid, dtype=np.float32)
        self.y_valid = np.asarray(y_valid, dtype=np.float32)
        self.x_test = np.asarray(x_test, dtype=np.float32)
        self.y_test = np.asarray(y_test, dtype=np.float32)

        logger.info(
            f"StudentTeacher: teacher dims {dims} ({act_name}), "
            f"{self.n_train}/{self.n_valid}/{self.n_test} train/valid/test samples, "
            f"standardize_targets={self.standardize_targets}"
        )

    def _batch(self, x, y, batchsize):
        n = x.shape[0]
        if batchsize is None:
            batchsize = n
        batchsize = int(batchsize)
        nb = n // batchsize
        if nb == 0:
            raise ValueError(
                f"batchsize {batchsize} larger than the {n}-sample split."
            )
        return [
            (x[i * batchsize : (i + 1) * batchsize], y[i * batchsize : (i + 1) * batchsize])
            for i in range(nb)
        ]

    def get_train_data(self, batchsize, rng=None, flatten=False, OL_eval_subset=False):
        batches = self._batch(self.x_train, self.y_train, batchsize)
        logger.info(f"Number of samples in training data: {len(batches) * int(batchsize)}")
        if OL_eval_subset:
            k = max(1, int(len(batches) * self.OL_eval_subset_split))
            return batches, batches[:k]
        return batches

    def get_valid_data(self, batchsize=None, flatten=False):
        return self._batch(self.x_valid, self.y_valid, batchsize)

    def get_test_data(self, batchsize=None, flatten=False):
        return self._batch(self.x_test, self.y_test, batchsize)

    def get_mock_data(
        self,
        batchsize: Optional[int] = None,
        rng=None,
        flatten: bool = False,
    ) -> tuple:
        """A single un-batched (x, y) example for model.init (correct shape/dtype)."""
        if batchsize is None:
            return self.x_train[0], self.y_train[0]
        return self.x_train[:batchsize], self.y_train[:batchsize]
