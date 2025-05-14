# Custom FLAX module for Dalian Layer
# Julian Rossbroich
# 2025


from typing import (
    Any,
    Callable,
    Optional,
    Tuple,
    Union,
)

from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen.module import compact
from flax.linen.module import Module
from jax import lax
import jax.numpy as jnp
from jax import custom_jvp

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any  # this could be a real type?
Array = Any
PrecisionLike = Union[
    None,
    str,
    lax.Precision,
    Tuple[str, str],
    Tuple[lax.Precision, lax.Precision],
]
DotGeneralT = Callable[..., Array]
ConvGeneralDilatedT = Callable[..., Array]

default_kernel_init = initializers.lecun_normal()


# Custom jnp.clip function that has the sam forward, but a custom backward 
# pass that returns the identity.
@custom_jvp
def dalian_clip(x, a_min, a_max):
    return jnp.clip(x, a_min, a_max)

dalian_clip.defjvps(lambda g, ans, x, a_min, a_max: g,
             None,
             None)

class DalianDense(Module):
  """
  Like flax.linen.Dense, but applies a Dalian constraint to the kernel.
  """

  features: int
  use_bias: bool = True
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = (
      initializers.zeros
  )
  # Deprecated. Will be removed.
  dot_general: Optional[DotGeneralT] = None
  dot_general_cls: Any = None

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param(
        'kernel',
        self.kernel_init,
        (jnp.shape(inputs)[-1], self.features),
        self.param_dtype,
    )
    if self.use_bias:
      bias = self.param(
          'bias', self.bias_init, (self.features,), self.param_dtype
      )
    else:
      bias = None
    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    if self.dot_general_cls is not None:
      dot_general = self.dot_general_cls()
    elif self.dot_general is not None:
      dot_general = self.dot_general
    else:
      dot_general = lax.dot_general
    y = dot_general(
        inputs,
        dalian_clip(kernel, 0, None),  # Dalian constraint
        (((inputs.ndim - 1,), (0,)), ((), ())),
        precision=self.precision,
    )
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y