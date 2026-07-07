"""Online-learning vector fields for balance-controlled plasticity.

This package was previously the single module ``bcp/onlinelearning.py``. It has
been split into sub-modules (``modes``, ``feedback``, ``assembly_vf``,
``simple_model``); the public names are re-exported here so existing imports
such as ``from bcp.onlinelearning import ExcInhAssemblyOnlineLearningVF``
continue to work unchanged.
"""

from .modes import OnlineLearningMode
from .assembly_vf import ExcInhAssemblyOnlineLearningVF
from .simple_model import SimplePopModel_NoHidden

__all__ = [
    "OnlineLearningMode",
    "ExcInhAssemblyOnlineLearningVF",
    "SimplePopModel_NoHidden",
]
