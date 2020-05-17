from catalyst.dl import registry

from .experiment import Experiment  # noqa: F401
from .model import UNet
from .runner import NeuroRunner as Runner  # noqa: F401

registry.Model(UNet)
