from catalyst.dl import registry, 
from catalyst.dl SupervisedRunner as Runner  # noqa: F401
from .experiment import Experiment  # noqa: F401
from .model import UNet

registry.Model(UNet)
