from catalyst.dl import registry

from .experiment import Experiment
from .model import UNet
from .runner import NeuroRunner as Runner

registry.Model(UNet)
