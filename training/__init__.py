from catalyst.dl import registry, SupervisedRunner as Runner  # noqa: F401

from experiment import Experiment  # noqa: F401
from model import UNet
from custom_metrics import CustomDiceCallback

registry.Model(UNet)
registry.Callback(CustomDiceCallback)
