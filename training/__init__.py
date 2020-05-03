# flake8: noqa
# isort:skip_file

from catalyst.dl import registry
from catalyst.dl import SupervisedRunner as Runner
from .experiment import Experiment
from .model import UNet

registry.Model(UNet)
