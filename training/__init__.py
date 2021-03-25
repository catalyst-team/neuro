try:
    from catalyst.dl import registry, SupervisedRunner as Runner  # noqa: F401
    from experiment import Experiment  # noqa: F401
    from model import UNet, MeshNet
    from callbacks import CustomDiceCallback, NiftiInferCallback

    registry.Model(UNet)
    registry.Model(MeshNet)
    registry.Callback(CustomDiceCallback)

except ImportError:
    print ("Catalyst not found. Loading production environment")
