try:
    from catalyst.registry import Registry
    from model import UNet, MeshNet
    from callbacks import CustomDiceCallback
    from runner import CustomSupervisedConfigRunner

    Registry(UNet)
    Registry(MeshNet)
    Registry(CustomDiceCallback)
    Registry(CustomSupervisedConfigRunner)

except ImportError:
    print("Catalyst not found. Loading production environment")
