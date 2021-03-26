from catalyst.dl import BatchMetricCallback
from custom_metrics import custom_dice_metric


class CustomDiceCallback(BatchMetricCallback):
    def __init__(
        self,
        input_key: str = "targets",
        output_key: str = "logits",
        prefix: str = "dice",
        eps: float = 1e-7,
        threshold: float = None,
        activation: str = "Softmax",
    ):
        super().__init__(
            prefix=prefix,
            metric_fn=custom_dice_metric,
            input_key=input_key,
            output_key=output_key,
            eps=eps,
            threshold=threshold,
            activation=activation,
        )
