from catalyst.dl import registry, Callback, BatchMetricCallback, CallbackOrder, State
from custom_metrics import custom_dict_metric


@registry.Callback

class NiftiInferCallback(Callback):

    def __init__(self, subm_file):
        super().__init__(CallbackOrder.Internal)
        self.subm_file = subm_file
        self.preds = []

    def on_batch_end(self, state: State):
        paths = state.input["paths"]
        preds = state.output["logits"].detach().cpu().numpy()
        preds = preds.argmax(axis=1)
        for path, pred in zip(paths, preds):
            self.preds.append((path, pred))


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
