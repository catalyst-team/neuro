from custom_metrics import custom_dice_metric
from functools import partial
from typing import Dict, Union, Iterable
from catalyst.metrics._functional_metric import FunctionalBatchMetric
from catalyst.callbacks.metric import FunctionalBatchMetricCallback


class CustomDiceCallback(FunctionalBatchMetricCallback):
    def __init__(
            self,
            input_key: Union[str, Iterable[str], Dict[str, str]],
            target_key: Union[str, Iterable[str], Dict[str, str]],
            metric_key: str,
            num_classes: int = 30,
            compute_on_call: bool = True,
            log_on_batch: bool = True,
            prefix: str = None,
            suffix: str = None,
        ):
            """Init."""
            metric_fn = partial(custom_dice_metric, num_classes=num_classes)
            super().__init__(
                metric=FunctionalBatchMetric(
                    metric_fn=metric_fn,
                    metric_key=metric_key,
                    compute_on_call=compute_on_call,
                    prefix=prefix,
                    suffix=suffix,
                ),
                input_key=input_key,
                target_key=target_key,
                log_on_batch=log_on_batch,
            )
