import torch
from catalyst.metrics.functional._segmentation import dice


def custom_dice_metric(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-7,
    threshold: float = None,
    num_classes: int = 30,
):

    outputs = torch.nn.functional.softmax(outputs)

    one_hot_targets = (
        torch.nn.functional.one_hot(targets, num_classes)
        .permute(0, 4, 1, 2, 3)
        .cuda()
    )

    macro_dice = dice(outputs, one_hot_targets, mode='macro')
    return macro_dice

