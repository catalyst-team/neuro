import torch


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

    targets = torch.flatten(one_hot_targets)
    outputs = torch.flatten(outputs)

    intersection = torch.sum(torch.dot(targets.double(), outputs.double()))
    union = torch.sum(targets) + torch.sum(outputs)
    dice = (2.0 * intersection + eps * (union == 0)) / (union + eps)

    return dice
