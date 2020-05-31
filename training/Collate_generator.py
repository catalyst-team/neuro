import torch


class CollateGeneratorFn:
    """
    Callable object doing job of ``collate_fn`` like ``default_collate``,
    but does not cast batch items with specified key to :class:`torch.Tensor`.
    Only adds them to list.
    Supports only key-value format batches
    """

    def __init__(self, max_batch_size):
        """
        Args:
            max_batch_size: max batch size
        """
        self.max_batch_size = max_batch_size

    def __call__(self, batch):
        """
        Args:
            batch: current batch
        Returns:
            batch: filter by max_batch_size
        """
        result = {}
        for key in batch[0][0].keys():
            items = [d[key] for d in batch[0][: self.max_batch_size]]
            result[key] = torch.stack(items)
        return result
