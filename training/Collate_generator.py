import torch


class CollateGeneratorFn:
    """
    Callable object doing job of ``collate_fn`` like ``default_collate``,
    but does not cast batch items with specified key to :class:`torch.Tensor`.
    Only adds them to list.
    Supports only key-value format batches
    """
    def __call__(self, batch):
        """
        Args:
            batch: current batch
        Returns:
            batch values filtered by `keys`
        """
        result = {}
        for  key in batch[0][0].keys():
            items = [d[key] for d in batch[0]]
            result[key] = torch.stack(items)
        return result
