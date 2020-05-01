for stage in stages:
    for epoch in epochs:
        for dataloader in dataloaders:
            for batch in dataloader:
                for new_batch in handle(batch):

