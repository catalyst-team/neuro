num_epochs = 30
logdir = "logs/unet"

optimizer = torch.optim.Adam(unet.parameters(), lr=0.01, weight_decay=0.0001)
scheduler = CosineAnnealingLR(optimizer, T_max=30)

runner = SupervisedRunner(
    input_key="images", input_target_key="labels", output_key="logits"
)

callbacks = [
    TensorboardLogger(),
    SchedulerCallback(reduced_metric="loss"),
    CustomDiceCallback(),
    CheckpointCallback(),
]

runner.train(
    model=unet,
    criterion=CrossEntropyLoss(),
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=callbacks,
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True,
)
