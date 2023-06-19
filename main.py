# %%
from utils import *
from trainer import *
from transformers import logging


logging.set_verbosity_error()

input_params = {
    "max_length": 8,
    "batch_size": 32,
    "test_size": 0.2,
    "n_epochs": 10,
    "seed_val": 42,
    "device": get_device(),
    "lr": 2e-5,
}

model, optimizer, scheduler, metric, loader = prepare_trainer_input(
    task="sst2", input_params=input_params
)

trainer = Trainer(
    n_epochs=input_params["n_epochs"],
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metric_func=metric.compute,
    loader=loader,
    device=input_params["device"],
    seed_val=input_params["seed_val"],
)

loss_values = trainer._train()

eval_metrics = trainer._eval(loader_key="train")
print(eval_metrics)
eval_metrics = trainer._eval(loader_key="validation")
print(eval_metrics)

# %%
