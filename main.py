# %%
from utils import *
from engine import *
from dataset_utils import *
from transformers import logging
from gftnet.model import FNetForSequenceClassification as FNetSC
from gftnet.configuration_fnet import FNetConfig
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description="Description of your script")
parser.add_argument("-f", "--flag", action="store_true", help="Enable the flag")

# Parse the command-line arguments
args = parser.parse_args()

# Call your function with the provided flag argument
from evaluate import load

logging.set_verbosity_error()

input_params = {
    "max_length": 8,
    "batch_size": 32,
    "test_size": 0.2,
    "n_epochs": 2,
    "seed_val": 42,
    "device": get_device(),
    "lr": 2e-5,
}

# chose sst2 or cola
task = "sst2"
task = "mnli" if task == "mnli-mm" else task
num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
device = input_params["device"]

fnet_config = FNetConfig(
    use_tpu_fourier_optimizations=True,
    tpu_short_seq_length=input_params["max_length"],
    num_labels=num_labels,
    original_version=args.flag,
    device=device,
)

dataset = Dataset(
    task, input_params["max_length"], input_params["batch_size"], to_end=100
)

loader = dataset.loader
for key, dataloader in loader.items():
    dataloader.dataset.tensors = [
        tensor.to(device) for tensor in dataloader.dataset.tensors
    ]

model = FNetSC.from_pretrained("google/fnet-base", config=fnet_config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=input_params["lr"], eps=1e-8)
total_steps = len(dataset.loader["train"]) * input_params["n_epochs"]
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
metric = load("glue", task)

engine = Engine(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metric_func=metric.compute,
    loader=loader,
    input_params=input_params,
)


loss_values = engine.train()
print(loss_values)


def get_mean_metric(metrics: list[dict]):
    # since all keys are same in metrics, we can just take the first one
    key = str(next(iter(train_metrics[0])))
    return key, np.mean([d[key] for d in metrics])


train_metrics = engine.eval(loader_key="train")
metric_name, avg = get_mean_metric(train_metrics)
print(f"Train {metric_name}: {avg}")

eval_metrics = engine.eval(loader_key="validation")
metric_name, avg = get_mean_metric(eval_metrics)
print(f"Validation {metric_name}: {avg}")


# %%
