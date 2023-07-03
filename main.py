# %%
from utils import *
from engine import *
from transformers import logging
from gftnet.model import FNetForSequenceClassification as FNetSC
from gftnet.configuration_fnet import FNetConfig
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from evaluate import load
from transformers import get_linear_schedule_with_warmup, FNetTokenizer
from datasets import load_dataset
from typing import Dict, Union, Any

parser = argparse.ArgumentParser(description="Description of your script")
parser.add_argument("-f", "--flag", action="store_true", help="Enable the flag")

# Parse the command-line arguments
args = parser.parse_args()

# Call your function with the provided flag argument

GLUE_TASKS = [
    "cola",
    "mnli",
    "mnli-mm",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "stsb",
    "wnli",
]

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_name = "wnli"
# qqp de F score la alakalı hata geliyor ona bir bak (belki az datadan dolayıdır to end 100 aldım)
# stsb regression olduğu için sonu argmaxlamak yerine squeezelemek lazım
# run_glue_no_train.py deki line 693 deki gibi if else statement lazım

# genel hepsini bir for loop içine alıp çalıştırmak lazım

logging.set_verbosity_error()

input_params = {
    "max_length": 8,
    "batch_size": 2,
    "test_size": 0.2,
    "n_epochs": 1,
    "seed_val": 42,
    "device": "cpu",
    "lr": 2e-5,
}

device = input_params["device"]
to_end = 100

# Preparing Dataset for all cases

raw_datasets = load_dataset("glue", task_name)
metric = load("glue", task_name)
tokenizer = FNetTokenizer.from_pretrained("google/fnet-base")


is_regression = task_name == "stsb"
if not is_regression:
    label_list = raw_datasets["train"].features["label"].names
    label2id = {v: i for i, v in enumerate(label_list)}
    id2label = {i: v for i, v in enumerate(label_list)}
    num_labels = len(label_list)
else:
    num_labels = 1

sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]


def preprocess_function(examples):
    if sentence2_key is None:
        texts = (examples[sentence1_key],)
    else:
        texts = (examples[sentence1_key], examples[sentence2_key])

    result = tokenizer(
        *texts,
        padding=True,
        max_length=input_params["max_length"],
        truncation=True,
        return_tensors="pt",
    )

    return result


processed_datasets = raw_datasets.map(preprocess_function, batched=True)
splits = ["train"] + ["validation_matched" if task_name == "mnli" else "validation"]

input_ids = {}
token_type_ids = {}
labels = {}
loader = {}
tensor_dataset = {}


for split in splits:
    # if split == "train":
    #     df_s = processed_datasets[split].to_pandas()[:to_end]
    # else:
    #     df_s = processed_datasets[split].to_pandas()
    df_s = processed_datasets[split].to_pandas()[:to_end]

    input_ids[split] = torch.LongTensor(df_s.input_ids.tolist()).to(device)
    token_type_ids[split] = torch.LongTensor(df_s.token_type_ids.tolist()).to(device)
    if not is_regression:
        labels[split] = torch.LongTensor(df_s.label.tolist()).to(device)
    else:
        labels[split] = torch.FloatTensor(df_s.label.tolist()).to(device)

    tensor_dataset[split] = TensorDataset(
        input_ids[split], token_type_ids[split], labels[split]
    )
    loader[split] = DataLoader(
        tensor_dataset[split], batch_size=input_params["batch_size"], shuffle=False
    )
# %%
fnet_config = FNetConfig(
    use_tpu_fourier_optimizations=True,
    tpu_short_seq_length=input_params["max_length"],
    num_labels=num_labels,
    original_version=args.flag,
    device=device,
)

fnet_config.device = device

# for key, dataloader in loader.items():
#     dataloader.dataset.tensors = [
#         tensor.to(device) for tensor in dataloader.dataset.tensors
#     ]

model = FNetSC.from_pretrained("google/fnet-base", config=fnet_config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=input_params["lr"], eps=1e-8)
total_steps = len(loader["train"]) * input_params["n_epochs"]
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps
)

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


validation_key = splits[1]
train_metrics = engine.eval(loader_key="train")
metric_name, avg = get_mean_metric(train_metrics)
print(f"Train {metric_name}: {avg}")

eval_metrics = engine.eval(loader_key=validation_key)
metric_name, avg = get_mean_metric(eval_metrics)
print(f"Validation {metric_name}: {avg}")


# %%
