# %%
import torch
import datetime

from random import seed
import torch.utils.data as Data
from datasets import load_dataset, load_metric
from gftnet.model import FNetForSequenceClassification
from gftnet.configuration_fnet import FNetConfig
from transformers import get_linear_schedule_with_warmup, FNetTokenizer

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


def prepare_trainer_input(task, input_params):
    task = "mnli" if task == "mnli-mm" else task
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2

    # define config
    configuration = FNetConfig(
        use_tpu_fourier_optimizations=True,
        tpu_short_seq_length=input_params["max_length"],
        num_labels=num_labels,
    )

    # define loader
    loader, y, df_s = init_loader(
        task=task,
        max_length=input_params["max_length"],
        batch_size=input_params["batch_size"],
    )
    # config needs to have gft_mat
    # configuration.gft_mat = torch.eye(configuration.tpu_short_seq_length)

    # define model
    model = FNetForSequenceClassification.from_pretrained(
        "google/fnet-base", config=configuration
    ).to(input_params["device"])

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=input_params["lr"], eps=1e-8)

    # define scheduler
    total_steps = len(loader["train"]) * input_params["n_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # define metric_func
    metric = load_metric("glue", task)

    return model, optimizer, scheduler, metric, loader


def init_loader(task, max_length, batch_size):
    df_s, x, y = {}, {}, {}
    input_ids, token_type_ids = {}, {}
    datasets, loader = {}, {}

    print(f"Task is: {task} ")

    dataset = load_dataset("glue", task)

    sentence1_key, sentence2_key = TASK_TO_KEYS[task]
    tokenizer = FNetTokenizer.from_pretrained("google/fnet-base")

    device = get_device()
    for split in ["train", "validation", "test"]:
        if split == "train" and device == torch.device("cpu"):
            to_end = 1000
        else:
            to_end = -1

        df_s[split] = dataset[split].to_pandas()[:to_end]
        x[split] = dataset[split].to_pandas().sentence.values[:to_end]
        y[split] = dataset[split].to_pandas().label.values[:to_end]

        input = tokenizer(
            list(x[split]),
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        input_ids[split], token_type_ids[split] = input.input_ids, input.token_type_ids

        df_s[split]["input_ids"] = input_ids[split].tolist()

        datasets[split] = Data.TensorDataset(
            input_ids[split], token_type_ids[split], torch.LongTensor(y[split])
        )

        loader[split] = Data.DataLoader(
            datasets[split], batch_size=batch_size, shuffle=False
        )

    return loader, y, df_s


def get_device():
    if torch.backends.cuda.is_built():
        print("CUDA")
        device = torch.device("cuda")
    elif torch.backends.mps.is_built():
        print("cpu")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        raise Exception("GPU is not avalaible!")
    return device


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


# BUNU EKLE

# def preprocess_function(examples):
#     if sentence2_key is None:
#         return tokenizer(examples[sentence1_key], truncation=True)
#     return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

#     encoded_dataset = dataset.map(preprocess_function, batched=True)
#     num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2


# %%
if __name__ == "__main__":
    task = "sst2"
    max_length = 4
    batch_size = 32

    df_s, x, y = {}, {}, {}
    input_ids, token_type_ids = {}, {}
    datasets, loader = {}, {}

    print(f"Task is: {task} ")

    dataset = load_dataset("glue", task)

    sentence1_key, sentence2_key = TASK_TO_KEYS[task]
    tokenizer = FNetTokenizer.from_pretrained("google/fnet-base")

    device = get_device()
    for split in ["train", "validation", "test"]:
        if split == "train" and device == torch.device("cpu"):
            to_end = 1000
        else:
            to_end = -1

        df_s[split] = dataset[split].to_pandas()[:to_end]
        x[split] = dataset[split].to_pandas().sentence.values[:to_end]
        y[split] = dataset[split].to_pandas().label.values[:to_end]

        input = tokenizer(
            list(x[split]),
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)

        input_ids[split], token_type_ids[split] = input.input_ids, input.token_type_ids

        datasets[split] = Data.TensorDataset(
            input_ids[split], token_type_ids[split], torch.LongTensor(y[split])
        )

        loader[split] = Data.DataLoader(
            datasets[split], batch_size=batch_size, shuffle=False
        )

# %%
