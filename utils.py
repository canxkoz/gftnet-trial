# %%
import torch
import datetime

from random import seed

import numpy as np
import random


def set_seed(seed_val) -> None:
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def get_device():
    if torch.backends.cuda.is_built():
        print("CUDA")
        device = torch.device("cuda")
    elif torch.backends.mps.is_built():
        print("mps")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        raise Exception("GPU is not avalaible!")
    return device


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def return_model_device(model):
    for param in model.parameters():
        if param.device.type == "mps":
            s = "mps"
        elif param.device.type == "cuda":
            s = "cuda"
        elif param.device.type == "cpu":
            s = "cpu"
    return s


# BUNU EKLE

# def preprocess_function(examples):
#     if sentence2_key is None:
#         return tokenizer(examples[sentence1_key], truncation=True)
#     return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

#     encoded_dataset = dataset.map(preprocess_function, batched=True)
#     num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2


# %%
if __name__ == "__main__":
    pass

# %%
