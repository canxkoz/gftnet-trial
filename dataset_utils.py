from utils import *
import torch.utils.data as Data


from transformers import get_linear_schedule_with_warmup, FNetTokenizer
from datasets import load_dataset

from dataclasses import dataclass
from typing import Dict, Union, Any

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


@dataclass
class Dataset:
    task: str
    max_length: int
    batch_size: int
    df_s = {}
    x = {}
    y = {}

    input_ids = {}
    token_type_ids = {}

    tensor_dataset = {}
    loader = {}

    tokenizer = FNetTokenizer.from_pretrained("google/fnet-base")
    to_end: int = -1

    def __post_init__(self):
        self.df_d = load_dataset("glue", self.task)
        for split in ["train", "validation", "test"]:
            self.df_s[split] = self.df_d[split].to_pandas()[: self.to_end]
            self.x[split] = self.df_d[split].to_pandas().sentence.values[: self.to_end]
            self.y[split] = self.df_d[split].to_pandas().label.values[: self.to_end]

            input = self.get_transformer_input(split)

            self.input_ids[split], self.token_type_ids[split] = (
                input.input_ids,
                input.token_type_ids,
            )

            self.df_s[split]["input_ids"] = self.input_ids[split].tolist()

            self.tensor_dataset[split] = Data.TensorDataset(
                self.input_ids[split],
                self.token_type_ids[split],
                torch.LongTensor(self.y[split]),
            )

            self.loader[split] = Data.DataLoader(
                self.tensor_dataset[split], batch_size=self.batch_size, shuffle=False
            )

    def __len__(self):
        return len(self.df_s)

    def get_transformer_input(self, split):
        return self.tokenizer(
            list(self.x[split]),
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )


if __name__ == "__main__":
    dataset = Dataset(task="sst2", max_length=8, batch_size=32)
    pass
