from typing import Callable
import torch
import numpy as np
from tqdm import tqdm
from utils import format_time, set_seed, return_model_device


class EngineConfig:
    def __init__(self) -> None:
        pass


class Engine:
    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metric_func: Callable[[np.ndarray, np.ndarray], dict],
        loader,
        is_regression: bool,
        input_params,
    ) -> None:
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer

        # Create your PyTorch model

        # Check if the model is on an MPS device
        s = return_model_device(model)
        print(f"Model is on {s} device:")

        self.metric_func = metric_func
        self.loader = loader

        self.n_epochs = input_params["n_epochs"]
        self.device = input_params["device"]
        self.seed_val = input_params["seed_val"]
        set_seed(self.seed_val)
        self.is_regression = is_regression

    def train(self):
        loss_values = []
        for _ in tqdm(range(0, self.n_epochs), desc="Epoch"):
            self.model.train()
            total_loss = 0
            for _, batch in enumerate(self.loader["train"]):
                b_input_ids, b_input_tokent_type_ids, b_labels = tuple(
                    t.to(self.device) for t in batch
                )

                self.model.zero_grad()

                loss = self.model(
                    b_input_ids, token_type_ids=b_input_tokent_type_ids, labels=b_labels
                ).loss
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss = total_loss / len(self.loader["train"])
            loss_values.append(avg_train_loss)
        return loss_values

    def eval(self, loader_key="validation"):
        eval_metrics = []
        for i, batch in enumerate(self.loader[loader_key]):
            b_input_ids, b_input_tokent_type_ids, b_labels = tuple(
                t.to(self.device) for t in batch
            )

            with torch.no_grad():
                # self.model.config.gft_mat = torch.rand(
                #     self.model.config.tpu_short_seq_length,
                #     self.model.config.tpu_short_seq_length,
                #     dtype=torch.complex64,
                # ).to(self.device)

                logits = self.model(b_input_ids, b_input_tokent_type_ids).logits

            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits, axis=1).flatten() if not self.is_regression \
                else logits.squeeze() 
            label_ids = b_labels.to("cpu").numpy()  # TODO: check if this is necessary

            """
            self.metric_func can be acc, f1, etc. depending on the task
            for sst2 it is acc
            it will return the acc of the current batch
            """
            d = self.metric_func(predictions=logits, references=label_ids)
            # key = list(d.keys())[0]  # sometimes accuracy sometimes f1
            # val_metric += d[key]
            # n_eval_steps += 1

            eval_metrics.append(d)
        return eval_metrics
