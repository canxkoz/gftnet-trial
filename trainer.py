from typing import Callable
import random
import torch
import numpy as np
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        n_epochs: int,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        metric_func: Callable[[np.ndarray, np.ndarray], dict],
        loader: torch.utils.data.DataLoader,
        device,
        seed_val: int = 42,
    ) -> None:
        self.n_epochs = n_epochs
        self.seed_val = seed_val
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer

        self.metric_func = metric_func
        self.loader = loader
        self.device = device

        self._set_seed()

    def _set_seed(self) -> None:
        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)

    def _train(self):
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

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

            avg_train_loss = total_loss / len(self.loader["train"])
            loss_values.append(avg_train_loss)
        return loss_values

    def _eval(self, loader_key="validation"):
        val_metric, n_eval_steps = 0, 0
        eval_metrics = []
        for _, batch in enumerate(self.loader[loader_key]):
            b_input_ids, b_input_tokent_type_ids, b_labels = tuple(
                t.to(self.device) for t in batch
            )

            with torch.no_grad():
                logits = self.model(
                    b_input_ids, token_type_ids=b_input_tokent_type_ids
                ).logits

            logits = logits.detach().cpu().numpy()
            logits = np.argmax(logits, axis=1).flatten()
            label_ids = b_labels.to("cpu").numpy()  # TODO: check if this is necessary

            # this returns a dictionary
            d = self.metric_func(predictions=logits, references=label_ids)
            key = list(d.keys())[0]  # sometimes accuracy sometimes f1
            val_metric += d[key]
            n_eval_steps += 1

        eval_metrics.append(val_metric / n_eval_steps)
        return eval_metrics
