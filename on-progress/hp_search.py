import torch
from utils import *
import optuna

from transformers import logging

param_dict = {
    "lr": [1e-5, 2e-5],
    "n_epochs": [1],
    "max_length": [16, 32],
    "batch_size": [32],
}


class BertObjective:
    def __init__(self, d, device):
        self.d = d
        self.device = device

    def __call__(self, trial: optuna.trial.Trial):
        self.lr = trial.suggest_float("lr", self.d["lr"][0], self.d["lr"][1], log=True)
        self.p = trial.suggest_float("lr", self.d["lr"][0], self.d["lr"][1], log=True)
        self.n_epochs = trial.suggest_categorical("n_epochs", self.d["n_epochs"])
        self.max_length = trial.suggest_categorical("max_length", self.d["max_length"])
        self.batch_size = trial.suggest_categorical("batch_size", self.d["batch_size"])

        # trainer = Trainer(model_init = model_init(p), )

        model, loader, optimizer, scheduler = init_objects(
            self.lr, self.n_epochs, self.max_length, self.batch_size
        )

        model.to(self.device)
        val_mcc, _ = train_eval_loop(
            model, loader, optimizer, scheduler, self.device, self.n_epochs
        )

        return val_mcc


device = torch.device("mps")
study = optuna.create_study(study_name="Stduy 0", direction="maximize")
study.optimize(BertObjective(param_dict, device), n_trials=3)

# Train again with best parameters
lr = study.best_params["lr"]
n_epochs = study.best_params["n_epochs"]
max_length = study.best_params["max_length"]
batch_size = study.best_params["batch_size"]

model, loader, optimizer, scheduler = init_objects(lr, n_epochs, max_length, batch_size)
model.to(device)
val_mcc, _ = train_eval_loop(model, loader, optimizer, scheduler, device, n_epochs)
