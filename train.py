#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

import os
import pip
import random
import subprocess
import sys

pip.install('wandb')

import wandb

if __name__ == "__main__":

    wandb.login(key="0085cae3625fea8bf03743455414f669a3eb6d3f")

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="ali-pai-test",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        }
    )

    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

    print("********** FINISH **********")
