#!/usr/bin/env python
# ! -*- coding: utf-8 -*-

import os
import pip
import random
import subprocess
import sys

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# package_path = os.path.join(current_dir, 'wandb', 'wandb-0.15.0-py3-none-any.whl')
# pip.main(['install', package_path])

package_path = os.path.join(current_dir, 'wandb', '*.whl')
wandb_path = os.path.join(current_dir, 'wandb')
# pip.main(['install', '--no-index', '--no-deps', package_path])

# cmd_str = "pip install --no-index --no-deps " + package_path
# subprocess.run(cmd_str, shell=True)

cmd1 = 'export PATH="$HOME/.local/bin:$PATH"'
subprocess.run(cmd1, shell=True)

for f in os.listdir(wandb_path):
    if f.endswith('whl'):
        wheel_path = wandb_path + '/' + str(f)
        print(wheel_path)
        cmd_str = "pip install --no-index --no-deps --user " + wheel_path
        subprocess.run(cmd_str, shell=True)
    # wheel_path = os.path.join(wandb_path)
    # cmd_str = "pip install --no-index --no-deps " + wheel_path
    # subprocess.run(cmd_str, shell=True)

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
