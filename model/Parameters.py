# -*- coding: utf-8 -*-
import os

import torch

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class Parameters:
    def __init__(self, param_file_path):
        params = {}
        if os.path.isfile(param_file_path):
            with open(param_file_path) as file:
                params = load(file, Loader=Loader)

        self.num_epochs = params.get("num_epochs", 100)  # Number of epochs to train for
        self.epoch_start = params.get(
            "epoch_start", 0
        )  # Start counting epochs from this number
        self.batch_size = params.get("batch_size", 1)  # Number of images in each batch
        self.checkpoint_step = params.get(
            "checkpoint_step", 2
        )  # How often to save checkpoints (epochs)
        self.validation_step = params.get(
            "validation_step", 2
        )  # How often to perform validation (epochs)
        self.num_validation = params.get(
            "num_validation", 1000
        )  # How many validation images to use
        self.num_workers = params.get("num_workers", 1)  # Number of workers
        self.learning_rate = params.get(
            "learning_rate", 0.001
        )  # learning rate used for training
        self.cuda = params.get("cuda", "0")  # GPU ids used for training
        self.use_gpu = params.get("use_gpu", True)  # whether to user gpu for training
        self.pretrained_model_path = params.get(
            "pretrained_model_path", None
        )  # path to pretrained model
        self.save_model_path = params.get(
            "save_model_path", "./.checkpoints"
        )  # path to save model
        self.log_file = params.get("log_file", "./train.log")  # path to log file

        self.use_gpu = self.use_gpu and torch.cuda.is_available()
