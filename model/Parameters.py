# -*- coding: utf-8 -*-
import os
import logging
from typing import Dict

import torch

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def get_default_parameters(
    num_epochs: int = 100,
    epoch_start: int = 0,
    batch_size: int = 1,
    checkpoint_step: int = 2,
    validation_step: int = 2,
    num_validation: int = 1000,
    num_workers: int = 1,
    learning_rate: float = 0.001,
    cuda: str = "0",
    use_gpu: bool = True,
    pretrained_model_path: float = None,
    save_model_path: str = "./.checkpoints",
    log_file: str = "./model.log",
) -> Dict:
    """
    Args:
        num_epochs: Number of epochs to train for
        epoch_start: Start counting epochs from this number
        batch_size: Number of images in each batch
        checkpoint_step: How often to save checkpoints (epochs)
        validation_step: How often to perform validation (epochs)
        num_validation: How many validation images to use
        num_workers: Number of workers
        learning_rate: learning rate used for training
        cuda: GPU ids used for training
        use_gpu: whether to user gpu for training
        pretrained_model_path: path to pretrained model
        save_model_path: path to save model
        log_file: path to log file
    """
    return locals()


class Parameters:
    def __init__(self, param_file_path):
        params = get_default_parameters()
        if os.path.isfile(param_file_path):
            with open(param_file_path) as file:
                params.update(load(file, Loader=Loader))

        self.__dict__.update(params)
        self.use_gpu = self.use_gpu and torch.cuda.is_available()

    def get_logger(self, name, level=logging.INFO):
        log = logging.getLogger(name)

        if self.epoch_start <= 0:
            with open(self.log_file, "w"):
                pass

        hdlr = logging.FileHandler(self.log_file)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        hdlr.setFormatter(formatter)
        log.addHandler(hdlr)
        log.setLevel(level)
        return log
