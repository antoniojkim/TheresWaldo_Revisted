# -*- coding: utf-8 -*-

import torch


class PolyLearningRateDecay(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        power=1,
        max_decay_steps=100,
        last_epoch=-1,
        final_learning_rate=0.001,
    ):
        self.optimizer = optimizer
        self.power = power
        self.current_step = 0
        self.max_decay_steps = max_decay_steps
        self.last_epoch = last_epoch
        self.final_learning_rate = final_learning_rate
        super().__init__(optimizer, last_epoch)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        pass

    def step(self, epoch=None):
        super().step(epoch)
        self.current_step += 1

    def __next__(self):
        self.step()
        return self

    def get_lr(self):
        if self.current_step > self.max_decay_steps:
            return [self.final_learning_rate for lr in self.base_lrs]

        return [
            (base_lr - self.final_learning_rate)
            * (1 - self.current_step / self.max_decay_steps) ** self.power  # noqa: W503
            + self.final_learning_rate  # noqa: W503
            for base_lr in self.base_lrs
        ]

    @property
    def learning_rate(self):
        return self._last_lr[0]


if __name__ == "__main__":
    import torchvision as torchv

    resnet = torchv.models.resnet50(pretrained=True)
    optimizer = torch.optim.SGD(
        resnet.parameters(), 0.1, momentum=0.9, weight_decay=1e-4
    )

    max_epoch = 30
    with PolyLearningRateDecay(optimizer, power=4, max_epoch=max_epoch) as scheduler:
        print(scheduler.learning_rate)
        for i in range(max_epoch):
            optimizer.step()
            next(scheduler)
            print(scheduler.learning_rate)
