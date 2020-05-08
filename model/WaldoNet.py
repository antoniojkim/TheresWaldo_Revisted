# -*- coding: utf-8 -*-

import torch


class WaldoNet(torch.nn.Module):
    @staticmethod
    def Conv2dBatch(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=True,
        **kwargs
    ):
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def __init__(self, load_path=None):
        super().__init__()

        self.block1 = torch.nn.Sequential(
            self.Conv2dBatch(3, 64, kernel_size=3, stride=1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = torch.nn.Sequential(
            self.Conv2dBatch(64, 128, kernel_size=3, stride=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block3 = torch.nn.Sequential(
            self.Conv2dBatch(128, 256, kernel_size=3, stride=2),
            self.Conv2dBatch(256, 128, kernel_size=1),
            self.Conv2dBatch(128, 256, kernel_size=3, stride=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block4 = torch.nn.Sequential(
            self.Conv2dBatch(256, 512, kernel_size=3, stride=2),
            self.Conv2dBatch(512, 256, kernel_size=1),
            self.Conv2dBatch(256, 512, kernel_size=3, stride=2),
        )
        self.block5 = torch.nn.Sequential(
            self.Conv2dBatch(512, 500, kernel_size=1),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        if load_path is not None:
            self.load(load_path)
        else:
            self.reset_parameters()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = torch.flatten(x, 1)
        return x

    def __call__(self, x):
        return self.forward(x)

    def num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def reset_parameters(self):
        init_funcs = {
            1: lambda x: torch.nn.init.normal_(x, mean=0.0, std=1.0),  # biases
            2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.0),  # weights
            3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.0),  # conv1D filters
            4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.0),  # conv2D filters
            "default": lambda x: torch.nn.init.constant(x, 1.0),
        }
        for p in self.parameters():
            init_func = init_funcs.get(len(p.shape), init_funcs["default"])
            init_func(p)

    def load(self, load_path, strict=False):
        self.load_state_dict(torch.load(load_path), strict=strict)

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)
