import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


class NonOverlappingLocallyConnected1d(nn.Module):
    def __init__(
        self, input_channels, out_channels, out_dim, bias=False,s=2
    ):
        super(NonOverlappingLocallyConnected1d, self).__init__()
        self.weight = nn.Parameter( # input [bs, cin, space], weight [cout, cin, space]
            torch.randn(
                out_channels,
                input_channels,
                out_dim * s, # 2 would become patch_size
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.randn(1, out_channels, out_dim))
        else:
            self.register_parameter("bias", None)

        self.input_channels = input_channels

    def forward(self, x):
        x = x[:, None] * self.weight # [bs, cout, cin, space]
        x = x.view(*x.shape[:-1], -1, 2) # [bs, cout, cin, space // 2, 2]
        x = x.sum(dim=[-1, -3]) # [bs, cout, space // 2]
        x /= self.input_channels ** .5
        if self.bias is not None:
            x += self.bias * 0.1
        return x


class LocallyHierarchicalNet(nn.Module):
    def __init__(self, input_channels, h, out_dim, num_layers, bias=False,s=2):
        super(LocallyHierarchicalNet, self).__init__()

        d = s ** num_layers

        self.hier = nn.Sequential(
            NonOverlappingLocallyConnected1d(
                input_channels, h, d // 2, bias,s
            ),
            nn.ReLU(),
            *[nn.Sequential(
                    NonOverlappingLocallyConnected1d(
                        h, h, d // 2 ** (l + 1), bias,s
                    ),
                    nn.ReLU(),
                )
                for l in range(1, num_layers)
            ],
        )
        self.beta = nn.Parameter(torch.randn(h, out_dim))

    def forward(self, x):
        y = self.hier(x)
        y = y.mean(dim=[-1])
        y = y @ self.beta / self.beta.size(0)
        return y
    
       
       
          