# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn

from typing import Type
import torch.nn.init as init


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
    
class Reshaper(nn.Module):
    
    def __init__(self, size_in: int, channel_in: int, size_out: int, channel_out: int,
                 normalization: Type[nn.Module] = LayerNorm2d,
                 activation: Type[nn.Module] = nn.GELU):
        
        super(Reshaper, self).__init__()
        
        if size_in == size_out:
            self.reshaper = nn.ModuleList([
                nn.Conv2d(channel_in, channel_out, kernel_size=1),
            ])
            
        elif size_in < size_out:
            n = math.log2(size_out // size_in)
            if n.is_integer():
                n = int(n)
            else:
                raise ValueError(f"size_out must be a multiple of size_in, got {size_out} and {size_in}")
            
            self.reshaper = nn.ModuleList()
            for _ in range(n):
                self.reshaper.extend([
                    nn.ConvTranspose2d(channel_in, channel_in // 2, kernel_size=2, stride=2),
                    normalization(channel_in // 2),
                    activation(),
                ])
                channel_in = channel_in // 2
            self.reshaper.extend([
                nn.Conv2d(channel_in, channel_out, kernel_size=1),
            ])
        
        else:
            n = math.log2(size_in // size_out)
            if n.is_integer():
                n = int(n)
            else:
                raise ValueError(f"size_in must be a multiple of size_out, got {size_in} and {size_out}")
            
            self.reshaper = nn.ModuleList()
            for _ in range(n):
                self.reshaper.extend([
                    nn.Conv2d(channel_in, channel_in * 2, kernel_size=3, stride=2, padding=1),
                    normalization(channel_in * 2),
                    activation(),
                ])
                channel_in = channel_in * 2
            self.reshaper.extend([
                nn.Conv2d(channel_in, channel_out, kernel_size=1),
            ])
        self._initialize_weights()
            
    
    def _initialize_weights(self):
        for module in self.reshaper:
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.fill_(0)
    
    def forward(self, x):
        for layer in self.reshaper:
            x = layer(x)
        return x
