import torch.nn as nn
import torch
conv = nn.Conv2d(1, 1, 2, stride=1)
input = torch.randn(1, 4, 4)
print(conv(input))