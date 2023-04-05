import torch.nn as nn
import torch
torch.manual_seed(0) 

conv = nn.Conv2d(3, 2, 2)
input = torch.randn(3, 10, 10)
print("input")
print(input)
print("bias")
print(conv.bias)
print("weight")
print(conv.weight)
print("output")
print(conv(input))

#input: 3,10,10
#bias: 1x2
#weight: 2x3x2x2
#output: 2x9x9

#input channels, output channels, size of kernal