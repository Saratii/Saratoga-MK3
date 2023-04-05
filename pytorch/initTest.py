import torch.nn as nn
import torch
torch.manual_seed(0) 

conv = nn.Conv2d(4, 3, 3)

print(conv.kernel_size)




for k in conv.kernel_size:
    print(k)