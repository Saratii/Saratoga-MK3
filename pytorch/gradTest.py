from torch import *
from torch.nn import *

manual_seed(0)
set_printoptions(10);
conv = Conv2d(1, 3, 2, bias=False) #grey scale input, feature sets, kernal size
input = randn(1, 4, 4)
print(input)
print(conv(input))
