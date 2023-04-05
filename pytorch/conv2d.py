import torch.nn as nn
import torch
torch.manual_seed(0) 

conv = nn.Conv2d(3, 2, 2)
input = torch.randn(3, 10, 10)

out = conv(input)
print(out.size())
print(out)

#input: 3,10,10
#bias: 1x2
#weight: 2x3x2x2
#output: 2x9x9

#input channels, output channels, size of kernal

#-0.2200,  0.4209,  0.3891,  0.5247,  0.1562,  0.3495, -0.2014,0.1352,  0.3869,-0.3524,  0.4762, -0.3580,  0.0781,  0.4595,  0.3204, -0.3139,0.4369, -0.0668,0.1266, -0.0281,  0.0219, -0.3503, -0.8508, -0.3736, -0.0594,0.4342, -0.0397,-0.0036,  0.2592,  0.1561,  0.6666, -0.2666, -0.2953,  0.0030,-0.3614, -1.1817,0.0595, -0.1816,  0.3106, -0.0189,  0.0076, -0.5597,  0.1968,0.4679,  0.3871,-0.2154, -0.4466, -0.6319, -0.9016, -0.4225, -0.8336, -0.6215,-0.1708,  0.0895,-0.3704,  0.5322, -0.1454, -0.0124, -0.1547,  0.6363,  0.0062,-0.1669, -0.3771,-0.8525,  0.7721,  0.5218,  0.2267,  0.0602,  0.4005,  0.6504,0.5556, -0.2787,0.3362,  0.5274, -0.0952,  0.3594, -0.5949, -0.1167,  0.4206,-0.1545,  0.1088,-0.1752, -0.0347,  0.4933,  0.2374, -0.6243, -0.2605, -0.7097,-1.3053, -0.6729,0.4036,  0.1611,  0.4553, -0.2491, -0.3209,  0.4474,  0.5935,-0.3583,  0.0476,-0.0668,  0.7819,  1.1413,  0.7632,  0.2232,  0.2744, -0.0411,0.6053,  0.7352,-0.2617,  0.4620,  0.6738,  0.3826, -1.0443, -0.8067,  0.8293,0.8309,  0.4130,0.9796,  1.3051,  1.1141,  0.2967,  0.1148, -0.7397,  0.4408,0.3586, -0.9063,0.4992,  0.2094,  0.4256, -0.2214, -0.1760,  0.6599, -0.1225,-0.6409, -0.6755,0.0073, -0.0027, -0.6996, -0.8825, -1.1005, -0.5838, -0.8581,0.2281,  1.0349,-0.0043,  0.2559, -0.4881, -0.1149, -0.2618,  0.5771, -0.4618,-0.5093, -0.2962,-1.3430, -0.2727,  0.1503,  0.9286,  1.1572,  1.4308,  1.2364,1.6413,  0.1275