import torch

input = torch.randn(1, 224, 224) #images, image size

conv = torch.nn.Conv2d(1, 96, 11, stride=4) #grey scale input, feature sets, kernal size
relu = torch.nn.ReLU()
maxPool = torch.nn.MaxPool2d(3, stride=2)
conv2 = torch.nn.Conv2d(96, 256, 5, padding=2)
relu2 = torch.nn.ReLU()
maxPool2 = torch.nn.MaxPool2d(3, stride=2)
conv3 = torch.nn.Conv2d(256, 384, 3, padding=1)
relu3 = torch.nn.ReLU()
conv4 = torch.nn.Conv2d(384, 384, 3, padding=1)
relu4 = torch.nn.ReLU()
conv5 = torch.nn.Conv2d(384, 256, 3, padding=1)
relu5 = torch.nn.ReLU()
maxPool3 = torch.nn.MaxPool2d(3, stride=2)

dense1 = torch.nn.Linear(6400, 4096)
relu6 = torch.nn.ReLU()
drop1 = torch.nn.Dropout(p=0.5)
dense2 = torch.nn.Linear(4096, 4096)
relu7 = torch.nn.ReLU()
drop2 = torch.nn.Dropout(p=0.5)
dense3 = torch.nn.Linear(4096, 10)
soft = torch.nn.Softmax(dim=0)
loss = torch.nn.CrossEntropyLoss()
target = torch.zeros(10)
target[2] = 1

LEARNING_RATE = 0.01
for i in range(1):
    out1 = maxPool(relu(conv(input)))
    out2 = maxPool2(relu2(conv2(out1)))
    out3 = relu3(conv3(out2))
    out4 = relu4(conv4(out3))
    out5 = maxPool3(relu5(conv5(out4)))
    out6 = drop1(relu6(dense1(torch.flatten(out5))))
    out7 = drop2(relu7(dense2(out6)))
    print(out7)
    out8 = soft(dense3(out7))
    out9 = loss(out8, target)
    # print(f'Iteration: {i}')
    # print(out9)
    out9.backward()
    dense3.weight = torch.nn.Parameter(dense3.weight - dense3.weight.grad * LEARNING_RATE)
    dense3.bias = torch.nn.Parameter(dense3.bias - dense3.bias.grad * LEARNING_RATE)
    dense2.weight = torch.nn.Parameter(dense2.weight - dense2.weight.grad * LEARNING_RATE)
    dense2.bias = torch.nn.Parameter(dense2.bias - dense2.bias.grad * LEARNING_RATE)
    dense1.weight = torch.nn.Parameter(dense1.weight - dense1.weight.grad * LEARNING_RATE)
    dense1.bias = torch.nn.Parameter(dense1.bias - dense1.bias.grad * LEARNING_RATE)
    conv5.weight = torch.nn.Parameter(conv5.weight - conv5.weight.grad * LEARNING_RATE)
    conv5.bias = torch.nn.Parameter(conv5.bias - conv5.bias.grad * LEARNING_RATE)
    conv4.weight = torch.nn.Parameter(conv4.weight - conv4.weight.grad * LEARNING_RATE)
    conv4.bias = torch.nn.Parameter(conv4.bias - conv4.bias.grad * LEARNING_RATE)
    conv3.weight = torch.nn.Parameter(conv3.weight - conv3.weight.grad * LEARNING_RATE)
    conv3.bias = torch.nn.Parameter(conv3.bias - conv3.bias.grad * LEARNING_RATE)
    conv2.weight = torch.nn.Parameter(conv2.weight - conv2.weight.grad * LEARNING_RATE)
    conv2.bias = torch.nn.Parameter(conv2.bias - conv2.bias.grad * LEARNING_RATE)
    conv.weight = torch.nn.Parameter(conv.weight - conv.weight.grad * LEARNING_RATE)
    conv.bias = torch.nn.Parameter(conv.bias - conv.bias.grad * LEARNING_RATE)
 






