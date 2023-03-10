import torch.nn as nn
import torch
import time

# torch.manual_seed(0)
dense = nn.Linear(10, 20)
dense2 = nn.Linear(20, 5)
input = torch.randn(10, requires_grad=True)
expected = torch.zeros(5)
expected[1] = 1
out = dense(input)
relu = nn.ReLU()
soft = nn.Softmax()
loss = nn.CrossEntropyLoss()
LEARNING_RATE = 0.0001

# for i in range(1):
#     out = dense(input)
#     out = relu(out)
#     out = dense2(out)
#     l = loss(out, expected)
#     l.backward()
#     with torch.no_grad():
#         dense.weight -= dense.weight.grad * LEARNING_RATE
#         dense2.weight -= dense2.weight.grad * LEARNING_RATE
#     dense.weight.grad.zero_()
#     if i%100 == 0:
#         print("Loss:", l)
#         print(soft(out))
#         print("\n")

input = torch.randn(40000)

dense = nn.Linear(40000, 4096);
soft = nn.Softmax();
start_time = time.time()
soft(dense(input))
print(f'Finished in: {(time.time() - start_time) * 1000}ms')

