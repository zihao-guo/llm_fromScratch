import torch
import torch.nn as nn

# dropout_layer = nn.Dropout(p = 0.5)

# t1 = torch.Tensor([1, 2, 3])
# t2 = dropout_layer(t1)

# print(t1)
# print(t2)

# layer = nn.Linear(in_features=3, out_features=5, bias=True)
# t1 = torch.Tensor([1, 2, 3])

# t2 = torch.Tensor([[1, 2, 3]])
# output2 = layer(t2)
# print(output2)
# # Linear change is to multiply the applied tensor by a w matrix and then + b

# # Change tensor shape
# t1 = torch.Tensor([[1, 2, 3], [4, 5, 6]])
# t1 = t1.transpose(0, 1)
# print(t1)