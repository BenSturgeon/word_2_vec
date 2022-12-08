import torch


# 1. Create a tensor with 2 dimensions
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[1, 2, 3], [4, 5, 6]])
b.unsqueeze(2)
mul_result = torch.mul(a, b)
print(mul_result)