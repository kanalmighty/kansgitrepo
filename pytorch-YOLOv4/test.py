import torch
import pycuda.gpuarray as gpuarray
import numpy as np

x = torch.tensor([1,1,1,1])
y = torch.tensor([[2,2,2,2],[2,2,2,2]])
z = torch.tensor([[[2,2,2,2]],[[3,3,3,3]],[[4,4,4,4]]])
print(z+y)