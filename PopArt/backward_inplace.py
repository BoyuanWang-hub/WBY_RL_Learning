import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

if __name__ == '__main__':
    x = torch.tensor([1.], requires_grad=True)
    w = torch.tensor([2.], requires_grad=True)

    # create a computation graph as follows:
    # x → y → z
    #       ↗
    #     w
    y = x ** 2
    z = w * y + 1.

    w.data *= 2
    # w.data = w.data * 2 ### 两种不同的赋值方式 ###

    z.backward()

    print(x.grad)
    print(w.grad)