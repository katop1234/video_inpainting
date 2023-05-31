import torch
import numpy as np

### CONSTS
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
np_mean = mean.numpy()
np_std = std.numpy()
### CONSTS