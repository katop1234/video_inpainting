import torch

### CONSTS
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).cuda()
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).cuda()
### CONSTS