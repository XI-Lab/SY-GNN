import torch

def batch_index_select(input, dim, index):
    views = [input.shape[0], -1, 1]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index1 = (index.view(views).expand(expanse) - 1).clamp(0)
    return torch.gather(input, dim, index1) * index.float().clamp(0, 1).unsqueeze(-1)
