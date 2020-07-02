import torch
from torch import nn as nn
import torch.nn.functional as F

__all__ = [
    'AttentionReadout'
]


class AttentionReadout(nn.Module):

    def __init__(self, f_in, f_attn, f_out):
        super(AttentionReadout, self).__init__()
        self.in_features = f_in
        self.out_features = f_out

        self.W1 = nn.Parameter(torch.zeros(size=(f_attn, f_in)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W2 = nn.Parameter(torch.zeros(size=(f_out, f_attn)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

    def forward(self, H: torch.Tensor):
        x = self.W1 @ H.permute(0, 2, 1)  # f_attn x atoms = (f_attn x f_in) @ (f_in x atoms)
        x = self.W2 @ torch.tanh(x)  # f_out x atoms = (f_out x f_attn) @ (f_attn x atoms)
        x = F.softmax(x, dim=2)
        y = x @ H  # f_out x f_in = (f_out x atoms) @ (atoms x f_in)
        return y.view(y.shape[0], -1)
