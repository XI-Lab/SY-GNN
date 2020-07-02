import torch
from torch import nn as nn

def hard_cutoff(distances, cutoff=5.0):
    """
    Hard cutoff function.

    Args:
        distances (torch.Tensor): Interatomic distances (Nbatch x Nat x Nneigh)
        cutoff (float): Cutoff value, all values beyond are set to 0

    Returns:
        torch.Tensor: Tensor holding values of the cutoff function (Nbatch x Nat x Nneigh)
    """
    mask = (distances <= cutoff).float()
    return distances * mask


class HardCutoff(nn.Module):
    """
    Class wrapper for hard cutoff function.

    Args:
        cutoff (float): Cutoff radius.
    """

    def __init__(self, cutoff=5.0):
        super(HardCutoff, self).__init__()
        self.register_buffer('cutoff', torch.FloatTensor([cutoff]))

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Interatomic distances.

        Returns:
            torch.Tensor: Values of cutoff function.
        """
        return hard_cutoff(distances, cutoff=self.cutoff)

