import torch
import torch.nn as nn

import pack.nn.acsf
import pack.nn.activations
import pack.nn.base
import pack.nn.cfconv
import pack.nn.neighbors
from pack.nn.cutoff import HardCutoff


class Interaction_module(nn.Module):
    """
    Interaction for modeling quantum interactions of atomistic
    systems with cosine cutoff.

    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_spatial_basis (int): number of input features of filter-generating networks
        n_filters (int): number of filters used in continuous-filter convolution
        normalize_filter (bool): if true, divide filter by number of neighbors over which convolution is applied
    """

    def __init__(self, n_atom_in, n_atom_out, n_spatial_basis, n_filters, cutoff,
                 cutoff_network=HardCutoff, normalize_filter=False):
        super(Interaction_module, self).__init__()

        # initialize filters
        self.filter_network = nn.Sequential(
            pack.nn.base.Dense(n_spatial_basis, n_filters,
                               activation=pack.nn.activations.shifted_softplus),
            pack.nn.base.Dense(n_filters, n_filters)
        )
        if cutoff_network is not None:
            self.cutoff_network = cutoff_network(cutoff)
        else:
            self.cutoff_network = None

        # initialize interaction blocks
        self.cfconv = pack.nn.cfconv.CFConv(n_atom_in, n_filters,
                                            n_atom_out,
                                            self.filter_network,
                                            cutoff_network=self.cutoff_network,
                                            activation=pack.nn.activations.shifted_softplus,
                                            normalize_filter=normalize_filter)
        self.dense = pack.nn.base.Dense(n_atom_out, n_atom_out)

    def forward(self, x, r_ij, neighbors, neighbor_mask, f_ij=None, cut=None):
        """
        Args:
            x (torch.Tensor): Atom-wise input representations.
            r_ij (torch.Tensor): Interatomic distances.
            neighbors (torch.Tensor): Indices of neighboring atoms.
            neighbor_mask (torch.Tensor): Mask to indicate virtual neighbors
                introduced via zeros padding.
            f_ij (torch.Tensor): Use at your own risk.
            cut (torch.Tensor): The size of primitive cell.

        Returns:
            torch.Tensor: Representation.
        """
        v = self.cfconv(x, r_ij, neighbors, neighbor_mask, f_ij, cut=cut)
        v = self.dense(v)
        return v
