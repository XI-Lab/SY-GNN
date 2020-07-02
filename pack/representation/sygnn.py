import torch
import torch.nn as nn

import pack.nn.acsf
import pack.nn.activations
import pack.nn.base
import pack.nn.cfconv
import pack.nn.neighbors
from pack.data import Structure
from pack.nn.cutoff import HardCutoff
from pack.representation import Interaction_module, batch_index_select


class SYGNN(nn.Module):
    """
    dense connection, share feature across symmetrical atoms
    Args:
        n_atom_basis (int): number of features used to describe atomic environments
        n_filters (int): number of filters used in continuous-filter convolution
        n_interactions (int): number of interaction blocks
        cutoff (float): cutoff radius of filters
        n_gaussians (int): number of Gaussians which are used to expand atom distances
        normalize_filter (bool): if true, divide filter by number of neighbors
            over which convolution is applied
        max_z (int): maximum allowed nuclear charge in dataset. This determines
            the size of the embedding matrix.
    """

    def __init__(self, n_atom_basis=128, n_filters=128, n_interactions=1, cutoff=5.0, n_gaussians=25,
                 normalize_filter=False, max_z=100,
                 cutoff_network=HardCutoff, trainable_gaussians=False,
                 distance_expansion=None, sym_label=True):
        super(SYGNN, self).__init__()

        if cutoff_network == 'hard':
            cutoff_network = HardCutoff
        else:
            cutoff_network = None

        # atom type embeddings
        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)

        # spatial features
        self.distances = pack.nn.neighbors.AtomDistances()
        if distance_expansion is None:
            self.distance_expansion = pack.nn.acsf.GaussianSmearing(
                0.0, cutoff, n_gaussians, trainable=trainable_gaussians)
        else:
            self.distance_expansion = distance_expansion

        # interaction network
        self.interactions = nn.ModuleList()
        for i in range(n_interactions):
            self.interactions.append(
                Interaction_module(n_atom_in=n_atom_basis * (i + 1),
                                   n_atom_out=n_atom_basis,
                                   n_spatial_basis=n_gaussians,
                                   n_filters=n_filters,
                                   cutoff_network=cutoff_network,
                                   cutoff=cutoff,
                                   normalize_filter=normalize_filter))
        self.sym_label = sym_label

    def forward(self, inputs):
        """
        Args:
            inputs (dict of torch.Tensor): format dictionary of input tensors.

        Returns:
            torch.Tensor: Final Atom-wise representation.
            torch.Tensor: Atom-wise representation of intermediate layers.
        """
        atomic_numbers = inputs[Structure.Z]
        positions = inputs[Structure.R]
        cell = inputs[Structure.cell]
        cell_offset = inputs[Structure.cell_offset]
        neighbors = inputs[Structure.neighbors]
        neighbor_mask = inputs[Structure.neighbor_mask]
        if self.sym_label:
            label = inputs[Structure.tags]

        # atom embedding
        x = self.embedding(atomic_numbers)

        # spatial features
        r_ij = self.distances(positions, neighbors, cell, cell_offset,
                              neighbor_mask=neighbor_mask)
        f_ij = self.distance_expansion(r_ij)

        for interaction in self.interactions:
            if self.sym_label:
                # As Equation (2) in the paper, the atoms in differnt primitive cells should has the same representation
                # In this step, we only  update the representation in the first primitive cell (and the center atom)
                v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij, cut=torch.max(label))
                # Duplicate the first primitive cell to other cells
                v_ = batch_index_select(v, 1, label)
                x = torch.cat([x, v_], dim=-1)
            else:
                v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij, cut=None)
                x = torch.cat([x, v], dim=-1)

        return x
