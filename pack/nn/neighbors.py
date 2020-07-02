import torch
from torch import nn as nn


def atom_distances(positions, neighbors, cell=None, cell_offsets=None,
                   return_vecs=False,
                   return_directions=False, neighbor_mask=None):
    """
    Use advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors. Indices of the
    neighbors to consider are stored in neighbors.

    Args:
        positions (torch.Tensor): Atomic positions, differentiable torch Variable (B x N_at x 3)
        neighbors (torch.Tensor): Indices of neighboring atoms (B x N_at x N_nbh)
        cell (torch.Tensor): cell for periodic systems (B x 3 x 3)
        cell_offsets (torch.Tensor): offset of atom in cell coordinates (B x N_at x N_nbh x 3)
        return_directions (bool): If true, also return direction cosines.
        neighbor_mask (torch.Tensor, optional): Boolean mask for neighbor positions. Required for the stable
                                                computation of forces in molecules with different sizes.

    Returns:
        torch.Tensor: Distances of every atom to its neighbors (B x N_at x N_nbh)
        torch.Tensor: Direction cosines of every atom to its neighbors (B x N_at x N_nbh x 3) (optional)
    """

    # Construct auxiliary index vector
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch, device=positions.device, dtype=torch.long)[:,
            None, None]
    # Get atomic positions of all neighboring indices
    pos_xyz = positions[idx_m, neighbors[:, :, :], :]

    # Subtract positions of central atoms to get distance vectors
    dist_vec = pos_xyz - positions[:, :, None, :]

    # add cell offset
    if cell is not None:
        B, A, N, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)
        dist_vec += offsets

    # Compute vector lengths
    distances = torch.norm(dist_vec, 2, 3)

    if neighbor_mask is not None:
        # Avoid problems with zero distances in forces (instability of square root derivative at 0)
        # This way is neccessary, as gradients do not work with inplace operations, such as e.g.
        # -> distances[mask==0] = 0.0
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
        distances = tmp_distances

    if return_directions or return_vecs:
        tmp_distances = torch.ones_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]

        if return_directions:
            dist_vec = dist_vec / tmp_distances[:, :, :, None]
        return distances, dist_vec
    return distances


class AtomDistances(nn.Module):
    """
    Layer that calculates all pair-wise distances between atoms.

    Use advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors. Indices of the
    neighbors to consider are stored in neighbors.

    Args:
        return_directions (bool): If true, also return direction cosines.
    """

    def __init__(self, return_directions=False):
        super(AtomDistances, self).__init__()
        self.return_directions = return_directions

    def forward(self, positions, neighbors, cell=None, cell_offsets=None,
                neighbor_mask=None):
        """
        Args:
            positions (torch.Tensor): Atomic positions, differentiable torch Variable (B x N_at x 3)
            neighbors (torch.Tensor): Indices of neighboring atoms (B x N_at x N_nbh)
            cell (torch.tensor): cell for periodic systems (B x 3 x 3)
            cell_offsets (torch.Tensor): offset of atom in cell coordinates (B x N_at x N_nbh x 3)
            neighbor_mask (torch.Tensor, optional): Boolean mask for neighbor positions. Required for the stable
                                                    computation of forces in molecules with different sizes.

        Returns:
            torch.Tensor: Distances of every atoms to its neighbors (B x N_at x N_nbh)
        """
        return atom_distances(positions, neighbors, cell, cell_offsets,
                              return_directions=self.return_directions,
                              neighbor_mask=neighbor_mask)
