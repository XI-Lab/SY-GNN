r"""
Classes for output modules.
"""

import numpy as np
import torch
import torch.nn as nn

import pack.nn.activations
import pack.nn.base
import pack.nn.blocks
from pack.data import Structure
from pack.atomistic import OutputModule
from pack.datasets.qm_sym import properties
from torch.nn import functional as F
from pack.nn import AttentionReadout


class Atomwise(OutputModule):
    """
    Predicts atom-wise contributions and accumulates global prediction, e.g. for the energy.

    Args:
        n_in (int): input dimension of representation (default: 128)
        requires_dr (bool): True, if derivative w.r.t. atom positions is required (default: False)
        mean (torch.FloatTensor): mean of property (default: None)
        stddev (torch.FloatTensor): standard deviation of property (default: None)
        atomref (torch.Tensor): reference single-atom properties
        train_embeddings (bool): if set to true, atomref will be ignored and learned from data (default: None)

    Returns:
        tuple: prediction for property
        If return_contributions is true additionally returns atom-wise contributions.
        If requires_dr is true additionally returns derivative w.r.t. atom positions.
    """

    def __init__(self, n_in=128,
                 requires_dr=False, mean=None, stddev=None,
                 atomref=None,
                 train_embeddings=False):
        super(Atomwise, self).__init__(requires_dr)

        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(
                torch.from_numpy(atomref[:, :-1].astype(np.float32)),
                freeze=train_embeddings)
        self.get_representation = pack.nn.base.GetItem('representation')
        self.dense1 = nn.Linear(n_in, n_in * 2)
        self.dense2 = nn.Linear(n_in * 2, n_in * 2)
        self.readout_avg = AttentionReadout(n_in * 2, n_in * 2, 4)
        self.dense_avg = nn.Linear(n_in * 8, len(properties) - 1)  # 3) remove mu
        # for mu
        self.atom_pool = pack.nn.base.Aggregate(axis=1, mean=True)
        self.dense_mu = nn.Linear(n_in * 2, 1)  # 3)

        # Make standardization separate
        self.standardize = pack.nn.base.ScaleShift(mean[:-1], stddev[:-1])

    def forward(self, inputs):
        r"""
        predicts atomwise property
        """
        atomic_numbers = inputs[Structure.Z]
        positions = inputs[Structure.R]
        atom_mask = inputs[Structure.atom_mask][:, :, None]
        x = self.get_representation(inputs)
        x = F.leaky_relu(self.dense1(x))
        x = F.leaky_relu(self.dense2(x))  # shape: [batch, atoms, f_in * 2]


        x_avg = self.readout_avg(x)  # shape: [batch, f_in * 2 * 4]
        y_avg = self.dense_avg(x_avg)  # shape: [batch, properties]

        y = self.standardize(y_avg)

        charges = self.dense_mu(x) * atom_mask  # mu
        yi_mu = positions * charges
        y_mu = self.atom_pool(yi_mu)

        y_mu = torch.norm(y_mu, dim=1, keepdim=True)

        if self.atomref is not None:  # for properties U0 U G H
            y0 = self.atomref(atomic_numbers).sum(axis=1)
            y = y + y0

        y = torch.cat([y, y_mu], dim=1)
        result = {}
        for i in range(len(properties)):
            result[properties[i]] = y[:, i]

        return result
