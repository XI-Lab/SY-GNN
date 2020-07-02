from collections import Iterable

import torch.nn as nn
from pack.data import Structure


class AtomisticModel(nn.Module):
    """
    Assembles an atomistic model from a representation module and one or more output modules.

    Returns either the predicted property or a dict (output_module -> property),
    depending on whether output_modules is a Module or an iterable.

    Args:
        representation(nn.Module): neural network that builds atom-wise features of shape batch x atoms x features
        output_modules(nn.OutputModule or iterable of nn.OutputModule): predicts desired property from representation


    """

    def __init__(self, representation, output_modules):
        super(AtomisticModel, self).__init__()

        self.representation = representation

        if isinstance(output_modules, Iterable):
            self.output_modules = nn.ModuleList(output_modules)
            self.requires_dr = False
            for o in self.output_modules:
                if o.requires_dr:
                    self.requires_dr = True
                    break
        else:
            self.output_modules = output_modules
            self.requires_dr = output_modules.requires_dr

    def forward(self, inputs):
        r"""
        predicts property

        """
        if self.requires_dr:
            inputs[Structure.R].requires_grad_()
        out_info = self.representation(inputs)

        inputs['representation'] = out_info

        if isinstance(self.output_modules, nn.ModuleList):
            outs = {}
            for output_module in self.output_modules:
                outs[output_module] = output_module(inputs)
        else:
            outs = self.output_modules(inputs)
        return outs


class OutputModule(nn.Module):
    r"""
    Base class for output modules.

    Args:
        requires_dr (bool): specifies if the derivative of the ouput is required
    """

    def __init__(self, requires_dr=False):
        self.requires_dr = requires_dr
        super(OutputModule, self).__init__()

    def forward(self, inputs):
        r"""
        Should be overwritten
        """
        raise NotImplementedError
