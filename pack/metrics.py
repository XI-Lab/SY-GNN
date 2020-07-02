import numpy as np
import torch
from pack.data import Structure

__all__ = ['ModelBias', 'MeanSquaredError', 'RootMeanSquaredError', 'MeanAbsoluteError', 'RootMeanSquaredErrorInt',
           'MeanAbsoluteErrorInt']


class Metric:
    r"""
    Base class for all metrics.

    Metrics measure the performance during the training and evaluation.

    Args:
        name (str): name used in logging for this metric. If set to `None`, `MSE_[target]` will be used (Default: None)
    """

    def __init__(self, name=None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    def add_batch(self, batch, result):
        """ Add a batch to calculate the metric on """
        raise NotImplementedError

    def aggregate(self):
        """ Aggregate metric over all previously added batches """
        raise NotImplementedError

    def reset(self):
        """ Reset the metric after aggregation to collect new batches """
        pass


class ModelBias(Metric):
    r"""
    Calculates the bias of the model. For non-scalar quantities, the mean of all components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs (Default: None)
        name (str): name used in logging for this metric. If set to `None`, `MSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        name = 'Bias_' + target if name is None else name
        super(ModelBias, self).__init__(name)
        self.target = target
        self.model_output = model_output
        self.element_wise = element_wise
        self.l2loss = 0.
        self.n_entries = 0.

    def reset(self):
        self.l2loss = 0.
        self.n_entries = 0.

    def _get_diff(self, y, yp):
        return y - yp

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff.view(-1)).detach().cpu().data.numpy()
        if self.element_wise:
            self.n_entries += torch.sum(batch[Structure.atom_mask]) * y.shape[-1]
        else:
            self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.l2loss / self.n_entries


class MeanSquaredError(Metric):
    r"""
    Metric for mean square error. For non-scalar quantities, the mean of all components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs (Default: None)
        name (str): name used in logging for this metric. If set to `None`, `MSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False,
                 mean=None, stddev=None):
        name = 'MSE_' + target if name is None else name
        super(MeanSquaredError, self).__init__(name)
        self.target = target
        self.model_output = model_output
        self.element_wise = element_wise
        self.l2loss = 0.
        self.n_entries = 0.
        self.mean = mean
        self.stddev = stddev

    def reset(self):
        self.l2loss = 0.
        self.n_entries = 0.

    def _get_diff(self, y, yp):
        diff = y - yp
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target].squeeze()
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
                # result = result * (self.stddev + 1e-9) + self.mean  # denormalize the predicted value
            yp = result

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()
        if self.element_wise:
            self.n_entries += torch.sum(batch[Structure.atom_mask]) * y.shape[-1]
        else:
            self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.l2loss / self.n_entries


class RootMeanSquaredError(MeanSquaredError):
    r"""
    Metric for root mean square error. For non-scalar quantities, the mean of all components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs (Default: None)
        name (str): name used in logging for this metric. If set to `None`, `RMSE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False,
                 mean=None, stddev=None):
        name = 'RMSE_' + target if name is None else name
        super(RootMeanSquaredError, self).__init__(target, model_output, name,
                                                   element_wise=element_wise, mean=mean, stddev=stddev)

    def aggregate(self):
        return np.sqrt(self.l2loss / self.n_entries)


class MeanAbsoluteError(Metric):
    r"""
    Metric for mean absolute error. For non-scalar quantities, the mean of all components is taken.

    Args:
        target (str): name of target property
        model_output (int, str): index or key, in case of multiple outputs (Default: None)
        name (str): name used in logging for this metric. If set to `None`, `MAE_[target]` will be used (Default: None)
        element_wise (bool): set to True if the model output is an element-wise property (forces, positions, ...)
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False,
                 mean=None, stddev=None):
        name = 'MAE_' + target if name is None else name
        super(MeanAbsoluteError, self).__init__(name)
        self.target = target
        self.element_wise = element_wise
        self.model_output = model_output
        self.l1loss = 0.
        self.n_entries = 0.
        self.mean = mean
        self.stddev = stddev

    def reset(self):
        self.l1loss = 0.
        self.n_entries = 0.

    def _get_diff(self, y, yp):
        diff = y - yp
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target].squeeze()
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
                    # print(result.shape)
            else:
                result = result[self.model_output]
                # result = result * (self.stddev + 1e-9) + self.mean # denormalize the predicted value
            yp = result

        diff = self._get_diff(y, yp)
        self.l1loss += torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()
        if self.element_wise:
            self.n_entries += torch.sum(batch[Structure.atom_mask]) * y.shape[-1]
        else:
            self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.l1loss / self.n_entries


class ClassPrecentageError(Metric):
    r"""
    Metric for classification prencentage error. For non-scalar quantities, the mean of all components is taken.
    Args:
        target (str): name of target property
        name (str): name used in logging for this metric. If set to `None`, `MAE_[target]` will be used (Default: None)
    """

    def __init__(self, target, name=None):
        name = 'CE_' + target if name is None else name
        super(ClassPrecentageError, self).__init__(name)
        self.target = target
        self.precentloss = 0.
        self.n_entries = 0.

    def reset(self):
        self.precentloss = 0.
        self.n_entries = 0.

    def _get_diff(self, y, yp):
        yp = yp.max(dim=1)[1]
        diff = (yp != y.squeeze(dim=1).long()).sum().float()
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target]
        if type(result) is dict:
            yp = result['y']
        else:
            yp = result

        diff = self._get_diff(y, yp)
        self.precentloss += diff.detach().cpu().data.numpy()
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.precentloss / self.n_entries

class MeanAbsoluteErrorInt(MeanAbsoluteError):
    r"""
            Int the result before calculate MAE.
        """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        super(MeanAbsoluteErrorInt, self).__init__(target, model_output, name,
                                                   element_wise=element_wise)

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = torch.round(result.clamp(min=0.))
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = torch.round(result.clamp(min=0.))

        diff = self._get_diff(y, yp)
        self.l1loss += torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()
        if self.element_wise:
            self.n_entries += torch.sum(batch[Structure.atom_mask]) * y.shape[-1]
        else:
            self.n_entries += np.prod(y.shape)


class RootMeanSquaredErrorInt(RootMeanSquaredError):
    r"""
        Int the result before calculate RMSE.
    """

    def __init__(self, target, model_output=None, name=None, element_wise=False):
        super(RootMeanSquaredErrorInt, self).__init__(target, model_output, name,
                                                      element_wise=element_wise)

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = torch.round(result.clamp(min=0.))
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = torch.round(result.clamp(min=0.))

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()
        if self.element_wise:
            self.n_entries += torch.sum(batch[Structure.atom_mask]) * y.shape[-1]
        else:
            self.n_entries += np.prod(y.shape)
