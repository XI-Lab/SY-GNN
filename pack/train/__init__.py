r"""
Classes to manage the training process.

pack.train.Trainer encapsulates the training loop. It also can automatically monitor the performance on the
validation set and contains logic for checkpointing. The training process can be customized using Hooks which derive
from pack.train.Hooks.

"""


from .hooks import *
from .trainer import Trainer