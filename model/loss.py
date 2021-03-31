import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyloss:
    def __init__(self, _device):
        self.loss = nn.CrossEntropyLoss().to(_device)

    def __call__(self, _logits, _targets):
        return self.loss(input=_logits, target=_targets)


def nll_loss(output, target):
    return F.nll_loss(output, target)
