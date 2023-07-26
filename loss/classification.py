import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def __init__(self, _class_ignore: int = -1, _reduction: str = 'mean') -> None:
        super(CrossEntropy, self).__init__()
        self.class_ignore = _class_ignore
        self.criteria = nn.CrossEntropyLoss(ignore_index=self.class_ignore, reduction=_reduction)

    def forward(self, _logits: torch.Tensor, _labels: torch.Tensor) -> torch.Tensor:
        return self.criteria(_logits, _labels)


class NegativeLogLikelihood(nn.Module):
    def __init__(self) -> None:
        super(NegativeLogLikelihood, self).__init__()

    def forward(self, _logits: torch.Tensor, _labels: torch.Tensor) -> torch.Tensor:
        return F.nll_loss(_logits, _labels)
