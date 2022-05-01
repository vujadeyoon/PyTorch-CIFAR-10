import torch


class Accuracy(object):
    def __init__(self) -> None:
        super(Accuracy, self).__init__()

    def __call__(self, _output: torch.Tensor, _target: torch.Tensor) -> float:
        with torch.no_grad():
            pred = torch.argmax(_output, dim=1)
            assert pred.shape[0] == len(_target)
            correct = 0
            correct += torch.sum(_target.eq(pred)).item()
        return correct / len(_target)


class AccuarcyTopK(object):
    def __init__(self, _k: int = 1) -> None:
        super(AccuarcyTopK, self).__init__()
        self.k = _k

    def __call__(self, _output: torch.Tensor, _target: torch.Tensor) -> float:
        with torch.no_grad():
            pred = torch.topk(_output, self.k, dim=1)[1]
            assert pred.shape[0] == len(_target)
            correct = 0
            for i in range(self.k):
                correct += torch.sum(_target.eq(pred[:, i])).item()
        return correct / len(_target)
