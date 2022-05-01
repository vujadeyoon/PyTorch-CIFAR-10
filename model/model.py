import torch.nn.functional as F
from base import BaseModel
from torchvision.models import resnet18 as resnet18_1
from model.arxiv.resnet import ResNet18 as resnet18_2


class ResNet_18_1(BaseModel):
    def __init__(self, num_class=10):
        super().__init__()
        self.model = resnet18_1(pretrained=False, num_classes=num_class)

    def forward(self, x):
        x = self.model(x)
        res = F.log_softmax(x, dim=1)

        return res


class ResNet_18_2(BaseModel):
    def __init__(self, num_class=10):
        super().__init__()
        self.model = resnet18_2(_num_classes=num_class)

    def forward(self, x):
        res = self.model(x)

        return res