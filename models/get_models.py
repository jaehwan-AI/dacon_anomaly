import timm
import torch.nn as nn


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.model = timm.create_model('', pretrained=True,)

        self.model.reset_classifier(num_classes=88)
    
    def forward(self, x):
        return self.model(x)
