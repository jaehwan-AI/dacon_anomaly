import timm
import torch.nn as nn


class Detector(nn.Module):
    def __init__(self, args):
        super(Detector, self).__init__()
        if args.model == 'resnetv2':
            self.model = timm.create_model('resnetv2_50x3_bitm_in21k', pretrained=args.pretrained,)

        self.model.reset_classifier(num_classes=88)
    
    def forward(self, x):
        return self.model(x)
