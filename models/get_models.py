import timm
import torch
import torch.nn as nn


class Detector(nn.Module):
    def __init__(self, args):
        super(Detector, self).__init__()
        if args.model == 'resnetv2':
            self.model = timm.create_model('resnetv2_50x3_bitm_in21k', pretrained=args.pretrained,)

        self.model.reset_classifier(num_classes=88)
    
    @torch.cuda.amp.autocast()
    def forward(self, x):
        return self.model(x)


class ClassClassifier(nn.Module):
    def __init__(self, num_features=1000, num_classes=15, drop=0.1):
        super().__init__()
        self.classifier = nn.Linear(num_features, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        return self.classifier(self.act(self.drop(x)))


class StateClassifier(nn.Module):
    def __init__(self, num_features=1000, num_classes=9, drop=0.1):
        super().__init__()
        self.classifier = nn.Linear(num_features, num_classes)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    @torch.cuda.amp.autocast()
    def forward(self, x, mask):
        x = self.classifier(self.act(self.drop(x)))
        x.masked_fill_(mask, -10000.)
        return x
