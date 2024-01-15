import torch.nn as nn
import torch
import numpy as np
from fusiontransformer.resnet import resnet34
from fusiontransformer.transformer import FusionTransformer
import torch.nn.functional as F

class fc(nn.Module):
    def __init__(self):
        super(fc, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 72),
            #nn.BatchNorm1d(72),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(72, 6),
            #nn.BatchNorm1d(6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class reshape(nn.Module):
    def __init__(self):
        super(reshape, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1000, 512),
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class sigmoid(nn.Module):
    def __init__(self):
        super(sigmoid, self).__init__()
        self.fc = nn.Sequential(
            nn.Sigmoid(),
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.encoder_img = resnet34()
        self.fusion = FusionTransformer()
        self.re1 = reshape()
        self.re2 = reshape()
        self.re3 = reshape()
        self.sig1 = sigmoid()
        self.sig2 = sigmoid()
        self.sig3 = sigmoid()
        self.fc = fc()

    def forward(self, img1, img2, img3):

        batch_size = img1.size(0)

        x1,out1 = self.encoder_img(img1)
        x1 = x1.view(batch_size, x1.size(1), -1)
        x1 = x1.permute(0, 2, 1)#[B,49,512]
        x2,out2 = self.encoder_img(img2)
        x2 = x2.view(batch_size, x2.size(1), -1)
        x2 = x2.permute(0, 2, 1)#[B,49,512]
        x3,out3 = self.encoder_img(img3)
        x3 = x3.view(batch_size, x3.size(1), -1)
        x3 = x3.permute(0, 2, 1)#[B,49,512]

        fusion,out = self.fusion(x1,x2,x3)
        
        return fusion, out