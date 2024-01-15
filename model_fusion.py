import torch.nn as nn
import torch
import torch.nn.functional as F
from fusiontransformer.img_model import Classifier
from tabtransformer.cli_model import TabTransformer

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class CliEncoder(nn.Module):
    def __init__(self):
        super(CliEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class ImgDecoder(nn.Module):
    def __init__(self):
        super(ImgDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class CliDecoder(nn.Module):
    def __init__(self):
        super(CliDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class FusionClassifier(nn.Module):
    def __init__(self):
        super(FusionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(544, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class DecoupleFusioner(nn.Module):
    def __init__(self):
        super(DecoupleFusioner, self).__init__()

        self.img_extractor = Classifier()
        self.cli_extractor = TabTransformer()
        self.algn = nn.Linear(32, 512)
        self.img_encoder = ImgEncoder()
        self.cli_encoder = CliEncoder()
        self.shared_encoder = SharedEncoder()
        self.cli_decoder = CliDecoder()
        self.img_decoder = ImgDecoder()
        self.classifier = FusionClassifier()
        self.apply(_init_vit_weights)

    def forward(self, img1, img2, img3, cli):

        img_feature,img_out = self.img_extractor(img1,img2,img3)
        cli_feature,cli_out = self.cli_extractor(cli)

        img_related = self.shared_encoder(img_feature)
        cli_related = self.shared_encoder(self.algn(cli_feature))

        img_unrelated = self.img_encoder(img_feature)
        cli_unrelated = self.cli_encoder(cli_feature)

        out = self.classifier(torch.cat([img_feature,cli_feature],dim=1))

        img_feature_rec = self.img_decoder(torch.cat([cli_related,img_unrelated],dim=1))
        cli_feature_rec = self.cli_decoder(torch.cat([img_related,cli_unrelated],dim=1))

        # return out,img_related,cli_related,img_unrelated,cli_unrelated
        return out,img_feature_rec,cli_feature_rec,img_feature,cli_feature

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)