import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torchvision.models import resnet18, resnet34, resnet50,resnet101,\
                                resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2

import torch.utils.model_zoo as model_zoo
import os
import sys



##############################
#         Encoder
##############################


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        model = resnet18(pretrained=True)
        trained_kernel = model.conv1.weight
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*1, dim=1)
        model.conv1 = new_conv
        self.feature_extractor_7 = nn.Sequential(*list(model.children())[:-1])
        self.final_7 = nn.Sequential(
            nn.AlphaDropout(0.5),
            nn.Linear(model.fc.in_features, latent_dim), 
            nn.BatchNorm1d(latent_dim, momentum=0.01),
            nn.ReLU(),

        )


    def forward(self, x):
        
        #with torch.no_grad():
        x = self.feature_extractor_7(x)
        x = x.view(x.size(0), -1)
        #print(self.final(x))
        
        return self.final_7(x)


##############################
#         ConvLSTM
##############################

#dim=-1 is the right most dimension

class ConvLSTM(nn.Module):
    def __init__(
        self, num_classes, latent_dim=1024,hidden_dim=2048
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim)
        hidden_dim2 = int(hidden_dim/4)
        hidden_dim3 = int(hidden_dim2/4)
        hidden_dim4 = int(hidden_dim3/4)
        #self.inceptionresnetV2 = InceptionResNetV2()
        self.output_layers = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.AlphaDropout(0.5),
            nn.Linear(hidden_dim2,hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3,hidden_dim4),
            nn.BatchNorm1d(hidden_dim4, momentum=0.01),
            nn.ReLU(),
            nn.AlphaDropout(0.5),
            nn.Linear(hidden_dim4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        x = self.encoder(x) 
        
        return self.output_layers(x)