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
#      Attention Module
##############################


class Attention(nn.Module):
    def __init__(self, latent_dim):
        super(Attention, self).__init__()
        self.attention_module = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Softmax(dim=0)
            )

    def forward(self, x):
        attention_w = self.attention_module(x)
        return attention_w



##############################
#         Encoder
##############################


class Encoder(nn.Module):
    def __init__(self, latent_dim, attention):
        super(Encoder, self).__init__()
        self.feature_extractor_1, self.final_1 = self.create_feature_extractor_and_final_layers(resnet18, latent_dim)
        self.feature_extractor_2, self.final_2 = self.create_feature_extractor_and_final_layers(resnet18, latent_dim)
        self.feature_extractor_3, self.final_3 = self.create_feature_extractor_and_final_layers(resnet18, latent_dim)
        self.feature_extractor_4, self.final_4 = self.create_feature_extractor_and_final_layers(resnet18, latent_dim)
        self.feature_extractor_5, self.final_5 = self.create_feature_extractor_and_final_layers(resnet18, latent_dim)
        self.feature_extractor_6, self.final_6 = self.create_feature_extractor_and_final_layers(resnet18, latent_dim)
        self.feature_extractor_7, self.final_7 = self.create_feature_extractor_and_final_layers(resnet18, latent_dim)
        self.feature_extractor_8, self.final_8 = self.create_feature_extractor_and_final_layers(resnet18, latent_dim)
        self.feature_extractor_9, self.final_9 = self.create_feature_extractor_and_final_layers(resnet18, latent_dim)
        self.attention = attention
        self.attention_layer = Attention(latent_dim)

    def create_feature_extractor_and_final_layers(model, latent_dim):
        model = model(pretrained=True)
        trained_kernel = model.conv1.weight
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:,:] = torch.stack([torch.mean(trained_kernel, 1)]*1, dim=1)
        model.conv1 = new_conv

        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        final = nn.Sequential(
            nn.AlphaDropout(0.5),
            nn.Linear(model.fc.in_features, latent_dim), 
            nn.BatchNorm1d(latent_dim, momentum=0.01),
            nn.ReLU(),

        )


        return feature_extractor, final



    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        
        with torch.no_grad():
            x1 = self.feature_extractor_1(x1)
            x2 = self.feature_extractor_2(x2)
            x3 = self.feature_extractor_3(x3)
            x4 = self.feature_extractor_4(x4)
            x5 = self.feature_extractor_5(x5)
            x6 = self.feature_extractor_6(x6)
            x7 = self.feature_extractor_7(x7)
            x8 = self.feature_extractor_8(x8)
            x9 = self.feature_extractor_9(x9)


            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            x3 = x3.view(x3.size(0), -1)
            x4 = x4.view(x4.size(0), -1)
            x5 = x5.view(x5.size(0), -1)
            x6 = x6.view(x6.size(0), -1)
            x7 = x7.view(x7.size(0), -1)
            x8 = x8.view(x8.size(0), -1)
            x9 = x9.view(x9.size(0), -1)

            x1 = self.final_1(x1)
            x2 = self.final_2(x2)
            x3 = self.final_3(x3)
            x4 = self.final_4(x4)
            x5 = self.final_5(x5)
            x6 = self.final_6(x6)
            x7 = self.final_7(x7)
            x8 = self.final_8(x8)
            x9 = self.final_9(x9)


        if self.attention:
            attention_weight = self.attention_layer(torch.stack((x1,x2,x3,x4,x5,x6,x7,x8,x9)))

            x1 = torch.mul(x1, attention_weight[0])
            x2 = torch.mul(x1, attention_weight[1])
            x3 = torch.mul(x1, attention_weight[2])
            x4 = torch.mul(x1, attention_weight[3])
            x5 = torch.mul(x1, attention_weight[4])
            x6 = torch.mul(x1, attention_weight[5])
            x7 = torch.mul(x1, attention_weight[6])
            x8 = torch.mul(x1, attention_weight[7])
            x9 = torch.mul(x1, attention_weight[8])

        x = [x1, x2, x3, x4, x5, x6, x7, x8, x9]

        x = torch.cat(x, 1)

        #print(self.final(x))
        
        return x


##############################
#         ConvLSTM
##############################

#dim=-1 is the right most dimension

class ConvLSTM(nn.Module):
    def __init__(
        self, num_classes, latent_dim=1024,hidden_dim=2048, attention=False,
    ):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder(latent_dim, attention)
        latent_dim = int(latent_dim*5)
        hidden_dim2 = int(hidden_dim/4)
        hidden_dim3 = int(hidden_dim2/4)
        hidden_dim4 = int(hidden_dim3/4)
        #self.inceptionresnetV2 = InceptionResNetV2()
        self.output_layers_final = nn.Sequential(
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

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8, x9):
        
        x = self.encoder(x1, x2, x3, x4, x5, x6, x7, x8, x9) 
        
        return self.output_layers_final(x)