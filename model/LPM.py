'''
LPM implementation

"Each feature map is reduced
to a fixed dimensional feature vector through a global average
pooling (GAP) layer and a fully-connected layer. Then,
all features are concatenated and pass through another fully-connected layer..."

From (https://arxiv.org/abs/1905.03677)
'''

import torch
import torch.nn as nn
import torch.functional as F


class LPM(nn.Module):
    '''
    network.py 의 모델에서 받은 out_features는 아래와 같음

    Ex) 
    ResNet-50

    Stage 2 -> (96, 256, 56, 56)
    Stage 3 -> (96, 512, 28, 28)
    Stage 4 -> (96, 1024, 14, 14)
    Stage 5 -> (96, 2048, 7, 7)
    '''

    def __init__(self, feature_shapes, in_dim):
        super(LPM, self).__init__()

        self.in_dim = in_dim
        self.shapes = feature_shapes
        self.fc = nn.Linear(4 * self.in_dim, 1)

        self.modules = []
        for shape in self.shapes:
            self.modules.append(self.make_sub(shape))


    def forward(self, x):
        # x : [*feataures]

        out = []
        for i in range(len(x)):
            out.append(self.modules[i](x[i]))

        return self.fc(torch.cat(out, 1))


    def make_sub(self, shape):
        sub_module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(shape[1], self.in_dim),
            nn.ReLU()
        )
        return sub_module


def check():
    out_features = [(96, 256, 56, 56), (96, 512, 28, 28), (96, 1024, 14, 14), (96, 2048, 7, 7)]
    in_dim = 128

    lpm = LPM(out_features, in_dim)
    x = [torch.randn((i)) for i in out_features]

    assert lpm(x).shape == (96, 1), "Wrong Output shape"