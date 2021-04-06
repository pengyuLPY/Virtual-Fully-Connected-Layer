from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import math
from torch.nn import Parameter
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, out_features=200, s=32.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label, testing=False):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = 1.0 - torch.pow(cosine, 2)
        sine = torch.where(sine > 0 , sine, torch.zeros(sine.size(), device='cuda'))
        sine = torch.sqrt(sine)
        #sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if testing == False:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
            return output
        else:
            output0 = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output00 = output0*self.s

            output1 = cosine
            output11 = output1*self.s
            return output00, output11
