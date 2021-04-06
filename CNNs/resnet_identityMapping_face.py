import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import CNNs.basic_layer as basic_layer
from CNNs.arcFace import ArcMarginProduct
from CNNs.DictarcFace import DictArcMarginProduct
import torch.nn.functional as F

import torch

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        if downsample is None:
            self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.downsample is not None:
            before = x
            residual = self.downsample(before)
        else:
            residual = x
            before = self.bn1(x)
            before = self.relu(before)


        out = self.conv1(before)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)



        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        if downsample is None:
                self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        if self.downsample is not None:
            before = x
            residual = self.downsample(before)
        else:
            residual = x
            before = self.bn0(x)
            before = self.relu(before)

        out = self.conv1(before)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)




        out += residual

        return out


class ResNet_feature_extraction_net(nn.Module):

    def __init__(self, block, layers, planes, feature_len=256):
        self.inplanes = 64
        super(ResNet_feature_extraction_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0])
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2)

        self.bn0 = nn.BatchNorm2d(64)
        self.bn1 = nn.BatchNorm2d(planes[0]*block.expansion)
        self.bn2 = nn.BatchNorm2d(planes[1]*block.expansion)
        self.bn3 = nn.BatchNorm2d(planes[2]*block.expansion)
        self.bn4 = nn.BatchNorm2d(planes[3]*block.expansion)
        self.relu = nn.ReLU(inplace=True)


        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.view = basic_layer.View()
        self.feature_new = nn.Linear(planes[3]*7*7*block.expansion, feature_len)
        self.bn_feature = nn.BatchNorm1d(feature_len)

        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            # elif isinstance(m, nn.BatchNorm2d):
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.bn4(x)
        x = self.relu(x)

        # x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = self.view(x, [x.size(0), -1])
        x = self.feature_new(x)
        x = self.bn_feature(x)

        return x

class ResNet(nn.Module):

    def __init__(self, block, layers, planes, feature_len=256, num_classes_1=74974, num_classes_2=74974, label_dict=None):
        super(ResNet, self).__init__()
        self.feature_extraction_net = ResNet_feature_extraction_net(block, layers, planes, feature_len)
        self.dropout = nn.Dropout(p=0.25, inplace=True)
        #self.fc = nn.Linear(feature_len, num_classes, bias=False)
        self.fc = ArcMarginProduct(feature_len, num_classes_1, m = 0.5)
        self.fc2 = DictArcMarginProduct(feature_len, out_features=num_classes_2, out_features_test=num_classes_2, label_dict=label_dict, m = 0.5)
        #self.fc = ArcMarginProduct(feature_len, num_classes)


    def forward(self, x1, x2, y1, y2, label_set_2=None, testing=False, extract_feature = False):
        x = torch.cat((x1,x2),dim=0)
        x = self.feature_extraction_net(x)

        if extract_feature:
                return x

        x = self.dropout(x)
        split_size_1 = x1.size(0)
        split_size_2 = x2.size(0)
        x1 = x.narrow(0, 0, split_size_1)
        x2 = x.narrow(0, split_size_1, split_size_2)

        x1 = self.fc(x1, y1, testing)
        x2,y2_new = self.fc2(x2, y2, label_set_2, testing)
        return x1, x2, y2_new

    def forward_feat(self, x, testing = False, extract_feature = False):
        x = self.feature_extraction_net(x)

        if extract_feature:
                return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], [64,128,256,512], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3],[64,128,256,512],  **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], [64,128,256,512], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], [64,128,256,512], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], [64,128,256,512], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet200(pretrained=False, **kwargs):
    """Constructs a ResNet-200 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], [64,128,256,512], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def resnet269(pretrained=False, **kwargs):
    """Constructs a ResNet-269 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 30, 48, 3], [64,128,256,512], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
