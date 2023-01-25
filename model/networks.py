'''
ResNet-18 뿐 아니라 여러 모델에서 작동할 수 있게
+
Model Freeze 적용 가능하게 해서 여러가지 실험 돌려볼 수 있는 implementation
'''

import torch
import torch.nn as nn
import torchvision.models


def torchhub_load(repo, model, **kwargs):
    network = torch.hub.load(repo, model=model, **kwargs)
    return network

class Identity(nn.Module):
    '''
    Identity layer
    '''
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_backbone(name, pretrained):
    '''
    ResNet 계열, RegNet 모델 사용 가능
    '''
    if name == 'resnet18':
        network = torchvision.models.resnet18(pretrained=pretrained)
        n_outputs = 512

    elif name == 'resnet50':
        network = torchvision.models.resnet50(pretrained=pretrained)
        n_outputs = 2048

    elif name == 'regnety':
        network = torchhub_load('facebook/research/swag', model='regnety_16gf', pretrained=pretrained)

        network.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )
        n_outputs = 3024

    elif name == 'resnet50_barlowtwins':
        network = torchhub_load('facebook/research/barlowtwins:main', model='resnet50')
        n_outputs = 2048

    else:
        raise ValueError(name)


    if name.startswith('resnet'):
        del network.fc
        network.fc = Identity()

    return network, n_outputs


BLOCKNAMES = {
    'resnet':{
        'stem' : ['conv1', 'bn1', 'relu', 'maxpool'],
        'block1': ['layer1'],
        'block2': ['layer2'],
        'block3': ['layer3'],
        'block4': ['layer4'],
    },

    'regnety':{
        'stem' : ['stem'],
        'block1': ['trunk_output.block1'],
        'block2': ['trunk_output.block2'],
        'block3': ['trunk_output.block3'],
        'block4': ['trunk_output.block4'],
    }
}


def get_module(module, name):
    for n, m in module.named_modules():
        return m


def build_blocks(model, block_name_dict):
    blocks = []
    for key, name_list in block_name_dict.items():
        block = nn.ModuleList()
        for module_name in name_list:
            module = get_module(model, module_name)
            block.append(module)

    return blocks


def _freeze(model):
    for p in model.parameters():
        p.requires_grad_(False)


class BaseModel(torch.nn.Module):
    '''
    모델 구축
    '''
    def __init__(self, input_shape, model, pretrained, freeze=None, feat_layers=None):
        super().__init__()

        self.networks, self.n_outputs = get_backbone(model, pretrained)

        if model == 'resnet18' or model.startswith('resnet50'):
            block_names = BLOCKNAMES['resnet']

        elif model == 'regnety':
            block_names = BLOCKNAMES['regnety']

        else:
            raise ValueError(model)

        
        self._features = []
        self.feat_layers = self.build_feature_hooks(feat_layers, block_names)
        self.blocks = build_blocks(self.network, block_names)
        self.freeze(freeze)

        self.freeze_bn()


    def freeze(self, freeze):
        if freeze is not None:
            if freeze == 'all':
                _freeze(self.network)
            else:
                for block in self.blocks[:freeze+1]:
                    _freeze(block)

    
    def hook(self, module, input, output):
        self._features.append(output)


    def build_feature_hooks(self, feats, block_names):
        assert feats in ['stem_block', 'block']

        if feats is None:
            return []

        if feats.startswith('stem'):
            last_stem_name = block_names['stem'][-1]
            feat_layers = [last_stem_name]
        else:
            feat_layers = []

        for name, module_names in block_names.items():
            if name == 'stem':
                continue

            module_name = module_names[-1]
            feat_layers.append(module_name)

        
        for n, m in self.network.named_modules():
            if n in feat_layers:
                m.register_forward_hook(self.hook)
        
        return feat_layers

    
    def forward(self, x):
        self.clear_features()
        out = self.dropout(self.network(x))

        return out, self._features

    
    def clear_features(self):
        self._features.clear()

    

def BaseFeaturizer(input_shape, model, pretrained, **kwargs):
    return BaseModel(input_shape, model, pretrained, **kwargs)