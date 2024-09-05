# elatentlpips: https://github.com/mingukkang/elatentlpips
# The CC-BY-NC license
# See license file or visit https://github.com/mingukkang/elatentlpips for details

# elatentlpips/vgg16.py


from collections import namedtuple
from collections import OrderedDict
import os
import torch
from torchvision import models as tv
from copy import deepcopy


class CalibratedLatentVGG16BN(torch.nn.Module):
    def __init__(self, num_latent_channels, encoder, pretrained=False, requires_grad=True):
        super(CalibratedLatentVGG16BN, self).__init__()
        model = LatentVGG16BN(num_latent_channels, requires_grad=True)
        if pretrained:
            # Load checkpoint
            url_path = f'https://huggingface.co/Mingguksky/elatentlpips/resolve/main/elatentlpips_ckpt/{encoder}_latent_vgg16.pth.tar'
            if not os.path.exists('./ckpt'):
                os.makedirs('./ckpt', exist_ok=True)
            
            if not os.path.exists(f'./ckpt/{encoder}_latent_vgg16.pth.tar'):
                torch.hub.download_url_to_file(url_path, f'./ckpt/{encoder}_latent_vgg16.pth.tar')
            ckpt = torch.load(f'./ckpt/{encoder}_latent_vgg16.pth.tar')
            new_state_dict = OrderedDict()
            for k, v in ckpt['state_dict'].items():
                name = k[7:] if k.startswith('module.') else k  # remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
            print(f"Loading VGG16bn-base from: {url_path} | Top-1 acc.: {ckpt['best_acc1']}")

            # Copy the pretrained features
            vgg_pretrained_features = deepcopy(model.model.features)
            
            # Delete the model and checkpoint to free up memory
            del model
            del ckpt
        else:
            vgg_pretrained_features = tv.vgg16_bn(pretrained=False).features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5

        for x in range(7):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 14):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(14, 24):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(24, 34):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(34, 44):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out
    
class LatentVGG16BN(torch.nn.Module):
    def __init__(self, num_latent_channels, requires_grad=True, pretrained=False):
        super(LatentVGG16BN, self).__init__()
        self.model = tv.vgg16_bn(pretrained=pretrained)
        self.model.features[0] = torch.nn.Conv2d(num_latent_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        
        # Find the indices of the first three MaxPool2d layers
        delete_idx = [i for i, layer in enumerate(self.model.features) if isinstance(layer, torch.nn.MaxPool2d)][:3]

        # Replace the identified MaxPool2d layers with nn.Identity()
        for idx in delete_idx:
            self.model.features[idx] = torch.nn.Identity()

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = self.model(X)
        return out
    
class VGG16BN(torch.nn.Module):
    def __init__(self, requires_grad=True, pretrained=False):
        super(VGG16BN, self).__init__()
        self.model = tv.vgg16_bn(pretrained=pretrained)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        out = self.model(X)
        return out
