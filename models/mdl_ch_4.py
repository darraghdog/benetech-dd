from torch.nn import functional as F
import timm
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn
# import segmentation_models_pytorch as smp
# from segmentation_models_pytorch.decoders.unet.model import UnetDecoder, SegmentationHead


class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        elif n_dims == 4:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1, 1)) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            if len(Y.shape) == 1:
                Y = coeffs * Y + (1 - coeffs) * Y[perm]
            elif len(Y.shape) == 2:
                Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
            else:
                Y = coeffs.view(-1, 1, 1) * Y + (1 - coeffs.view(-1, 1, 1)) * Y[perm]
        return X, Y

# from torchvision.ops import masks_to_boxes


from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import timm
from torch.distributions import Beta




def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")


class Mixup(nn.Module):
    def __init__(self, mix_beta, mixadd=False):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixadd = mixadd

    def forward(self, X, Y):

        bs = X.shape[0]
        n_dims = len(X.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(X.device)

        if n_dims == 2:
            X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
        elif n_dims == 3:
            X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
        else:
            X = coeffs.view(-1, 1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]

        if self.mixadd:
            Y = (Y + Y[perm]).clip(0, 1)
        else:
            if len(Y.shape) == 1:
                Y = coeffs * Y + (1 - coeffs) * Y[perm]
            else:
                Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]

        return X, Y



class Net(nn.Module):

    def __init__(self, cfg: Any):
        super(Net, self).__init__()

        self.cfg = cfg
        self.n_classes = self.cfg.n_classes
        self.backbone = timm.create_model(cfg.backbone, 
                                          pretrained=cfg.pretrained, 
                                          num_classes=0, 
                                          global_pool="", 
                                          in_chans=self.cfg.in_channels)

        if 'efficientnet' in cfg.backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]['num_chs']

        if cfg.pool == "gem":
            self.global_pool = GeM(p_trainable=cfg.gem_p_trainable)
        elif cfg.pool == "identity":
            self.global_pool = torch.nn.Identity()
        elif cfg.pool == "avg":
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.head = torch.nn.Linear(backbone_out, self.n_classes)
        self.return_embeddings = cfg.return_embeddings
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):

        x = batch['input']

        x = self.backbone(x)
        x = self.global_pool(x)
        x = x[:,:,0,0]

        logits = self.head(x)
        
        loss = self.loss_fn(logits,batch["target"])
        acc = (logits.argmax(1) == batch["target"]).float().mean()
        outputs = {}
        outputs['loss'] = loss
        outputs['loss_acc'] = acc
        if not self.training:
            outputs["logits"] = logits
            if self.return_embeddings:
                outputs["embedding"] = x
 

        return outputs
