from torch.nn import functional as F
import timm
from torch import nn
import torch
from torch.nn import functional as F
from torch import nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.model import UnetDecoder, SegmentationHead


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


class Net(nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        
        self.n_classes = cfg.n_classes
        in_chans = 3
        
        
        self.encoder = timm.create_model(cfg.backbone, pretrained=cfg.pretrained,features_only=True,in_chans=in_chans)
        encoder_channels = tuple([in_chans] + [self.encoder.feature_info[i]['num_chs'] for i in range(len(self.encoder.feature_info))])
        self.decoder = UnetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            n_blocks=5,
            use_batchnorm=True,
            center=False,
            attention_type=None,
        )
        
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=self.n_classes,
            activation=None,
            kernel_size=3,
        )
        
        self.bce_seg = nn.BCEWithLogitsLoss()
        self.nms_kernel_size = cfg.nms_kernel_size
        self.nms_padding = cfg.nms_padding
        self.n_threshold = cfg.n_threshold
#         #if also classify
#         backbone_out = encoder_channels[-1]

#         if cfg.pool == 'max':
#             self.pool = nn.AdaptiveMaxPool2d(1)
#         elif cfg.pool == 'gem':
#             pass
#         else:
#             self.pool = nn.AdaptiveAvgPool2d(1)

#         self.head_in_units = backbone_out
#         self.head = nn.Linear(self.head_in_units, self.n_classes)

#         self.bce_cls = nn.BCEWithLogitsLoss()
        self.return_logits = cfg.return_logits
#         self.calculate_loss = cfg.calc_loss

    def forward(self, batch):

        x_in = batch['input']
        
        
        enc_out = self.encoder(x_in)
        
        decoder_out = self.decoder(*[x_in] + enc_out)
        x_seg = self.segmentation_head(decoder_out)
    
        output = {}
#         if (not self.training) & self.return_logits:

#                 pred = (x_seg.sigmoid().max(1)[0] > 0.5).long()
#                 box = torch.zeros((pred.shape[0],4), device=pred.device)

#                 not_empty = pred.sum((1,2))>10
#                 b = masks_to_boxes(pred[not_empty])
#                 box[not_empty] = b
#                 output['logits'] = box        
        loss = self.bce_seg(x_seg,batch['mask'].unsqueeze(1))
        output['loss'] = loss
        if not self.training:
            #nms
#             x_pred = x_seg.sigmoid()
#             x_pooled = F.max_pool2d(x_pred, kernel_size=self.nms_kernel_size, stride=1, padding=self.nms_padding)
#             x_pred[x_pred != x_pooled] = 0
#             n_pred = (x_pred[:,0] > self.n_threshold).sum((1,2))
#             loss_acc = (n_pred == batch['n']).float().mean()
#             output['loss_acc'] = loss_acc
#             output['n_pred'] = n_pred
            if self.return_logits:
                output['pred_mask'] = x_seg.sigmoid()
        
        return output
