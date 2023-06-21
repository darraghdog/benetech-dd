from typing import Any, Dict
import copy
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
import timm
from torch.distributions import Beta
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence
from transformers import DetrConfig, DetrForObjectDetection#, DetrHungarianMatcher
from transformers.models.detr.modeling_detr import DetrHungarianMatcher, generalized_box_iou, center_to_corners_format
import torch
import torch
from torch import Tensor
from transformers import DetrImageProcessor
from transformers.models.detr.image_processing_detr import center_to_corners_format
import imagesize
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor, AutoTokenizer, Pix2StructConfig
from transformers import Pix2StructVisionModel, T5ForConditionalGeneration
from torch.nn.utils.rnn import pad_sequence

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


'''
self = Net(cfg)
'''

class Net(nn.Module):

    def __init__(self, cfg: Any):
        super(Net, self).__init__()

        self.cfg = cfg
        
        config = Pix2StructConfig()
        config.vision_config.use_bfloat16 = cfg.use_bfloat16
        config.text_config.use_bfloat16 = cfg.use_bfloat16
        self.backbone = Pix2StructForConditionalGeneration.from_pretrained(cfg.backbone, config = config)
        
        self.processor = Pix2StructProcessor.from_pretrained(cfg.backbone)
        sep_pos = self.processor.tokenizer.encode('\;', add_special_tokens = False)
        # wt1 = self.backbone.decoder.embed_tokens.weight[sep_pos]
        wt0 = self.backbone.decoder.embed_tokens.weight#[:-2]
        emb_wt = torch.cat((wt0, wt0[sep_pos]))
        self.backbone.resize_token_embeddings(2+self.backbone.config.text_config.vocab_size)
        self.loss_weights = torch.ones(self.backbone.config.text_config.vocab_size)
        self.loss_weights[-2:] *= self.cfg.break_wt
        self.backbone.decoder.embed_tokens.load_state_dict({'weight': emb_wt})
       
        self.backbone1 = copy.deepcopy(self.backbone)
        #self.backbone2 = copy.deepcopy(self.backbone)
        #self.backbone3 = copy.deepcopy(self.backbone)
        del self.backbone1.encoder
        #del self.backbone2.encoder
        #del self.backbone3.encoder

        # self.processor = Pix2StructProcessor.from_pretrained(cfg.backbone)
        self.in_features = self.backbone.config.vision_config.hidden_size
        self.chart_classifier = torch.nn.Linear(in_features=self.in_features, out_features=len(self.cfg.chart_map ), bias=True)
        # self.dtype_classifier = torch.nn.Linear(in_features=self.in_features, out_features=len(self.cfg.dtype_map ), bias=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        calc_grad = False
        for nm, param in self.backbone.named_parameters():
            if cfg.freeze_from in nm:
                calc_grad = True
            param.requires_grad = calc_grad
        # for nm, param in self.backbone.named_parameters():
        #     print(param.requires_grad, nm)
        self.loss_weights = torch.nn.Parameter(self.loss_weights, requires_grad=False)
        # self.cfg.break_wt = 2.
        
        self.criterion_table = nn.CrossEntropyLoss(weight = self.loss_weights , ignore_index=-100, reduction="mean")

    def forward(self, batch):
        
        labels1 = batch['series_label_1'].permute(1,0)
        labels2 = batch['series_label_2'].permute(1,0)
        #labels3 = batch['series_label_3'].permute(1,0)
        #labels4 = batch['series_label_4'].permute(1,0)
        encoding = {k:v[:,0] for k,v in batch.items() if k in ['flattened_patches', 'attention_mask']}
        
        enc_output = self.backbone.encoder(**encoding, return_dict = False)
        output  = self.backbone (**encoding, labels = labels1, encoder_outputs = enc_output, return_dict = False)
        output2 = self.backbone1(**encoding, labels = labels2, encoder_outputs = enc_output, return_dict = False)
        #output3 = self.backbone2(**encoding, labels = labels3, encoder_outputs = enc_output, return_dict = False)
        #output4 = self.backbone3(**encoding, labels = labels4, encoder_outputs = enc_output, return_dict = False)

        loss_table1 = self.criterion_table(output [1].contiguous().view(-1, output [1].size(-1)), labels1.contiguous().view(-1))
        loss_table2 = self.criterion_table(output2[1].contiguous().view(-1, output2[1].size(-1)), labels2.contiguous().view(-1))
        #loss_table3 = self.criterion_table(output3[1].contiguous().view(-1, output3[1].size(-1)), labels3.contiguous().view(-1))
        #loss_table4 = self.criterion_table(output4[1].contiguous().view(-1, output4[1].size(-1)), labels4.contiguous().view(-1))
        loss_table = (loss_table1 + loss_table2)/2 #+ loss_table3 + loss_table4) / 4
        
        # Predict chart type from encoder
        logits_chart = self.chart_classifier(enc_output[0][:,0])
        loss_chart = self.criterion(logits_chart, batch['chart_label'])
        
        loss = loss_table + 0.2 * loss_chart # + 0.05 * loss_dtype_x + 0.05 * loss_dtype_y
        
        outputs = {}
        outputs["loss"] = loss
        outputs["loss_table"] = loss_table
        outputs["loss_table1"] = loss_table1
        outputs["loss_table2"] = loss_table2
        outputs["loss_chart"] = loss_chart
        
        if not self.training:
            for tt, o in enumerate([output, output2]):
                logits_table = o[1].argmax(-1)
                padding = (0, self.cfg.max_label_length - o[1].size(1))
                logits_table  = F.pad(logits_table, padding, mode="constant", value=0)
                outputs[f"logits_table_{tt+1}"] = logits_table
            outputs[f"logits_table"] = outputs[f"logits_table_1"]
            outputs["logits_chart"] = logits_chart
            # outputs["logits_dtype"] = logits_dtype
 
        return outputs
