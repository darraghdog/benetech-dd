from typing import Any, Dict

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
        
        self.processor = Pix2StructProcessor.from_pretrained(cfg.backbone, is_vqa = False)
        sep_pos = self.processor.tokenizer.encode('\;', add_special_tokens = False)
        # wt1 = self.backbone.decoder.embed_tokens.weight[sep_pos]
        wt0 = self.backbone.decoder.embed_tokens.weight#[:-2]
        emb_wt = torch.cat((wt0, wt0[sep_pos]))
        self.backbone.resize_token_embeddings(2+self.backbone.config.text_config.vocab_size)
        self.loss_weights = torch.ones(self.backbone.config.text_config.vocab_size)
        self.loss_weights[-2:] *= self.cfg.break_wt
        self.backbone.decoder.embed_tokens.load_state_dict({'weight': emb_wt})
        
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
        
        labels = batch['series_label_1'].permute(1,0)
        encoding = {k:v[:,0] for k,v in batch.items() if k in ['flattened_patches', 'attention_mask']}
        predictions = self.backbone.generate(**encoding, 
                                                max_new_tokens=self.cfg.max_label_length, 
                                                early_stopping=True,
                                                pad_token_id=self.processor.tokenizer.pad_token_id,
                                                eos_token_id=self.processor.tokenizer.eos_token_id,
                                                use_cache=True,
                                                num_beams=self.cfg.num_beams,
                                                temperature=self.cfg.temperature,
                                                top_k=self.cfg.top_k,
                                                return_dict_in_generate=True,
                                                output_hidden_states = True)
        logits_chart = self.chart_classifier(predictions.encoder_hidden_states[-1][:,0])
        logits_table = predictions.sequences #[:,1:]
        # print(logits_table)
        loss = self.criterion(logits_chart, batch['chart_label'])

        outputs = {}
        outputs["loss"] = loss
        if not self.training:
            # Calculate the required padding
            padding = (0, self.cfg.max_label_length - logits_table.size(1))
            # Pad the array along the 3rd dimension
            logits_table  = F.pad(logits_table, padding, mode="constant", value=0)
            outputs["logits_table"] = logits_table
            outputs["logits_chart"] = logits_chart
 
        return outputs
