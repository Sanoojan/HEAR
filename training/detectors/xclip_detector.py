"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the XCLIPDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{ma2022x,
  title={X-clip: End-to-end multi-grained contrastive learning for video-text retrieval},
  author={Ma, Yiwei and Xu, Guohai and Sun, Xiaoshuai and Yan, Ming and Zhang, Ji and Ji, Rongrong},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={638--647},
  year={2022}
}
"""

import logging

import torch
import torch.nn as nn
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector

logger = logging.getLogger(__name__)
from networks.classifiers import *
from training.networks.loss_helpers import *
import wandb
import einops


@DETECTOR.register_module(module_name='xclip')
class XCLIPDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        D= config.get('dim', 768)
        topk_percent=config.get('topk_percent',0.1)
        spatio_temporal_patches= config.get('spatio_temporal_patches', False)
        self.backbone = self.build_backbone(config)
        # self.fc_norm = nn.LayerNorm(1024)
        # self.temporal_module = self.build_temporal_module(config)
        if config['classifier'] == 'linear':
            self.head = nn.Linear(D, 2)
        elif config['classifier'] == 'PatchClS_3DConv_TopKFrame':
            self.head = PatchClS_3DConv_TopKFrame(D=D, topk_percent=topk_percent, spatio_temporal_patches=spatio_temporal_patches)
        elif config['classifier']== 'PatchCls_2DConv_TopK':
            self.head = PatchCls_2DConv_TopK(D=D, topk_percent=topk_percent, spatio_temporal_patches=spatio_temporal_patches)
        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        # from transformers import XCLIPVisionModel
        # backbone = XCLIPVisionModel.from_pretrained("microsoft/xclip-large-patch14")
        
        from transformers import XCLIPModel
        backbone = XCLIPModel.from_pretrained("microsoft/xclip-large-patch14")

        return backbone

    def build_temporal_module(self, config):
        return nn.LSTM(input_size=2048, hidden_size=512, num_layers=3, batch_first=True)

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()

        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        # b, t, c, h, w = data_dict['image'].shape
        # frame_input = data_dict['image'].reshape(-1, c, h, w)
        # # get frame-level features
        # frame_level_features = self.backbone.features(frame_input)
        # frame_level_features = F.adaptive_avg_pool2d(frame_level_features, (1, 1)).reshape(b, t, -1)
        # # get video-level features
        # video_level_features = self.temporal_module(frame_level_features)[0][:, -1, :]
        videos=data_dict['image']
        t=self.config['clip_size']
        b, c, h, w = videos.shape
        batch_size=b//t
        num_frames=t
        # if videos.dim()==4:
        #     b, c, h, w = videos.shape
        #     t=self.config['clip_size']
        #     videos=videos.reshape(-1,t, c, h, w)
        # outputs=self.backbone.get_video_features(videos.reshape(batch_size,t, c, h, w), output_hidden_states=True)
        outputs= self.backbone.get_video_features_all_patch_all_layers(videos.reshape(batch_size,num_frames, c, h, w), output_hidden_states=True,return_dict=True) # L,B,N,D
        L,B,N,T,D= outputs.shape
        
        outputs=outputs.view(L*B,T, N, D)
        outputs=outputs[:,:,1:]
        
        # breakpoint()
        # video_level_features=outputs.mean(1)
        
        # outputs = self.backbone(videos, output_hidden_states=True)
        # sequence_output = outputs['pooler_output'].reshape(batch_size, num_frames, -1)
        # video_level_features = self.fc_norm(sequence_output.mean(1))
        # breakpoint()
        # video_level_features = outputs.mean(1)
        

        return outputs

    def classifier(self, features: torch.tensor,return_patch_scores=False) -> torch.tensor:
        return self.head(features, return_patch_scores=return_patch_scores) if return_patch_scores else self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict,epoch=None) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        other_loss=0
        total_epochs= self.config.get('nEpochs', 15)
        
        cfg = self.config['Patch_MIL_ranking_all_loss']
        if cfg.get('use_variable_margin', False):
            max_margin= cfg.get('max_margin', 0.9)
            min_margin= cfg.get('min_margin', 0.1)
            Variable_margin= min(max_margin, min_margin + epoch / total_epochs)
        else:
            Variable_margin= cfg.get('margin', 0.3)
        
        
        cfg = self.config['Patch_MIL_ranking_all_loss']
        Other_Patch_loss=Patch_MIL_ranking_all_loss(pred_dict['patch_scores'], label, cfg=cfg,margin=Variable_margin)
        
        wandb.log({'Other_Patch_loss': Other_Patch_loss.item()})
        other_loss += Other_Patch_loss
        loss_dict = {'overall': other_loss}
        
        # loss = self.loss_func(pred, label)
        # loss_dict = {'overall': loss}
        wandb.log({'Patch_MIL_ranking_all_loss': other_loss}) 

        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

        return metric_batch_dict

    def forward(self, data_dict: dict, inference=False) -> dict:
        
        with torch.no_grad():
            features = self.features(data_dict)
        return_patch_scores=True if self.config['classifier'] in ['PatchClS_3DConv_TopKFrame','PatchCls_2DConv_TopK'] else False
        # pred = self.classifier(features)
        pred, patch_scores = self.classifier(features, return_patch_scores=return_patch_scores)
        features = features.mean(dim=(0, 2, 3))  # L,B,N,D -> B,D
        # breakpoint()
        clip_size=self.config.get('clip_size',2)
        batch_size= data_dict['image'].shape[0]//clip_size
        LB,P= pred.shape
        pred=pred.view(LB//batch_size, batch_size, -1)
        patch_scores=patch_scores.view(LB//batch_size, batch_size, -1)
        pred=pred.mean(0)
        patch_scores=patch_scores.mean(0)
        # breakpoint()
        
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob,'patch_scores': patch_scores, 'feat': features}


        return pred_dict
