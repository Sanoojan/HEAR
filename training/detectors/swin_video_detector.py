
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='videoswin')
class VideoSwinDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)

        self.feat_dim = 1024  # swin3d_b output dim
        self.fc_norm = nn.LayerNorm(self.feat_dim)
        self.head = nn.Linear(self.feat_dim, 2)

        self.loss_func = self.build_loss(config)

    def build_backbone(self, config):
        # from torchvision.models.video import swin3d_b, Swin3D_B_Weights

        # weights = Swin3D_B_Weights.KINETICS400_V1
        # backbone = swin3d_b(weights=weights)

        # # remove classification head
        # backbone.head = nn.Identity()
        
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        backbone = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/video-mask2former-swin-large-youtubevis-2019-instance"
        )

        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def features(self, data_dict: dict) -> torch.Tensor:
        videos = data_dict['image']  # (B, T, C, H, W)

        if videos.dim() == 4:
            b, c, h, w = videos.shape
            t = self.config['clip_size']
            videos = videos.view(-1, t, c, h, w)

        # Swin expects (B, C, T, H, W)
        videos = videos.permute(0, 2, 1, 3, 4)

        feats = self.backbone(videos)  # (B, 1024)

        feats = self.fc_norm(feats)
        return feats

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict, epoch=None) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)

        wandb.log({'loss': loss.item(), 'epoch': epoch})
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        from metrics.base_metrics_class import calculate_metrics_for_train

        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())

        self.video_names = []
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]

        return {
            'cls': pred,
            'prob': prob,
            'feat': features
        }