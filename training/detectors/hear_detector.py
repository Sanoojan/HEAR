"""Shared HEAR heads, evidence aggregation, loss, and metrics for ViT backbones."""

import torch
import torch.nn as nn
import wandb

from metrics.base_metrics_class import calculate_metrics_for_train
from loss import LOSSFUNC
from networks.classifiers import HEARClassifier
from networks.loss_helpers import frame_level_mil_ranking_loss
from .base_detector import AbstractDetector


class HEARDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config['classifier'] not in {'HEARClassifier', 'PatchClS_TripleStream_3DConv_TopKFrame'}:
            raise ValueError('Only HEARClassifier is supported.')
        supported_losses = {'All_layer_CE', 'frame_level_mil_ranking'}
        # Accept the former loss name when reading saved experiment configs.
        loss_names = {
            'frame_level_mil_ranking' if name == 'Patch_frame_MIL_ranking_all' else name
            for name in config.get('additional_losses', [])
        }
        if loss_names != supported_losses:
            raise ValueError(f'HEAR requires additional_losses: {sorted(supported_losses)}')
        if config.get('Image_diff', False) or config.get('mid_frame_prediction', False):
            raise ValueError('Image-difference and mid-frame prediction ablations were removed.')
        if config.get('All_layer_CE', {}).get('use_ranking_loss', False):
            raise ValueError('The All_layer_CE ranking ablation was removed.')
        if not config.get('return_patch_scores', True):
            raise ValueError('HEAR requires refined patch evidence for MIL supervision.')

        if not config.get('spatio_temporal_patches', True):
            raise ValueError('HEAR requires coherence-refined patch evidence.')

        self.layer_sep_classifiers = config.get('layer_sep_classifiers', False)
        self.backbone = self.build_backbone(config)
        feature_dim, num_layers, self.num_prefix_tokens = self.backbone_dimensions()
        head_kwargs = {
            'feature_dim': feature_dim,
            'topk_ratio': config.get('topk_percent', 0.1),
        }
        if self.layer_sep_classifiers:
            self.head = nn.ModuleList([
                HEARClassifier(**head_kwargs)
                for _ in range(num_layers)
            ])
        else:
            self.head = HEARClassifier(**head_kwargs)
        self.loss_func = self.build_loss(config)

    def backbone_dimensions(self):
        """Return feature dimension, encoder depth, and prefix-token count."""
        raise NotImplementedError

    def normalize_tokens(self, tokens):
        """Apply the backbone's output normalization to intermediate tokens."""
        raise NotImplementedError

    def build_loss(self, config):
        return LOSSFUNC[config['loss_func']]()

    def classifier(self, features, return_patch_scores=False):
        """Classify patch tokens from each encoder layer, excluding CLS and register tokens."""
        hidden_states = features.hidden_states
        T = self.config.get('clip_size', 1)
        B = hidden_states[1].shape[0] // T
        if self.config.get('apply_layernorm', False):
            hidden_states = [hidden_states[0]] + [
                self.normalize_tokens(h) for h in hidden_states[1:]
            ]
        outcomes = []
        for i in range(1, len(hidden_states)):
            head = self.head[i - 1] if self.layer_sep_classifiers else self.head
            patches = hidden_states[i][:, self.num_prefix_tokens:, :].view(B, T, -1, hidden_states[i].shape[-1])
            outcomes.append(head(
                patches,
                return_patch_scores=return_patch_scores,
            ))
        return outcomes

    def get_losses(self, data_dict: dict, pred_dict: dict, epoch=None) -> dict:
        """Combine layer CE with frame-level MIL ranking of refined evidence."""
        assert epoch is not None
        label = data_dict['label']
        pred_list = pred_dict['pred_list']
        ce_cfg = self.config['All_layer_CE']
        if ce_cfg.get('use_mean_pred', False):
            mean_pred = torch.mean(torch.stack(pred_list), dim=0)
            ce_loss = self.loss_func(mean_pred, label)
            wandb.log({'All_layer_CE_mean_prob_loss': ce_loss.item()})
        else:
            ce_loss = sum(self.loss_func(pred, label) for pred in pred_list) / len(pred_list)
            wandb.log({'All_layer_CE_loss': ce_loss.item()})
        overall = ce_loss * ce_cfg['CE_alpha']
        loss_dict = {'overall': overall}

        cfg = self.config.get('frame_level_mil_ranking', self.config.get('Patch_frame_MIL_ranking_all', {}))
        ranking_loss, ranking_metrics = frame_level_mil_ranking_loss(
            pred_dict['patch_scores'], label,
            margin=cfg.get('margin', 0.6),
            topk_ratio=cfg.get('topk_percent', 0.1),
            weight=cfg.get('alpha', 1.0),
        )
        wandb.log({'mil_ranking_loss': ranking_loss.item()})
        loss_dict['overall'] = overall + ranking_loss
        for key, value in ranking_metrics.items():
            wandb.log({key: value})
            loss_dict[key] = value
        return loss_dict

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict['label'].detach(), pred_dict['cls'].detach(),
        )
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference=False, return_patch_scores=False) -> dict:
        # Keep the backbone gradient behavior controlled by the optimizer config.
        if self.config['optimizer'].get('classifier_only', False):
            with torch.no_grad():
                features = self.features(data_dict)
        else:
            features = self.features(data_dict)
        outcomes = self.classifier(features, return_patch_scores=True)
        pred_list = [outcome[0] for outcome in outcomes]
        pred = torch.mean(torch.stack(pred_list), dim=0)
        return {
            'cls': pred,
            'prob': torch.softmax(pred, dim=1)[:, 1],
            'feat': features.pooler_output,
            'patch_scores': torch.mean(torch.stack([outcome[1] for outcome in outcomes]), dim=0),
            'pred_list': pred_list,
            'prob_list': [torch.softmax(p, dim=1)[:, 1] for p in pred_list],
        }

