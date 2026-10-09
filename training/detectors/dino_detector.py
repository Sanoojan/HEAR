"""HEAR with DINOv3 patch features and backbone-specific normalization."""

from types import SimpleNamespace

from detectors import DETECTOR
from networks.models.dino.eva import (
    vit_base_patch16_dinov3,
    vit_large_patch16_dinov3,
)
from .hear_detector import HEARDetector


@DETECTOR.register_module(module_name='dino')
class DINODetector(HEARDetector):
    def build_backbone(self, config):
        backbones = {
            'vit_base_patch16_dinov3': vit_base_patch16_dinov3,
            'vit_large_patch16_dinov3': vit_large_patch16_dinov3,
        }
        model_name = config['dino_model']
        if model_name not in backbones:
            raise ValueError(f'Unsupported DINO backbone: {model_name}')
        return backbones[model_name](pretrained=True, num_classes=0, global_pool='')

    def backbone_dimensions(self):
        return self.backbone.num_features, len(self.backbone.blocks), self.backbone.num_prefix_tokens

    def normalize_tokens(self, tokens):
        return self.backbone.norm(tokens)

    def features(self, data_dict):
        last_tokens, hidden_states = self.backbone.forward_features(
            data_dict['image'], return_all_layers=True,
        )
        # Match CLIP's hidden-state convention: position zero precedes the blocks.
        # The shared classifier uses entries 1..L and skips every prefix token.
        return SimpleNamespace(
            hidden_states=(None, *hidden_states),
            pooler_output=last_tokens[:, 0, :],
        )
