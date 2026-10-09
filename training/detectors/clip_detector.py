"""HEAR with frozen CLIP features.

Adapted from DeepfakeBench (Zhiyuan Yan, zhiyuanyan@link.cuhk.edu.cn).
CLIP: Radford et al., Learning Transferable Visual Models From Natural
Language Supervision, ICML 2021.
"""

from transformers import AutoProcessor

from detectors import DETECTOR
from networks.models.clip.clip import CLIPModel
from .hear_detector import HEARDetector


@DETECTOR.register_module(module_name='clip')
class CLIPDetector(HEARDetector):
    def build_backbone(self, config):
        _, backbone = get_clip_visual(
            config.get('clip_backbone', 'openai/clip-vit-large-patch14')
        )
        return backbone

    def backbone_dimensions(self):
        return self.backbone.config.hidden_size, self.backbone.config.num_hidden_layers, 1

    def normalize_tokens(self, tokens):
        return self.backbone.post_layernorm(tokens)

    def features(self, data_dict):
        return self.backbone(data_dict['image'], output_hidden_states=True)


def get_clip_visual(model_name='openai/clip-vit-base-patch16'):
    processor = AutoProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return processor, model.vision_model
