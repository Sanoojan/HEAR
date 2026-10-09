"""HEAR local evidence estimation and coherence refinement (paper Section 3.2)."""

import math

import torch.nn as nn


class HEARClassifier(nn.Module):
    """One layer's local evidence estimator, CRM, and top-K aggregation."""

    # Retain appearance weights from checkpoints predating the paper cleanup.
    _legacy_module_names = {
        'patch_fc': 'local_evidence_estimator',
        'agg3d': 'coherence_refinement',
    }
    _removed_module_names = (
        'patch_diff_fc', 'patch_second_diff_fc', 'agg3d_diff', 'agg3d_second_diff',
    )

    def __init__(self, feature_dim, hidden_dim=128, topk_ratio=0.1):
        super().__init__()
        if not 0 < topk_ratio <= 1:
            raise ValueError('topk_ratio must be in (0, 1].')
        self.topk_ratio = topk_ratio

        def evidence_estimator():
            return nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        self.local_evidence_estimator = evidence_estimator()

        def coherence_refinement():
            return nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv3d(8, 2, kernel_size=1),
            )

        self.coherence_refinement = coherence_refinement()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Load retained appearance weights and discard removed branch weights."""
        for name in self._removed_module_names:
            for key in list(state_dict):
                if key.startswith(prefix + name + '.'):
                    state_dict.pop(key)
        for old_name, new_name in self._legacy_module_names.items():
            old_prefix = prefix + old_name + '.'
            for key in list(state_dict):
                if key.startswith(old_prefix):
                    new_key = prefix + new_name + '.' + key[len(old_prefix):]
                    if new_key in state_dict:
                        raise RuntimeError(f'Checkpoint contains both {key} and {new_key}.')
                    state_dict[new_key] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @staticmethod
    def _refine_evidence(features, estimator, refinement):
        """Map [B, T, N, D] features to binary logits and fake evidence."""
        batch_size, frames, patches, feature_dim = features.shape
        side = math.isqrt(patches)
        if side * side != patches:
            raise ValueError('Patch tokens must form a square spatial grid.')
        local_evidence = estimator(features.reshape(-1, feature_dim))
        evidence_volume = local_evidence.view(batch_size, 1, frames, side, side)
        refined_logits = refinement(evidence_volume)
        patch_evidence = refined_logits.softmax(dim=1)[:, 1].view(batch_size, frames, patches)
        return refined_logits, patch_evidence

    def _aggregate_video_logits(self, refined_logits):
        """Average logits from the top-K fake-evidence patches in every frame.

        Preserve the existing two-class logit aggregation. The detector averages
        these logits across layers before softmax; the paper instead describes
        averaging patch probabilities in Equations (4)-(5).
        """
        batch_size, classes, frames, height, width = refined_logits.shape
        logits = refined_logits.view(batch_size, classes, frames, height * width)
        fake_evidence = logits.softmax(dim=1)[:, 1:2]
        k = max(1, int(self.topk_ratio * (height * width)))
        indices = fake_evidence.topk(k, dim=-1).indices.expand(-1, classes, -1, -1)
        return logits.gather(dim=-1, index=indices).mean(dim=(2, 3))

    def forward(self, patch_features, return_patch_scores=True):
        """Return video logits and optional refined patch evidence [B, T, N]."""
        refined_logits, patch_evidence = self._refine_evidence(
            patch_features, self.local_evidence_estimator, self.coherence_refinement,
        )
        video_logits = self._aggregate_video_logits(refined_logits)
        if return_patch_scores:
            return video_logits, patch_evidence
        return video_logits
