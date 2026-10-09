"""Frame-level multiple-instance ranking supervision for HEAR (Section 3.3)."""

import torch
import torch.nn.functional as F


def frame_level_mil_ranking_loss(patch_evidence, labels, *, margin=0.6, topk_ratio=0.1, weight=1.0):
    """Rank fake frames above real frames using their top-K patch evidence.

    Args:
        patch_evidence: Refined fake probabilities of shape [B, T, N], already
            averaged across transformer layers by the detector.
        labels: Binary video labels [B] (0: real, 1: fake).
        margin: Required separation between fake and real frame scores.
        topk_ratio: Fraction of patches selected within each frame.
        weight: Multiplier applied to the ranking loss.

    Uses all real/fake frame pairs and the mean squared hinge penalty from the
    existing implementation. Variances are diagnostics, not regularizers.
    A batch containing only one class contributes a differentiable zero loss.
    """
    if not 0 < topk_ratio <= 1:
        raise ValueError('topk_ratio must be in (0, 1].')
    k = max(1, int(patch_evidence.shape[-1] * topk_ratio))
    frame_scores = patch_evidence.topk(k, dim=2).values.mean(dim=2)
    real_scores = frame_scores[labels == 0]
    fake_scores = frame_scores[labels == 1]
    zero = patch_evidence.sum() * 0.0
    if real_scores.numel() == 0 or fake_scores.numel() == 0:
        return zero, {
            'ranking_loss': zero,
            'real_frame_score_variance': zero.detach(),
            'fake_frame_score_variance': zero.detach(),
        }

    real_frames = real_scores.reshape(-1, 1)
    fake_frames = fake_scores.reshape(1, -1)
    ranking_loss = F.relu(margin - (fake_frames - real_frames)).pow(2).mean()
    return weight * ranking_loss, {
        'ranking_loss': ranking_loss,
        'real_frame_score_variance': real_scores.var(unbiased=False),
        'fake_frame_score_variance': fake_scores.var(unbiased=False),
    }
