from sklearn import metrics
import numpy as np
import torch
import torchvision
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm
import cv2
import os
import json

def build_fake_frame_dict(metadata_path, num_sampled_frames=None):

    """
    If num_sampled_frames is None:
        → returns ORIGINAL frame indices (recommended for your case)

    If num_sampled_frames is provided:
        → returns sampled-frame indices (old behavior)
    """

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    fake_dict = {}

    for item in metadata:

        file_path = item["file"]
        modify_type = item["modify_type"]

        # Convert path → your naming format
        vid_name = file_path.replace("/", "_").replace(".mp4", "")

        total_frames = item["video_frames"]

        # -----------------------------
        # Real videos → no fake frames
        # -----------------------------
        if modify_type == "real":
            fake_dict[vid_name] = set()
            continue

        segments = item["visual_fake_segments"]

        fake_frames = set()

        # -----------------------------
        # Build ORIGINAL fake frames
        # -----------------------------
        for seg in segments:

            start_t, end_t = seg

            fps = 25  # metadata fps

            start_f = int(start_t * fps)
            end_f = int(end_t * fps)

            start_f = max(start_f, 0)
            end_f = min(end_f, total_frames - 1)

            fake_frames.update(range(start_f, end_f + 1))

        # -------------------------------------------------
        # If no sampling → return ORIGINAL frame indices
        # -------------------------------------------------
        if num_sampled_frames is None:
            fake_dict[vid_name] = fake_frames
            continue

        # -------------------------------------------------
        # If sampling requested → map to sampled indices
        # -------------------------------------------------
        sampled = np.linspace(
            0,
            total_frames - 1,
            num_sampled_frames
        ).astype(int)

        sampled_fake = set()

        for i, f in enumerate(sampled):
            if f in fake_frames:
                sampled_fake.add(i)

        fake_dict[vid_name] = sampled_fake

    return fake_dict

def parse_metric_for_print(metric_dict):
    if metric_dict is None:
        return "\n"
    str = "\n"
    str += "================================ Each dataset best metric ================================ \n"
    for key, value in metric_dict.items():
        if key != 'avg':
            str= str+ f"| {key}: "
            for k,v in value.items():
                str = str + f" {k}={v} "
            str= str+ "| \n"
        else:
            str += "============================================================================================= \n"
            str += "================================== Average best metric ====================================== \n"
            avg_dict = value
            for avg_key, avg_value in avg_dict.items():
                if avg_key == 'dataset_dict':
                    for key,value in avg_value.items():
                        str = str + f"| {key}: {value} | \n"
                else:
                    str = str + f"| avg {avg_key}: {avg_value} | \n"
    str += "============================================================================================="
    return str


def get_test_metrics(y_pred, y_true, img_names):
    def get_video_metrics(image, pred, label):
        result_dict = {}
        new_label = []
        new_pred = []
        # print(image[0])
        # print(pred.shape)
        # print(label.shape)
        for item in np.transpose(np.stack((image, pred, label)), (1, 0)):

            s = item[0]
            if '\\' in s:
                parts = s.split('\\')
            else:
                parts = s.split('/')
            a = parts[-2]
            b = parts[-1]

            if a not in result_dict:
                result_dict[a] = []

            result_dict[a].append(item)
        image_arr = list(result_dict.values())

        for video in image_arr:
            pred_sum = 0
            label_sum = 0
            leng = 0
            for frame in video:
                pred_sum += float(frame[1])
                label_sum += int(frame[2])
                leng += 1
            new_pred.append(pred_sum / leng)
            new_label.append(int(label_sum / leng))
        fpr, tpr, thresholds = metrics.roc_curve(new_label, new_pred)
        v_auc = metrics.auc(fpr, tpr)
        fnr = 1 - tpr
        v_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return v_auc, v_eer


    y_pred = y_pred.squeeze()
    # For UCF, where labels for different manipulations are not consistent.
    y_true[y_true >= 1] = 1
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # eer
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    # ap
    ap = metrics.average_precision_score(y_true, y_pred)
    # acc
    prediction_class = (y_pred > 0.5).astype(int)
    correct = (prediction_class == np.clip(y_true, a_min=0, a_max=1)).sum().item()
    acc = correct / len(prediction_class)
    if type(img_names[0]) is not list:
        # calculate video-level auc for the frame-level methods.
        v_auc, _ = get_video_metrics(img_names, y_pred, y_true)
    else:
        # video-level methods
        v_auc=auc

    return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap, 'pred': y_pred, 'video_auc': v_auc, 'label': y_true}


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

def visualize_batch(images, save_path="grid.png"):
    """
    images: tensor (B, 3, H, W) normalized by CLIP mean/std
    """

    # Move mean & std to the same device
    mean = CLIP_MEAN.to(images.device)
    std = CLIP_STD.to(images.device)

    # --- De-normalize CLIP images ---
    images_denorm = images * std + mean

    # Clamp to valid range
    images_denorm = images_denorm.clamp(0, 1)

    # Create a grid (8x4 by default for 32)
    grid = make_grid(images_denorm, nrow=8, padding=2)

    # Convert to uint8 HWC
    ndarr = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    img = Image.fromarray(ndarr)

    img.save(save_path)
    print(f"[Saved] {save_path}")
    
    
def patch_score_visualize(
    X, y, scores, preds, save_dir='vis_scores', apply_softmax=False,
    fps=25, video_names=None, append_original=True, start_index=0,
    mean=(0.48145466, 0.4578275, 0.40821073),
    std=(0.26862954, 0.26130258, 0.27577711),
):
    """Write one named heatmap video per clip without dataset-specific metadata."""
    os.makedirs(save_dir, exist_ok=True)
    X = X.detach().cpu()
    scores = scores.detach().cpu()
    if X.ndim == 4:
        X = X.unsqueeze(1)
        if scores.ndim == 2:
            scores = scores.unsqueeze(1)
    if scores.shape[-1] == 257:
        scores = scores[:, :, 1:]
    B, T, C, H, W = X.shape
    side = int(scores.shape[-1] ** 0.5)
    if scores.shape[:2] != (B, T) or side * side != scores.shape[-1]:
        raise ValueError('Patch scores must match the clips and form a square spatial grid.')
    if video_names is None:
        video_names = [f'video_{start_index + i}' for i in range(B)]
    if len(video_names) != B or len(y) != B or len(preds) != B:
        raise ValueError('Expected one name, label, and prediction per clip.')

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    scores = scores.softmax(dim=-1) if apply_softmax else scores.relu()
    score_min = scores.min()
    score_range = (scores.max() - score_min).clamp_min(1e-6)
    for b, name in enumerate(video_names):
        # Dataset identifiers may contain slashes; keep each output in save_dir.
        safe_name = ''.join(c if c.isalnum() or c in '._-' else '_' for c in str(name))
        filename = f'gt{int(y[b])}_pred{float(preds[b]):.4f}_in{start_index + b}_name{safe_name}.mp4'
        out_path = os.path.join(save_dir, filename)
        width = W * 2 if append_original else W
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, H))
        if not writer.isOpened():
            raise RuntimeError(f'Could not open video writer: {out_path}')
        try:
            for t in range(T):
                img = (X[b, t] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                patch = scores[b, t]
                heat = ((patch - score_min) / score_range).clamp(0, 1).reshape(side, side).numpy()
                heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)
                heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
                heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
                overlay = np.ascontiguousarray(((0.45 * heat_color + 0.55 * img) * 255).astype(np.uint8))
                original = np.ascontiguousarray((img * 255).astype(np.uint8))
                k = max(1, int(patch.numel() * 0.1))
                frame_score = patch.topk(k).values.mean().item()
                cv2.putText(original, f'score: {frame_score:.3f}', (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                combined = np.concatenate([original, overlay], axis=1) if append_original else overlay
                writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
    print(f'Saved {B} visualizations to {save_dir}')
