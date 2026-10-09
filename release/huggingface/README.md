---
base_model: openai/clip-vit-large-patch14
library_name: pytorch
tags:
  - deepfake-detection
  - video-classification
  - clip
  - hear
---

# HEAR — CLIP-L/14 checkpoint

Checkpoint for **HEAR: Hierarchical Evidence Accumulation with Coherence Refinement for Deepfake Face Video Detection**.

This is the original research checkpoint, trained using the FaceForensics++ c23 setup. It is supplied as a PyTorch state dictionary (`hear_clip_l14.pth`), not as a Transformers `from_pretrained` model. Load it using the HEAR repository's `training/test.py` and the included `clip_inference.yaml` configuration. Labels are 0 for real and 1 for fake.

The cleaned HEAR loader maps legacy appearance-head parameter names and discards the removed auxiliary temporal-difference branch weights. Predictions use the appearance evidence path. This checkpoint is for CLIP-L/14, not DINOv3.

The training, testing, and checkpoint-selection framework and major portions of the implementation are adapted from [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). Evaluation datasets and access instructions are provided by DeepfakeBench and [DF40](https://github.com/YZY-stack/DF40).

A model score is a prediction, not proof that a video is authentic or manipulated. Performance can change across datasets, manipulation methods, and preprocessing pipelines. No new benchmark results are asserted by this model card.
