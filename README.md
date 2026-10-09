# HEAR

Implementation of **HEAR: Hierarchical Evidence Accumulation with Coherence Refinement for Deepfake Face Video Detection**.

HEAR extracts patch features from a pretrained vision transformer, estimates local fake evidence, and refines it with a lightweight Coherence Refinement Module (CRM). This repository supports CLIP-L/14 and DINOv3-L/16 backbones.

## Datasets

We train on **FaceForensics++ (FF++), c23 compression**, using its real videos and four manipulation types: Deepfakes, Face2Face, FaceSwap, and NeuralTextures. Cross-dataset evaluation uses:

- Celeb-DF-v2
- Deepfake Detection Challenge (DFDC)
- Deepfake Detection Challenge Preview (DFDCP)
- DeepFakeDetection (DFD)
- WildDeepFake (WDF)
- DF40, focusing on video face-swapping and face-reenactment methods

For dataset downloads, access requirements, face preprocessing, and split preparation, follow [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) and the [official DF40 repository](https://github.com/YZY-stack/DF40). Dataset files are not distributed with this repository.

Use the DeepfakeBench-style frame directories and JSON manifests. The relevant local layout is:

```text
HEAR/
├── DeepfakeDatasets/
│   ├── rgb/                         # FF++, Celeb-DF-v2, DFDC, etc.
│   └── DF40/Unzipped/               # DF40 frame data
├── preprocessing/
│   ├── dataset_json/                # Standard benchmark manifests
│   └── DF40_json/dataset_json/      # DF40 manifests
└── training/config/
    ├── train_config.yaml
    └── test_config.yaml
```

`DeepfakeDatasets` can be a local directory or a symlink to your dataset storage. Manifest frame paths must match the selected RGB root. HEAR consumes RGB face frames and video labels

Set the data paths in `training/config/train_config.yaml` for training and in `training/config/test_config.yaml` for evaluation. Run all commands below from the repository root.

## Environment setup

Use Linux with an NVIDIA GPU and a CUDA-compatible driver. Create the Conda environment named **HEAR**, install PyTorch first, and then install the remaining dependencies:

```bash
conda env create -f environment.yaml
conda activate HEAR

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-cache-dir --no-binary=dlib \
  --config-settings=--build-option=--no \
  --config-settings=--build-option=DLIB_USE_CUDA dlib==19.24.6
python -m pip install -r requirements.txt
python -m pip check
```

The environment file provides Python 3.10 and build tools for dependencies such as dlib. The PyTorch command installs CUDA 12.1 wheels; a separate CUDA toolkit installation is not needed for those wheels. See the [official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/#v251) if a different CUDA build is needed. Keep the matching PyTorch/torchvision versions.

Install dlib with CUDA disabled as shown above; it is needed by inherited imports, and its CPU build avoids conflicts with the PyTorch CUDA libraries. 

The dependency file pins the timm revision used by the local DINOv3 implementation and the Albumentations API used by the dataset loader. Keep these pins when reproducing this setup.

Check the entry points and GPU access:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python training/train.py --help
python training/test.py --help
```

Pretrained backbone weights are downloaded on first use. If a selected Hugging Face backbone requires access approval, obtain access on its model page and run `hf auth login` locally. For experiment tracking, run `wandb login`, or use `export WANDB_MODE=disabled` to disable remote logging.

## Training

The main CLIP configuration is `training/config/detector/clip.yaml`. Train on a single GPU with:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29300 \
  training/train.py \
  --detector_path training/config/detector/clip.yaml \
  --test_name HEAR_CLIP \
  --ddp
```

For DINOv3-L/16, use `--detector_path training/config/detector/dino.yaml` instead. Set dataset lists, batch sizes, and other training options in the detector config; shared dataset roots are in `training/config/train_config.yaml`.

The provided `train.sh` is a convenience wrapper.

The code inherits the training, testing, and checkpoint-selection workflow from DeepfakeBench, including selection by the configured evaluation metric.

## Pretrained checkpoint

The CLIP-L/14 checkpoint is available publicly on **[Hugging Face: Sanoojan/HEAR](https://huggingface.co/Sanoojan/HEAR)**.

| Model | Training data | Download |
| --- | --- | --- |
| HEAR, CLIP-L/14 | FaceForensics++ c23 | [hear_clip_l14.pth](https://huggingface.co/Sanoojan/HEAR/resolve/main/hear_clip_l14.pth) |

For local evaluation, place the CLIP checkpoint at `checkpoints/hear_clip_l14.pth` and use that path in `--weights_path`. 

Download the checkpoint and its checksum with:

```bash
hf download Sanoojan/HEAR hear_clip_l14.pth hear_clip_l14.pth.sha256 \
  --local-dir checkpoints
(cd checkpoints && sha256sum -c hear_clip_l14.pth.sha256)
```



## Testing

### Standard benchmarks

First set these values in `training/config/test_config.yaml`:

```yaml
rgb_dir: DeepfakeDatasets/rgb
dataset_json_folder: ./preprocessing/dataset_json
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python training/test.py \
  --detector_path training/config/detector/clip_inference.yaml \
  --test_dataset FaceForensics++ Celeb-DF-v2 DFDCP DFDC WDF DeepFakeDetection \
  --weights_path checkpoints/hear_clip_l14.pth \
  --test_name HEAR_benchmarks
```

### DF40

Switch `training/config/test_config.yaml` to the DF40 roots (the checked-in defaults):

```yaml
rgb_dir: DeepfakeDatasets/DF40/Unzipped
dataset_json_folder: ./preprocessing/DF40_json/dataset_json
```

For example, evaluate face reenactment on the FF++ and Celeb-DF backgrounds:

```bash
CUDA_VISIBLE_DEVICES=0 python training/test.py \
  --detector_path training/config/detector/clip_inference.yaml \
  --test_dataset fomm_ff fomm_cdf \
  --weights_path checkpoints/hear_clip_l14.pth \
  --test_name HEAR_DF40
```

Use the JSON filename stems in `preprocessing/DF40_json/dataset_json/` to select other DF40 subsets. `test.sh` contains a larger face-reenactment evaluation list and enables visualization. To use your downloaded checkpoint with that script:

```bash
WEIGHTS_PATH=checkpoints/hear_clip_l14.pth bash test.sh
```

For DINO evaluation, use `training/config/detector/dino.yaml` and a checkpoint trained with that configuration. Adjust `test_batchSize` and `workers` in the chosen detector config to fit available resources.

## Visualization

Add **`--visualize_patchwise`** to the same test command:

```bash
CUDA_VISIBLE_DEVICES=0 python training/test.py \
  --detector_path training/config/detector/clip_inference.yaml \
  --test_dataset DFDC \
  --weights_path checkpoints/hear_clip_l14.pth \
  --test_name HEAR_visualization \
  --visualize_patchwise
```

This example uses the DF40 roots above. The output contains the original face frames beside the patch-evidence heatmap, saved as named MP4 files under:

```text
visualizations/patch_scores/HEAR_visualization/DFDC/
```

Optionally add `--save_patch_scores` to save patch scores, video names, labels, and predictions under `visualizations/saved_patch_scores/<test_name>/<dataset>.pkl`. This flag works with or without visualization.

## Acknowledgments

This repository is built on [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). **Major portions of the code are obtained from and adapted from DeepfakeBench**, including dataset loading, preprocessing utilities, the training and testing framework, evaluation metrics, logging, and checkpoint management. We use its **training, testing, and checkpoint-selection protocol** as the basis of our experimental workflow. HEAR's evidence estimation, coherence refinement, and ranking supervision are integrated into that framework.

We thank the DeepfakeBench authors and the [DF40](https://github.com/YZY-stack/DF40) authors for releasing their code, datasets, and evaluation resources. We also acknowledge the CLIP, DINOv3, Hugging Face Transformers, and timm projects for their pretrained models and implementations. Please cite DeepfakeBench and DF40, in addition to HEAR, when using the corresponding code or datasets; their repositories provide citation information. Retain the upstream attribution and license notices for reused code.
