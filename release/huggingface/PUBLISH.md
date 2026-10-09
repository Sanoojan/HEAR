# Publish the HEAR checkpoint

The model card is prepared in `README.md` in this directory. The checkpoint currently available in the development workspace is:

```text
checkpoints/hear_clip_l14.pth
```

The published repository is [Sanoojan/HEAR](https://huggingface.co/Sanoojan/HEAR), with public visibility. Authenticate using `hf auth login`; do not put credentials in this repository.

To update the published files, run from the repository root:

```bash
export HEAR_MODEL_ID="Sanoojan/HEAR"
hf upload "$HEAR_MODEL_ID" \
  checkpoints/hear_clip_l14.pth \
  hear_clip_l14.pth
hf upload "$HEAR_MODEL_ID" release/huggingface/clip_inference.yaml clip_inference.yaml
hf upload "$HEAR_MODEL_ID" release/huggingface/README.md README.md
hf upload "$HEAR_MODEL_ID" release/huggingface/hear_clip_l14.pth.sha256 hear_clip_l14.pth.sha256
```

For a new checkpoint release, regenerate the SHA-256 file and update the model card before uploading. The current checkpoint is publicly downloadable without authentication.
