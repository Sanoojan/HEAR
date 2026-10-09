
mkdir -p Terminal_outputs

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=29300 training/train.py \
    --detector_path training/config/detector/clip.yaml \
    --test_name Hear \
    --ddp > Terminal_outputs/Hear.log 2>&1 &