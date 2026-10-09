# CUDA_VISIBLE_DEVICES=5 python3 training/test.py \
# --detector_path training/config/detector/clip_inference.yaml \
# --test_dataset "AVDeepfake1M" \
# --weights_path logs/training/clip_2026-03-03-14-16-53/test/avg/ckpt_best.pth \
# --test_name "AVDF1M_test" 

# \
# > Terminal_outputs/inference_new/my_test_Clip_L_inference6_benchmark6_2g3_Correct_Large_PatchClS_TripleStream_3DConv_TopKFrame_FF-NT_EXP.log 2>&1 &
# # # # 


# CUDA_VISIBLE_DEVICES=5 python3 training/test.py \
# --detector_path training/config/detector/clip_inference.yaml \
# --test_dataset "WDF" "DFDC" "FaceForensics++" "Celeb-DF-v2" "DFDCP"   \
# --weights_path logs/training/clip_2026-03-03-15-31-36/test/avg/ckpt_best.pth \
# --test_name "Clip_L_benchmark5_1g4_PatchClS_No3DConv_TopKFrame_FM_0.6_CE_0.0_MIL_1_Var_0_DC_topk0.1_" \
# --save_patch_scores \
# --visualize_patchwise \
# > Terminal_outputs/WACV/Clip_L_benchmark5_1g4_PatchClS_No3DConv_TopKFrame_FM_0.6_CE_0.0_MIL_1_Var_0_DC_topk0.1.log 2>&1 &


# CUDA_VISIBLE_DEVICES=4 python3 training/test.py \
# --detector_path training/config/detector/clip_inference.yaml \
# --test_dataset "AVDeepfake1M" \
# --weights_path logs/training/clip_2026-02-28-22-47-26/test/avg/ckpt_best.pth \
# --test_name "Clip_L_inference5_PatchClS_TripleStream_3DConv_TopKFrame_1_0_0_FM_0.6_CE_0.0_MIL_1_Var_0_DC-topk0.1_AVDF" \
# > Terminal_outputs/inference_new/my_test_Clip_L_inference5_PatchClS_TripleStream_3DConv_TopKFrame_1_0_0_FM_0.6_CE_0.0_MIL_1_Var_0_DC-topk0.1_AVDF.log 2>&1 &

# "FaceShifter" "Celeb-DF-v1" "DFDC" "UADFV" "FF-F2F" "FF-DF" "FF-FS" "FF-NT" "FaceForensics++" "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" "WDF" "DeepFakeDetection" "UADFV"
# python3 training/test.py \
# --detector_path training/config/detector/xclip.yaml \
# --test_dataset "DFDC" \
# --weights_path ./training/weights/ffpp_best.pth
# "FaceForensics++" "Celeb-DF-v1" "Celeb-DF-v2" "DFDCP" "DFDC" "UADFV"
# WANDB_DISABLED=true

# "heygen"
# clip_2026-02-28-22-47-26 Ours Final


# CUDA_VISIBLE_DEVICES=1  python3 training/test.py \
# --detector_path training/config/detector/clip_inference.yaml \
# --test_dataset   "FRAll_ff" \
# --weights_path logs/training/clip_2026-05-10-03-52-22/test/avg/ckpt_best.pth \
# --test_name "Clip_L_inference7_DF_40_baseline_FRAll_ff" \
# > Terminal_outputs/inference_new/my_test_Clip_L_inference7_DF_40_baseline_FRAll_ff3.log 2>&1 &


# --save_patch_scores \
# --visualize_patchwise 
# \
# > Terminal_outputs/inference_new/my_test_Clip_L_inference5_PatchClS_TripleStream_3DConv_TopKFrame_0.8_0.1_0.1_FM_0.6_CE_0.0_MIL_1_Var_0_DC_topk0.1_001.log 2>&1 &


# clip_2026-02-12-01-56-53. my_output_clip_L_PatchClS_3DConv_Patch_frame_MIL_ranking_all_margin_0.5_augmented

# "fsgan_ff" "fsgan_cdf" "faceswap_ff" "faceswap_cdf" "simswap_ff" "simswap_cdf" "inswap_ff" "inswap_cdf" "blendface_ff" "blendface_cdf" "uniface_ff" "uniface_cdf" "mobileswap_ff" "mobileswap_cdf" "e4s_ff" "e4s_cdf" "facedancer_ff" "facedancer_cdf" "deepfacelab"

mkdir -p Terminal_outputs/WACV
WEIGHTS_PATH="${WEIGHTS_PATH:-logs/training/clip_2026-02-28-22-47-26/test/avg/ckpt_best.pth}"
if [ ! -f "$WEIGHTS_PATH" ]; then
    echo "Checkpoint not found: $WEIGHTS_PATH. Set WEIGHTS_PATH to an existing checkpoint." >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=4 python3 training/test.py \
--detector_path training/config/detector/clip_inference.yaml \
--test_dataset "fomm_ff" "fomm_cdf" "facevid2vid_ff" "facevid2vid_cdf" "wav2lip_ff" "wav2lip_cdf" "MRAA_ff" "MRAA_cdf" "one_shot_free_ff" "one_shot_free_cdf" "pirender_ff" "pirender_cdf" "tpsm_ff" "tpsm_cdf" "lia_ff" "lia_cdf" "danet_ff" "danet_cdf" "sadtalker_ff" "sadtalker_cdf" "mcnet_ff" "mcnet_cdf" "hyperreenact_ff" "hyperreenact_cdf" "heygen" \
--weights_path "$WEIGHTS_PATH" \
--test_name "Clip_L_inference8_DF_40_Face_renanctment" \
--visualize_patchwise \
> Terminal_outputs/WACV/my_test_Clip_L_inference8_DF_40_Face_renanctment.log 2>&1 &
