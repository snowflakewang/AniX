#!/bin/bash

JOBS_DIR=$(dirname $(dirname "$0"))
export PYTHONPATH=${JOBS_DIR}:$PYTHONPATH
export MODEL_BASE="models"
checkpoint_path=${MODEL_BASE}"/base/hunyuancustom_editing_720P/mp_rank_00_model_states.pt"

export PYTHONPATH=./
export CPU_OFFLOAD=0
# export DISABLE_SP=1

export NPROC_PER_NODE=8
export NNODES=1
export lora_path=${MODEL_BASE}"/lora/pytorch_lora_kohaya_weights.safetensors"
export seed=3407

scene_name="futureUtopia"
character_name="orangeRobot"

echo "----- scene name ------"
echo "$scene_name"
echo "----------------------------------------"

echo "--- character name ---"
echo "$character_name"
echo "----------------------------------------"

echo "--- LoRA path ---"
echo "LoRA path: ${lora_path}"
echo "----------------------------------------"

scene_input_folder_name="input/scene_assets/${scene_name}"
character_input_folder_name="input/character_assets/${character_name}"

output_folder_name="output/${character_name}_${scene_name}"

torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
hymm_sp/inference_multi_gpu.py \
    --input "${character_input_folder_name}/input_list/character.list" \
    --input-video "${character_input_folder_name}/input_list/ar_cond.list" \
    --output-video "${character_input_folder_name}/input_list/output_video.list" \
    --input-scene-video "${character_input_folder_name}/input_list/scene.list" \
    --input-scene-mask-video "${character_input_folder_name}/input_list/scene_mask.list" \
    --pos-prompt "${character_input_folder_name}/input_list/pos_prompt.list" \
    --video-size 704 1248 \
    --bucket-sample-size 960 960 \
    --expand-scale 0 \
    --video-condition \
    --neg-prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion, blurring, text, subtitles, static, picture, black border." \
    --ckpt "${checkpoint_path}" \
    --sample-n-frames 129 \
    --cfg-scale 7.5 \
    --seed ${seed} \
    --infer-steps 30 \
    --use-deepcache 1 \
    --flow-shift-eval-video 13.0 \
    --multi-view-ids 0 2 4 6 \
    --num-clean-latent-frames 9 \
    --use-lora \
    --lora-rank 64 \
    --lora-scale 1.0 \
    --scene-mask-type "INPAINTED" \
    --lora-path "${lora_path}" \
    --save-path "${output_folder_name}"