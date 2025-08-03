#!/bin/bash

# initialize conda in *this* shell
source /root/miniconda3/etc/profile.d/conda.sh
conda activate cosmos-transfer1

# Change to cosmos-transfer1 directory
cd /root/cosmos-transfer1

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:=0}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:=./checkpoints}"
export NUM_GPU="${NUM_GPU:=1}"
PYTHONPATH=$(pwd) torchrun --nproc_per_node=$NUM_GPU --nnodes=1 --node_rank=0 cosmos_transfer1/diffusion/inference/transfer.py \
    --prompt "A white robotic gripper in bright morning lighting picks up a small wooden rectangular box from a table and places it into a yellow box."\
    --checkpoint_dir $CHECKPOINT_DIR \
    --video_save_folder outputs/augmented \
    --input_video_path /root/synphony/output.mp4 \
    --controlnet_specs assets/augment.json \
    --offload_text_encoder_model \
    --offload_guardrail_models \
    --num_gpus $NUM_GPU \
    --offload_prompt_upsampler \
    --blur_strength low \