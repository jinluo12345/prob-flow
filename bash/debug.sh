ACCELERATE_DISTRIBUTED_TYPE=MULTI_GPU \
set HF_ENDPOINT=https://hf-mirror.com
accelerate launch --num_processes 8  --num_machines 1  --multi-gpu --mixed_precision="fp16"  train_prob_flow_pick.py --config config/pickapick.yaml 