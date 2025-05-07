
accelerate launch --num_processes 1  --num_machines 1 --mixed_precision="fp16" train_prob_flow_new.py --config config/train_flow_prob_from_scratch.yaml 