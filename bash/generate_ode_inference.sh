# torchrun --nproc_per_node=1 generate_ode_inference.py \
#     --model_path "XCLiu/2_rectified_flow_from_sd_1_5" \
#     --dataset_path /remote-home1/share/hf_cache/huggingface/datasets/jasonhuang23___laion2b_en_sd2.1base/default/0.0.0/37b285ce60d3c62fdabbbf1cf4f8ec3effcc47f0 \
#     --data_files laion2b_en_sd2.1base-train-00100-of-00150.arrow \
#     --output_dir ./outputs \
#     --num_inference_steps 40 \
#     --batch_size 16 \
#     --fp16 \
#     --max_images 1000000 

# torchrun --nproc_per_node=1 generate_ode_inference.py \
#     --model_path "XCLiu/2_rectified_flow_from_sd_1_5" \
#     --dataset_path "jasonhuang23/laion2b_en_sd2.1base" \
#     --output_dir ./outputs \
#     --num_inference_steps 40 \
#     --batch_size 16 \
#     --fp16 \
#     --max_images 1000000

torchrun --nproc_per_node=8 generate_ode_inference.py \
    --model_path "XCLIU/2_rectified_flow_from_sd_1_5" \
    --dataset_path /remote-home1/share/hf_cache/huggingface/datasets/jasonhuang23___laion2b_en_sd2.1base/default/0.0.0/37b285ce60d3c62fdabbbf1cf4f8ec3effcc47f0 \
    --output_dir ./outputs_huge \
    --num_inference_steps 25 \
    --batch_size 32 \
    --fp16 \
    --max_images 600000  \
    --traverse