import os
import json
import argparse
import random
from datasets import load_dataset
import torch
import torch.distributed as dist
from tqdm import tqdm
import sys

sys.path.append('.')
sys.path.append('./')
from code.pipeline_rf import RectifiedFlowPipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-GPU inference with RectifiedFlowPipeline")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model or directory for .arrow files")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="HF dataset identifier or local dataset directory")
    parser.add_argument("--data_files", nargs="+", default=None,
                        help="Specific .arrow files under dataset_path (omit to load full split)")
    parser.add_argument("--traverse", action="store_true",
                        help="Traverse dataset_path for .arrow files in single-process mode")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated images and metadata")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps for generation")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Prompts per forward pass")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 precision for model inference")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Global limit of images to generate across all GPUs")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)),
                        help="Process rank (set by torchrun)")
    return parser.parse_args()


def find_arrow_files(root_dir):
    """
    Find .arrow files in root_dir and its immediate subdirectories.
    """
    files = []
    for entry in os.listdir(root_dir):
        path = os.path.join(root_dir, entry)
        if os.path.isfile(path) and entry.endswith('.arrow'):
            files.append(path)
        elif os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith('.arrow'):
                    files.append(os.path.join(path, f))
    return files


def main():
    args = parse_args()

    # 1. Initialize distributed environment
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = (rank == 0)

    # 2. Prepare model pipeline
    pipe = RectifiedFlowPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # 3. Build list of data sources (arrow files or HF dataset)
    sources = []  # list of tuples: (label, iterator or path)
    if args.dataset_path and not args.traverse:
        ds = (load_dataset(args.dataset_path, data_files=args.data_files, split='train')
              if args.data_files else load_dataset(args.dataset_path, split='train'))
        sources.append(('dataset', ds['TEXT']))
    else:
        root = args.dataset_path if args.dataset_path else args.model_path
        if args.traverse or not args.dataset_path:
            if not os.path.isdir(root):
                raise ValueError(f"Directory not found: {root}")
            arrow_files = find_arrow_files(root)
            if not arrow_files:
                raise ValueError(f"No .arrow files found in {root}")
            #random.shuffle(arrow_files)
            for f in arrow_files:
                sources.append((os.path.basename(f), f))

    # 4. Determine per-GPU limit
    per_gpu = (args.max_images + world_size - 1) // world_size if args.max_images else None

    # 5. Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'noise'), exist_ok=True)  # Create noise folder
    meta_tmp = os.path.join(args.output_dir, f"metadata_rank{rank}.jsonl")

    # 6. Sequential processing per source
    generated = 0
    global_limit = args.max_images or float('inf')

    with open(meta_tmp, 'w', encoding='utf-8') as meta_fp:
        for label, src in sources:
            # load prompts list
            if isinstance(src, str) and src.endswith('.arrow'):
                ds_f = load_dataset('arrow', data_files=src, split='train')
                prompts = ds_f['TEXT']
            else:
                prompts = src

            # Calculate the total number of batches
            num_batches = len(prompts) // args.batch_size + (1 if len(prompts) % args.batch_size > 0 else 0)

            # Split the dataset across GPUs
            # Each rank processes a subset of the data (using `rank` to select the correct subset)
            split_prompts = [prompts[i::world_size] for i in range(world_size)]
            prompts_for_rank = split_prompts[rank]

            # Pick iteration method to avoid output on non-main
            indices = range(0, len(prompts_for_rank), args.batch_size)
            iterator = tqdm(indices, total=len(prompts_for_rank) // args.batch_size, desc=f"GPU {rank} - {label}") if is_main else indices

            for i in iterator:
                if generated >= (per_gpu or float('inf')) or generated >= global_limit:
                    break
                batch = prompts_for_rank[i:i + args.batch_size]
                remaining = min(per_gpu or len(batch), int(global_limit - generated))
                batch = batch[:remaining]

                # Skip prompts that are too long (more than 77 tokens)
                batch = [text for text in batch if len(pipe.tokenizer(text)["input_ids"]) <= 77]
                if not batch:
                    continue  # If batch is empty after filtering, skip this iteration

                # try:
                # Run the pipeline
                outputs = pipe(batch, num_inference_steps=args.num_inference_steps,guidance_scale=1.5)

                # Check for NSFW and handle truncation
                for img, noise_tensor, text, nsfw_flag in zip(outputs.images, outputs.noise, batch, outputs.nsfw_content_detected):
                    if nsfw_flag:
                        print(f"Skipping NSFW image for prompt: {text}")
                        continue  # Skip the NSFW image

                    idx = rank * (per_gpu or 0) + generated if per_gpu else generated
                    image_fname = f"image/{idx:06d}.png"
                    noise_fname = f"noise/{idx:06d}.pt"  # Save noise image with a different name
                    
                    # Save image and noise
                    img.save(os.path.join(args.output_dir, image_fname))
                    torch.save(noise_tensor.detach().cpu(),
                        os.path.join(args.output_dir, noise_fname))

                    # Write metadata
                    json.dump({'file_name': image_fname, 'noise_file': noise_fname, 'text': text}, meta_fp, ensure_ascii=False)
                    meta_fp.write('\n')
                    generated += 1

                # except Exception as e:
                #     print(f"Error processing batch {i} for prompt: {batch}. Error: {str(e)}")
                #     continue  # Skip to next batch if error occurs

            # Release memory
            if isinstance(src, str) and src.endswith('.arrow'):
                del ds_f, prompts
                torch.cuda.empty_cache()

            if generated >= (per_gpu or float('inf')) or generated >= global_limit:
                break

    # 7. Synchronize and merge metadata
    dist.barrier()
    if is_main:
        final_meta = os.path.join(args.output_dir, 'metadata.jsonl')
        with open(final_meta, 'w', encoding='utf-8') as fout:
            for r in range(world_size):
                tmp = os.path.join(args.output_dir, f"metadata_rank{r}.jsonl")
                if os.path.exists(tmp):
                    with open(tmp, 'r', encoding='utf-8') as fin:
                        fout.writelines(fin)
                    os.remove(tmp)
        print(f"All done! Generated up to {args.max_images or 'all available'} images across {world_size} GPUs. Output in {args.output_dir}")

if __name__ == '__main__':
    main()
