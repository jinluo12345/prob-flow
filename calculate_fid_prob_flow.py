
#!/usr/bin/env python
# coding: utf-8
"""
Script to generate images with RectifiedFlowPipeline, resize real images to 512x512 if needed,
and calculate FID score between real and generated images.
"""

import os
import random
from PIL import Image
import torch
from tqdm import tqdm
from datasets import load_dataset
from pytorch_fid.fid_score import calculate_fid_given_paths
from code.pipeline_rf import RectifiedFlowPipeline,RectifiedFlowPipelineWithVar
from diffusers import StableDiffusionPipeline
import numpy as np
from train_prob_flow import UNet2DConditionModelWithVariance
#from modelscope import StableDiffusionPipeline
# Base folder for COCO images
PATH_TO_IMAGE_FOLDER = "COCO2017/val2017"
# Folder to save resized real images
RESIZED_REAL_FOLDER = "COCO2017/val2017_512"
# Folder to save generated images
GENERATED_FOLDER = "COCO2017/3rf_prob_generated"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# Ensure deterministic behavior (may impact performance)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def convert_val_images(input_dir: str, output_dir: str, size: int = 512):
    """
    Resize all images under input_dir to size x size and save to output_dir.
    Skips files that already exist in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in tqdm(os.listdir(input_dir), desc="Converting real images to 512x512"):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue
        src_path = os.path.join(input_dir, fname)
        dst_path = os.path.join(output_dir, fname)
        if os.path.exists(dst_path):
            # Skip already resized image
            continue
        try:
            with Image.open(src_path) as img:
                img = img.convert('RGB')
                img = img.resize((size, size), Image.LANCZOS)
                img.save(dst_path)
        except Exception as e:
            print(f"Failed to process {fname}: {e}")


def create_full_path(example):
    """
    Append base folder path to the file name to get full image path.
    """
    example["image_path"] = os.path.join(RESIZED_REAL_FOLDER, example["file_name"])
    return example


def load_and_process_dataset(dataset_path: str, n_samples: int = 5000):
    """
    Load the dataset, extract captions and (resized) image paths,
    then randomly sample n_samples of them.
    """
    # Load Hugging Face dataset
    dataset = load_dataset(dataset_path)
    # Map to add image_path field pointing to resized images
    dataset = dataset['validation'].map(create_full_path)
    captions = dataset['captions']
    image_paths = dataset['image_path']

    # Randomly sample indices
    sampled_indices = random.sample(range(len(captions)), n_samples)
    sampled_captions = [captions[i] for i in sampled_indices]
    sampled_image_paths = [image_paths[i] for i in sampled_indices]

    return sampled_captions, sampled_image_paths


def generate_images(pipe: RectifiedFlowPipeline, captions: list, output_dir: str,
                    num_inference_steps: int = 50, batch_size: int = 1):
    """
    Generate images from captions using RectifiedFlowPipeline.
    Skips any outputs flagged as NSFW.
    Returns a list of file paths for the saved (non-NSFW) images.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_images = []
    total = len(captions)

    for i in tqdm(range(0, total, batch_size), desc="Generating images"):
        raw_batch = captions[i:i + batch_size]
        batch = [random.choice(c) if isinstance(c, (list, tuple)) else c for c in raw_batch]
        outputs = pipe(
            batch,
            negative_prompt=["painting, unreal, twisted"] * len(batch),
            num_inference_steps=num_inference_steps,
            guidance_scale=1.5
        )

        for j, (img, nsfw_flag) in enumerate(zip(outputs.images, outputs.nsfw_content_detected)):
            prompt_text = batch[j]
            if nsfw_flag:
                print(f"Skipping NSFW image for prompt: {prompt_text!r}")
                continue  
            idx = len(generated_images)
            filename = os.path.join(output_dir, f"generated_{idx:06d}.png")
            img.save(filename)
            generated_images.append(filename)

    return generated_images


def calculate_fid_score(real_image_folder: str, generated_image_folder: str) -> float:
    """
    Calculate the FID score between two image folders.
    Uses num_workers=0 to avoid multiprocessing issues.
    """
    paths = [real_image_folder, generated_image_folder]
    fid_value = calculate_fid_given_paths(
        paths,
        batch_size=50,
        device='cuda',
        dims=2048,
        num_workers=0  # single-process DataLoader
    )
    return fid_value


def main():
    # Configuration parameters
    dataset_path = "phiyodr/coco2017"  # HF dataset identifier
    n_samples = 5000                       # Number of samples for quick test
    #model_path = "AI-ModelScope/stable-diffusion-v1-5"  # Pretrained model
    #model_path = "/remote-home1/lzjjin/.cache/modelscope/hub/models/AI-ModelScope/stable-diffusion-v1-5"  # Pretrained model
    model_path = "XCLIU/2_rectified_flow_from_sd_1_5"
    unet_path='/remote-home1/lzjjin/project/InstaFlow/flow-model-finetuned/checkpoint-2000/unet'
    num_inference_steps = 25           # Diffusion steps for generation
    batch_size = 12                  # Batch size per generation call

    # 1. Convert real images to 512x512 if not already done
    convert_val_images(PATH_TO_IMAGE_FOLDER, RESIZED_REAL_FOLDER, size=512)

    # 2. Load captions and real image paths
    captions, real_paths = load_and_process_dataset(dataset_path, n_samples)

    # 3. Initialize the RectifiedFlow generation pipeline
    #pipe = StableDiffusionPipeline.from_pretrained(model_path,torch_dtype=torch.float32)
    
    unet=UNet2DConditionModelWithVariance.from_pretrained(unet_path,torch_dtype=torch.float32)
    pipe = RectifiedFlowPipelineWithVar.from_pretrained(model_path,unet=unet,torch_dtype=torch.float32)

    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # 4. Generate images based on sampled captions
    generated_images = generate_images(
        pipe, captions, GENERATED_FOLDER,
        num_inference_steps=num_inference_steps,
        batch_size=batch_size
    )

    # 5. Compute FID score between resized real and generated images
    fid_score = calculate_fid_score(RESIZED_REAL_FOLDER, GENERATED_FOLDER)
    print(f"FID score between real and generated images: {fid_score}")

if __name__ == '__main__':
    main()

