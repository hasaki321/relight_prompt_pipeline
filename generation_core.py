import os
import json
import argparse
import sys
import logging
from pathlib import Path
from collections import defaultdict
import time
import traceback
import queue  # <-- Added import
import math
import numpy as np

import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
from PIL import Image

WIDTH= os.getenv("WIDTH")
HEIGHT= os.getenv("HEIGHT")

def batch_generate_images(tasks, model, device):
    """Generates a batch of images."""
    prompts = [task['prompt'] for task in tasks]
    output_paths = [task['output_path'] for task in tasks]
    logging.info(f"[{device}] Generating batch of {len(prompts)} images...")

    images = model(
        prompt=prompts, 
        num_inference_steps=50, 
        guidance_scale=3.5,
        width=WIDTH,
        height=HEIGHT,
    ).images
    for img, path in zip(images, output_paths):
        img.save(path)

    return output_paths

def batch_relight_images(tasks, model, device):
    """Relights a batch of images."""
    model, processor = model
    prompts = [task['prompt'] for task in tasks]
    base_image_paths = [task['base_image_path'] for task in tasks]
    output_paths = [task['output_path'] for task in tasks]
    logging.info(f"[{device}] Relighting batch of {len(prompts)} images...")

    images = [Image.open(p) for p in base_image_paths]
    images = [img.convert("RGB") for img in processor(images)] 
    relit_images = model(
        prompt=prompts,
        control_image=images,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=30,
        guidance_scale=10.0,
    ).images
    for img, path in zip(relit_images, output_paths):
        img.save(path)

    return output_paths

def batch_relight_videos(tasks, model, device):
    """Generates a batch of videos (T2V or I2V)."""
    prompts = [task['prompt'] for task in tasks]
    base_image_paths = [task['base_image_path'] for task in tasks]
    output_paths = [task['output_path'] for task in tasks]

    logging.info(f"[{device}] Relighting batch of {len(prompts)} videos...")

    # height = HEIGHT
    # width = WIDTH
    height = 512 # 430
    width = 512 # 832
    max_area = height * width
    aspect_ratio = height / width
    mod_value = model.vae_scale_factor_spatial * model.transformer.config.patch_size[1]
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    images = [Image.open(p).resize((width, height)) for p in base_image_paths]

    negative_prompts = ["Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"] * len(tasks)

    for i, path in enumerate(output_paths):
        mp4 = model(
            image=images[i],
            prompt=prompts[i],
            negative_prompt=negative_prompts[i],
            height=height,
            width=width,
            num_frames=81,
            guidance_scale=5.0,
            ).frames[0]
        export_to_video(mp4, path, fps=16)
        logging.info(f"[{device}] Video saved to {path}")
    return output_paths

def batch_generate_videos(tasks, model, device):
    """Generates a batch of videos (T2V or I2V)."""
    prompts = [task['prompt'] for task in tasks]
    output_paths = [task['output_path'] for task in tasks]

    logging.info(f"[{device}] Generating batch of {len(prompts)} videos...")

    negative_prompts = ["Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"] * len(tasks)

    frames = model(
         prompt=prompts,
         negative_prompt=negative_prompts,
         height=480, # 480
         width=832, # 832
         num_frames=81,
         guidance_scale=5.0,
        ).frames
    for mp4, path in zip(frames, output_paths):
        export_to_video(mp4, path, fps=16)
    return output_paths

def batch_lbm_images(tasks, model, device):
    """Relighting a batch of images using LBM"""
    lbm_t2i_pipe, birefnet = model
    def run_lbm(fg_image, prompt, width=WIDTH, height=HEIGHT, num_inference_steps=4):
        '''
        fg_image: PIL image, foreground image
        '''
        bg_image = lbm_t2i_pipe(
            prompt=prompt, 
            num_inference_steps=50, 
            guidance_scale=3.5,
            width=WIDTH,
            height=HEIGHT,
        ).images
        bg_image = bg_image[0]
        
        _, fg_mask = extract_object(birefnet, deepcopy(fg_image))
        fg_image = resize_and_center_crop(fg_image, width, height)
        fg_mask = resize_and_center_crop(fg_mask, width, height)
        
        img_pasted = Image.composite(fg_image, bg_image, fg_mask)

        img_pasted_tensor = ToTensor()(img_pasted).unsqueeze(0) * 2 - 1
        batch = {
            "source_image": img_pasted_tensor.cuda().to(torch.bfloat16),
        }

        z_source = lbm_model.vae.encode(batch[lbm_model.source_key])

        output_image = lbm_model.sample(
            z=z_source,
            num_steps=num_inference_steps,
            conditioner_inputs=batch,
            max_samples=1,
        ).clamp(-1, 1)

        output_image = (output_image[0].float().cpu() + 1) / 2
        output_image = ToPILImage()(output_image)

        # paste the output image on the background image
        output_image = Image.composite(output_image, bg_image, fg_mask)

        return output_image

    prompts = [task['background_prompt'] for task in tasks]
    base_image_paths = [task['base_image_path'] for task in tasks]
    output_paths = [task['output_path'] for task in tasks]
    logging.info(f"[{device}] Relighting batch of {len(prompts)} images by using LBM...")

    images = [Image.open(p) for p in base_image_paths]
    images = [img.convert("RGB") for img in images]
    
    for prompt, img, output_path in zip(prompts, images, output_paths):
        image_width, image_height = img.size
        num_inference_steps=4

        result = run_lbm(img, prompt, image_width, image_height, num_inference_steps)
        result.save(output_path)

    return output_paths

def batch_iclight_images(tasks, model, device):
    """Relighting a batch of images using IC-Light."""
    
    prompts = [task['background_prompt'] for task in tasks]
    lighting_prompts = [task['lighting_prompt'] for task in tasks]
    base_image_paths = [task['base_image_path'] for task in tasks]
    output_paths = [task['output_path'] for task in tasks]
    logging.info(f"[{device}] Relighting batch of {len(prompts)} images by using IC-Light...")

    images = [Image.open(p) for p in base_image_paths]
    images = [img.convert("RGB") for img in images]
    images = [np.array(img) for img in images]

    # Define parameters
    seed = 12345
    steps = 20
    a_prompt = "best quality"
    n_prompt = "lowres, bad anatomy, bad hands, cropped, worst quality"
    cfg = 7.0
    highres_scale = 1.5
    highres_denoise = 0.5
    num_samples = 1  # Adjust as needed

    for prompt, lighting_prompt, img, output_path in zip(prompts, lighting_prompts, images, output_paths):
        image_width, image_height, _ = img.shape
        ## Using the function called process_relight from IC-Light
        _ , result = model(
            img, # input_fg
            prompt,
            image_width,
            image_height,
            num_samples,
            seed,
            steps,
            a_prompt, 
            n_prompt, 
            cfg, 
            highres_scale, 
            highres_denoise,
            lighting_prompt=lighting_prompt
        )

        Image.fromarray(result[0]).save(output_path)

    return output_paths
