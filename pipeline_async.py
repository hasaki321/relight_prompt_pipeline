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

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, load_image
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from transformers import CLIPVisionModel
from image_gen_aux import DepthPreprocessor

from huggingface_hub import login
os.environ['HF_HOME'] = '/home/herongshen/relighting/models'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
login(token="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")

WIDTH=1280
HEIGHT=720

# --- Basic Setup ---
# This is safe at the global level as it only configures logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- Placeholder for Model Loading & Generation ---
# Implement your actual model loading and inference logic here.

def load_model(model_name, device):
    """Loads a model onto a specific device."""
    logging.info(f"[{device}] Loading model: {model_name}...")
    if "FLUX.1-Depth-dev-lora" in model_name:
        model = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda:6")
        model.load_lora_weights("black-forest-labs/FLUX.1-Depth-dev-lora", adapter_name="depth")
        model.set_adapters("depth", 0.85)
        model.to(device)
        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        model = (model, processor)
    elif "FLUX.1-Depth-dev" in model_name:
        model = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16).to("cuda")
        model.to(device)
        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        model = (model, processor)
    elif "FLUX" in model_name:
        model = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        model.to(device)
    elif "instruct-pix2pix" in model_name:
        model = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.bfloat16, safety_checker=None)
        model.to(device)
        model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
    elif "Wan2.1-T2V" in model_name:
        model_id = model_name
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.bfloat16)
        flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        model = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        model.scheduler = scheduler
        model.to(device)
    elif "Wan2.1-I2V" in model_name:
        model_id = model_name
        image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        model = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
        model.to(device)
    else:
        time.sleep(1)  # Simulate model loading time
        logging.info(f"[{device}] Model {model_name} loaded.")
        return {"name": model_name, "device": device} # Dummy object
    return model

def unload_model(model):
    """Unloads a model to free up GPU memory."""
    if model:
        logging.info(f"Unloading model: {model}...")
        del model
        torch.cuda.empty_cache()
        logging.info(f"[Memory cleared.]")

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
    height = 480
    width = 832
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
         height=480,
         width=832,
         num_frames=81,
         guidance_scale=5.0,
        ).frames
    for mp4, path in zip(frames, output_paths):
        export_to_video(mp4, path, fps=16)
    return output_paths

# --- The Worker Process ---
def worker(task_queue, shared_data_proxy, data_lock, args, device):
    """A worker process that runs on a single GPU."""
    process_name = mp.current_process().name
    logging.info(f"[{process_name}] assigned to {device}.")

    current_model = None
    current_model_name = None

    while True:
        try:
            task_batch = task_queue.get()
            if task_batch is None:
                logging.info(f"[{process_name}] Received sentinel. Shutting down.")
                break
            task_type = task_batch[0]['type']
            model_name = task_batch[0]['model_name']

            if model_name != current_model_name:
                unload_model(current_model)
                current_model = load_model(model_name, device)
                current_model_name = model_name

            if task_type == 'generate_image':
                batch_generate_images(task_batch, current_model, device)
            elif task_type == 'relight_image':
                batch_relight_images(task_batch, current_model, device)
            elif task_type == 'i2v':
                batch_relight_videos(task_batch, current_model, device)
            elif task_type == 't2v':
                batch_generate_videos(task_batch, current_model, device)

            with data_lock:
                # To modify the shared dict proxy, we need to get it, modify it, and set it back
                # This is a nuance of multiprocessing.Manager.dict
                current_data = dict(shared_data_proxy)
                for task in task_batch:
                    if task['type'] == 'generate_image':
                        current_data['editing_sets'][task['set_idx']]['path'] = task['output_path']
                    elif task['type'] == 'relight_image':
                        current_data['editing_sets'][task['set_idx']]['relighting_prompts'][task['relight_idx']]['path'] = task['output_path']
                    elif task['type'] == 'i2v':
                        current_data['editing_sets'][task['set_idx']]['video_relighting_prompts'][task['video_idx']]['path'] = task['output_path']
                    elif task['type'] == 't2v':
                        current_data['t2v_prompts'][task['video_idx']]['path'] = task['output_path']
                
                # Update the shared proxy
                shared_data_proxy.update(current_data)

                # Save progress to disk after each batch
                with open(args.prompts_file, 'w', encoding='utf-8') as f:
                    json.dump(current_data, f, indent=4, ensure_ascii=False)
            
            logging.info(f"[{process_name}] Completed batch of {len(task_batch)} {task_type} tasks. Progress saved.")

        except queue.Empty:
            # This should not happen with a blocking get(), but is good practice
            logging.info(f"[{process_name}] Queue empty. This should not be reached.")
            continue
        except Exception:
            logging.error(f"[{process_name}] Unhandled exception in worker loop:")
            logging.error(traceback.format_exc())
            break
            
    unload_model(current_model)
    logging.info(f"[{process_name}] Worker finished.")

# --- The Main Media Generation Pipeline ---
def advanced_media_generation_pipeline(args, prompts_dict):
    """Manages the media generation process using a multi-process worker pool."""
    # The Manager should be created here, inside the function that uses it.
    manager = Manager()
    shared_data = manager.dict(prompts_dict)
    data_lock = manager.Lock()
    task_queue = manager.Queue()

    all_tasks = defaultdict(list)
    base_output_dir = Path(args.prompts_file).stem

    # Task 1: Base Image Generation
    for i, item in enumerate(shared_data["editing_sets"]):
        if not item.get("path") or not Path(item.get("path")).exists():
            output_dir = Path(args.prompts_dir) / base_output_dir / "editing" / item["set_id"]
            output_dir.mkdir(parents=True, exist_ok=True)
            all_tasks[args.image_generation_model_name].append({
                'type': 'generate_image', 'model_name': args.image_generation_model_name,
                'set_idx': i, 'prompt': item["base_generation_prompt"],
                'output_path': str(output_dir / "base_img.png")})

    # Task 2, 3: Relighting and I2V
    for i, item in enumerate(shared_data["editing_sets"]):
        base_path = item.get("path")
        if not base_path or not Path(base_path).exists():
            continue
        # Relighting
        relight_dir = Path(base_path).parent / "relighting"
        relight_dir.mkdir(exist_ok=True)
        for j, p in enumerate(item["relighting_prompts"]):
            if not p.get("path") or not Path(p.get("path")).exists():
                 all_tasks[args.relighting_model_name].append({
                    'type': 'relight_image', 'model_name': args.relighting_model_name,
                    'set_idx': i, 'relight_idx': j, 'prompt': p["prompt"],
                    'base_image_path': base_path, 'output_path': str(relight_dir / f"variant_{j}.png")})

    for i, item in enumerate(shared_data["editing_sets"]):
        base_path = item.get("path")
        if not base_path or not Path(base_path).exists():
            continue
        # I2V
        video_dir = Path(base_path).parent / "video"
        video_dir.mkdir(exist_ok=True)
        for j, p in enumerate(item["video_relighting_prompts"]):
             if not p.get("path") or not Path(p.get("path")).exists():
                 all_tasks[args.I2V_model_name].append({
                    'type': 'i2v', 'model_name': args.I2V_model_name,
                    'set_idx': i, 'video_idx': j, 'prompt': p["prompt"],
                    'base_image_path': base_path, 'output_path': str(video_dir / f"variant_i2v_{j}.mp4")})
    # print(all_tasks[args.I2V_model_name])
    # Task 4: T2V Video Generation
    for i, item in enumerate(shared_data["t2v_prompts"]):
        if not item.get("path") or not Path(item.get("path")).exists():
            output_dir = Path(args.prompts_dir) / base_output_dir / "video"
            output_dir.mkdir(parents=True, exist_ok=True)
            all_tasks[args.T2V_model_name].append({
                'type': 't2v', 'model_name': args.T2V_model_name,
                'video_idx': i, 'prompt': item["prompt"],
                'output_path': str(output_dir / f"{item['id']}.mp4")})

    total_batches = 0
    for model_name, tasks in all_tasks.items():
        batch_size = args.batch_size
        if ("Wan" in model_name) or ("Vace" in model_name): batch_size = 2 #math.ceil(batch_size / 3)
        logging.info(f"Creating {len(tasks)} tasks for model {model_name} with batch size {batch_size}")
        for i in range(0, len(tasks), batch_size):
            task_queue.put(tasks[i:i + batch_size])
            total_batches += 1
    
    if total_batches == 0:
        logging.info("No new tasks to process. Everything is up to date.")
        return

    logging.info(f"Populated queue with {total_batches} batches of tasks.")

    processes = []

    for i in range(len(args.gpus)):
        p = mp.Process(target=worker, name=f"Worker-GPU-{args.gpus[i]}", args=(task_queue, shared_data, data_lock, args, f"cuda:{args.gpus[i]}"))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logging.info("All worker processes have finished.")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Advanced Data Generation Pipeline")
    # Add all your arguments here
    parser.add_argument('--mode', type=str, default='generation-only', choices=['prompt-only', 'generation-only', 'all'])
    parser.add_argument('--prompts_dir', type=str, default='./prompt')
    parser.add_argument('--prompts_file', type=str, required=True, help="Path to the JSON file with prompts.")
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help="List of GPU IDs to use.")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size for media generation on each GPU.")
    parser.add_argument('--image_generation_model_name', type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument('--relighting_model_name', type=str, default="black-forest-labs/FLUX.1-Depth-dev-lora")
    parser.add_argument('--T2V_model_name', type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument('--I2V_model_name', type=str, default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
    # ... add prompt generation args if needed for 'all' or 'prompt-only' modes
    args = parser.parse_args()
    
    # Placeholder for prompt generation logic if mode is 'all' or 'prompt-only'
    if args.mode in ['prompt-only', 'all']:
        logging.error("Prompt generation logic is not included in this script version. Please run in 'generation-only' mode.")
        # Here you would call your prompt_generation_pipeline
        sys.exit(1)

    if not Path(args.prompts_file).exists():
        logging.error(f"Prompts file not found: {args.prompts_file}. Cannot run in 'generation-only' mode.")
        sys.exit(1)
        
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts_dict = json.load(f)

    advanced_media_generation_pipeline(args, prompts_dict)


# This is the crucial part!
if __name__ == '__main__':
    # Set the start method for multiprocessing, must be done once and inside the __main__ block.
    # 'spawn' is required for CUDA safety.
    mp.set_start_method('spawn', force=True)
    
    # Call the main function to start the program
    main()