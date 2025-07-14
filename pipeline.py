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
import signal

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, load_image
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from transformers import CLIPVisionModel
from image_gen_aux import DepthPreprocessor

from huggingface_hub import login

from generation import *
from prompt import *

# os.environ['HF_HOME'] = '/mnt/data/sysu/Users/wangzh/ID-Relight/models' # modified from '/home/herongshen/relighting/models'
# os.environ['HF_HOME'] = '/home/herongshen/relighting/models'

# os.environ['OPENAI_API_KEY'] = 'sk-43c5ff1fb1374626a56c2dba2b7d9789'
# os.environ['BASE_URL'] = 'https://api.deepseek.com'
# os.environ['MODEL_NAME'] = 'deepseek-chat'

os.environ['OPENAI_API_KEY'] = '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'
os.environ['BASE_URL'] = 'https://openrouter.ai/api/v1'
os.environ['MODEL_NAME'] = 'openai/gpt-4o'

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
login(token="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")

MODEL_ROOT = Path('/mnt/data/sysu/Users/wangzh/ID-Relight/repo')
                       
WIDTH=1024
HEIGHT=1024

os.environ['WIDTH'] = WIDTH
os.environ['HEIGHT'] = HEIGHT

# --- Basic Setup ---
# This is safe at the global level as it only configures logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# --- Placeholder for Model Loading & Generation ---
# Implement your actual model loading and inference logic here.

def arg_parse():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Data Generation Pipeline for Relighting")
    parser.add_argument('--mode', type=str, required=True, choices=['prompt-only', 'generation-only', 'all'],
                        help="Execution mode.")

    # --- Path Arguments ---
    parser.add_argument('--prompts_dir', type=str, default='./prompt', help="Directory to store prompt templates and generated outputs.")
    parser.add_argument('--prompts_file', type=str, default=None, help="Path to the JSON file with prompts. Required for 'generation-only'. If None in 'all' or 'prompt-only', a new one is created.")

    # --- Prompt Generation Arguments ---
    parser.add_argument('--num_relight_vairant', type=int, default=3, help="Total number of relighting image vairants to generate prompts for.")
    parser.add_argument('--num_video_vairant', type=int, default=1, help="Total number of video image vairants to generate prompts for.")
    parser.add_argument('--num_img_sets', type=int, default=10, help="Total number of image sets to generate prompts for.")
    parser.add_argument('--num_videos', type=int, default=10, help="Total number of T2V videos to generate prompts for.")
    parser.add_argument('--prompts_batch_size', type=int, default=5, help="Number of prompts to request from LLM in a single API call.")

    # --- Model & GPU Arguments ---
    parser.add_argument('--LLM_model_name', type=str, default="gpt-4-turbo-preview")


    # Add all your arguments here
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help="List of GPU IDs to use.")
    parser.add_argument('--batch_size', type=int, default=5, help="Batch size for media generation on each GPU.")

    parser.add_argument('--image_generation_model_name', type=str, default="black-forest-labs/FLUX.1-dev")
    parser.add_argument('--relighting_model_name', type=str, default="black-forest-labs/FLUX.1-Depth-dev-lora")
    parser.add_argument('--lbm_model', type=str, default="jasperai/LBM_relighting")
    parser.add_argument('--iclight_model', type=str, default="lllyasviel/IC-Light")
    parser.add_argument('--T2V_model_name', type=str, default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument('--I2V_model_name', type=str, default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")

    args = parser.parse_args()

    if args.num_img_sets % args.prompts_batch_size != 0 or \
       args.num_videos % args.prompts_batch_size != 0:
        logging.warning("For simplicity, it's recommended that num_sets/videos be divisible by prompts_batch_size. Adjusting numbers.")
        # Simple adjustment, you can make this smarter
        args.num_img_sets = (args.num_img_sets // args.prompts_batch_size) * args.prompts_batch_size
        args.num_videos = (args.num_videos // args.prompts_batch_size) * args.prompts_batch_size
        logging.info(f"Adjusted numbers: num_img_sets={args.num_img_sets}, num_videos={args.num_videos}")

    if args.mode == 'generation-only' and not args.prompts_file:
        parser.error("--prompts_file is required for 'generation-only' mode.")
    
    if args.mode in ['prompt-only', 'all'] and not args.prompts_file:
        i = 1
        while Path(args.prompts_dir, f"prompts_{i}.json").exists():
            i += 1
        args.prompts_file = Path(args.prompts_dir, f"prompts_{i}.json")
        logging.info(f"No prompts_file specified. A new file will be created at: {args.prompts_file}")
    else:
        args.prompts_file = Path(args.prompts_file)

    return args


def generation(args, prompts_dict=None):
    if args.mode in ['generation-only']:
        if not Path(args.prompts_file).exists():
            logging.error(f"Prompts file not found: {args.prompts_file}. Cannot run in 'generation-only' mode.")
            sys.exit(1)
        with open(args.prompts_file, 'r', encoding='utf-8') as f:
            prompts_dict = json.load(f)
    elif args.mode in ['all']:
        if prompts_dict is None:
            logging.error(f"prompts_dict is not provided in 'all' mode!")
            sys.exit(1)

    advanced_media_generation_pipeline(args, prompts_dict)
    logging.info("Relighting pipeline finished successfully.")

def prompting(args):
    # --- Execute Pipeline ---
    prompts_dict = None
    if args.mode in ['prompt-only', 'all']:
        prompts_dict = prompt_generation_pipeline(args)
    else: # generation-only
        try:
            with open(args.prompts_file, 'r', encoding='utf-8') as f:
                prompts_dict = json.load(f)
            logging.info(f"Successfully loaded prompts from {args.prompts_file}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Failed to load or parse prompts file: {e}")
            sys.exit(1)
    
    logging.info("Promping pipeline finished successfully.")

    return prompts_dict


def main():
    # Add other model name args if needed
    args = arg_parse()
    
    # --- Argument Validation and Path Setup ---
    Path(args.prompts_dir).mkdir(exist_ok=True)
    
    # Register signal handler
    signal.signal(signal.SIGINT, save_on_exit)
    
    prompts_dict = prompting(args)

    generation(args, prompts_dict=prompts_dict)
    



if __name__ == '__main__':
    # Set the start method for multiprocessing, must be done once and inside the __main__ block.
    # 'spawn' is required for CUDA safety.
    mp.set_start_method('spawn', force=True)
    
    # Call the main function to start the program
    main()
