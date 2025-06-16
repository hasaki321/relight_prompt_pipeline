import os
import json
import argparse
import sys
import signal
import logging
from pathlib import Path
from itertools import cycle
from tqdm import tqdm
import torch
from openai import OpenAI

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

from huggingface_hub import login
os.environ['HF_HOME'] = '/home/herongshen/relighting/models'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
login(token="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")

# Make sure you have installed diffusers, transformers, accelerate
# from diffusers import AutoPipelineForText2Image, InstructPix2PixPipeline, DiffusionPipeline
# import openai # Or your preferred LLM library

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

# Global variables for signal handler to save progress on exit
PROMPT_DATA = {}
JSON_SAVE_PATH = None

def save_on_exit(signum, frame):
    """Signal handler to save progress before exiting."""
    logging.warning(f"Signal {signum} received. Saving progress before exiting...")
    if PROMPT_DATA and JSON_SAVE_PATH:
        try:
            with open(JSON_SAVE_PATH, 'w', encoding='utf-8') as f:
                json.dump(PROMPT_DATA, f, indent=4, ensure_ascii=False)
            logging.info(f"Progress successfully saved to {JSON_SAVE_PATH}")
        except Exception as e:
            logging.error(f"Failed to save progress on exit: {e}")
    sys.exit(0)

def save_progress():
    """Utility function to save the global prompt data."""
    if PROMPT_DATA and JSON_SAVE_PATH:
        with open(JSON_SAVE_PATH, 'w', encoding='utf-8') as f:
            json.dump(PROMPT_DATA, f, indent=4, ensure_ascii=False)

# --- Placeholder for API/Model Calls ---
# You need to implement these with your actual model loading and inference logic
def call_llm_api(system_prompt, user_prompt):
    """Placeholder for calling a Large Language Model API."""
    logging.info("Calling LLM API to generate prompts...")
    api_key = "sk-43c5ff1fb1374626a56c2dba2b7d9789"
    # api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            'type': 'json_object'
        },
        stream=False
    )
    print(json.loads(response.choices[0].message.content))
    return response.choices[0].message.content
# --- Pipeline Logic ---

def prompt_generation_pipeline(args):
    """Generates the prompt JSON file using an LLM."""
    global PROMPT_DATA, JSON_SAVE_PATH
    JSON_SAVE_PATH = args.prompts_file
    
    # Initialize prompt data structure
    PROMPT_DATA = {"editing_sets": [], "t2v_prompts": []}

    try:
        background_prompt = Path(args.prompts_dir, 'background.txt').read_text()
        editing_task_prompt = Path(args.prompts_dir, 'editing_task.txt').read_text()
        video_task_prompt = Path(args.prompts_dir, 'video_task.txt').read_text()
    except FileNotFoundError as e:
        logging.error(f"Prompt template file not found: {e}. Please create it.")
        sys.exit(1)

    # --- Editing Task Generation ---
    num_editing_batches = args.num_img_sets // args.prompts_batch_size
    logging.info(f"Generating {args.num_img_sets} editing prompts in {num_editing_batches} batches...")

    editing_counter = 0
    for i in tqdm(range(num_editing_batches), desc="Editing Prompt Batches"):
        instruction = f"\n\n**Instruction:** Now, generate **{args.prompts_batch_size}** unique sets with **{args.num_relight_vairant}** unique relighting_prompts vairants and **{args.num_video_vairant}** unique video_relighting_prompts vairants following the JSON schema provided above."
        user_prompt = editing_task_prompt + instruction
        try:
            response_json_str = call_llm_api(background_prompt, user_prompt)
            response_data = json.loads(response_json_str)

            for i in range(len(response_data["editing_sets"])):
                response_data["editing_sets"][i]["set_id"] = f"set_{editing_counter + i}"
            editing_counter += len(response_data["editing_sets"])
            PROMPT_DATA["editing_sets"].extend(response_data.get("editing_sets", []))
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse LLM response for editing prompts (Batch {i+1}): {e}")
            continue
    logging.info(f"Generated {editing_counter} editing prompt sets at all, required number: {args.num_videos}")

    # --- Video Task Generation ---
    video_counter = 0
    num_video_batches = args.num_videos // args.prompts_batch_size
    logging.info(f"Generating {args.num_videos} T2V prompts in {num_video_batches} batches...")
    for i in tqdm(range(num_video_batches), desc="T2V Prompt Batches"):
        instruction = f"\n\n**Instruction:** Now, generate **{args.prompts_batch_size}** unique prompts following the JSON schema provided above."
        user_prompt = video_task_prompt + instruction
        try:
            response_json_str = call_llm_api(background_prompt, user_prompt)
            response_data = json.loads(response_json_str)

            for i in range(len(response_data["t2v_prompts"])):
                response_data["t2v_prompts"][i]["id"] = f"video_{video_counter + i}"
            video_counter += len(response_data["t2v_prompts"])
            PROMPT_DATA["t2v_prompts"].extend(response_data.get("t2v_prompts", []))
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse LLM response for video prompts (Batch {i+1}): {e}")
            continue
    logging.info(f"Generated {video_counter} video prompts at all, required number: {args.num_videos}")

    save_progress()
    logging.info(f"Prompt generation complete. Saved to {JSON_SAVE_PATH}")
    return PROMPT_DATA

def main():
    parser = argparse.ArgumentParser(description="Data Generation Pipeline for Video Relighting")
    parser.add_argument('--mode', type=str, required=True, choices=['prompt-only', 'generation-only', 'all'],
                        help="Execution mode.")
    
    # --- Path Arguments ---
    parser.add_argument('--prompts_dir', type=str, default='./prompt',
                        help="Directory to store prompt templates and generated outputs.")
    parser.add_argument('--prompts_file', type=str, default=None,
                        help="Path to the JSON file with prompts. Required for 'generation-only'. If None in 'all' or 'prompt-only', a new one is created.")

    # --- Prompt Generation Arguments ---
    parser.add_argument('--num_relight_vairant', type=int, default=3, help="Total number of relighting image vairants to generate prompts for.")
    parser.add_argument('--num_video_vairant', type=int, default=1, help="Total number of video image vairants to generate prompts for.")
    parser.add_argument('--num_img_sets', type=int, default=10, help="Total number of image sets to generate prompts for.")
    parser.add_argument('--num_videos', type=int, default=10, help="Total number of T2V videos to generate prompts for.")
    parser.add_argument('--prompts_batch_size', type=int, default=5, help="Number of prompts to request from LLM in a single API call.")

    # --- Model & GPU Arguments ---
    parser.add_argument('--gpus', nargs='+', type=int, default=[0], help="List of GPU IDs to use for generation.")
    parser.add_argument('--LLM_model_name', type=str, default="gpt-4-turbo-preview")
    # Add other model name args if needed

    args = parser.parse_args()
    
    # --- Argument Validation and Path Setup ---
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

    Path(args.prompts_dir).mkdir(exist_ok=True)
    
    # Register signal handler
    signal.signal(signal.SIGINT, save_on_exit)
    
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

    if args.mode in ['generation-only', 'all']:
        if not prompts_dict:
             logging.error("Prompt dictionary is not available. Cannot proceed with generation.")
             sys.exit(1)
        media_generation_pipeline(args, prompts_dict)
    
    logging.info("Pipeline finished successfully.")


if __name__ == '__main__':
    main()