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
    # --- Example implementation with OpenAI (replace with your own) ---
    # client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    # response = client.chat.completions.create(
    #     model="gpt-4-turbo-preview", # or your preferred model
    #     response_format={"type": "json_object"},
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt}
    #     ]
    # )
    # return response.choices[0].message.content
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

    # -----------------------------------------------------------------
    # For testing without an API key, return a dummy JSON string
    # logging.warning("Using dummy LLM response. Implement `call_llm_api` for real use.")
    # if "editing_sets" in user_prompt:
    #     return json.dumps({
    #         "editing_sets": [{
    #             "set_id": "set_001", "description": "A vintage brass telescope on a wooden tripod, catching the light.",
    #             "base_generation_prompt": "ultra-realistic product shot of a vintage brass telescope on a mahogany tripod, centered, filling the frame, studio lighting, 8k, sharp focus.",
    #             "path": None,
    #             "relighting_prompts": [
    #                 {"prompt": "The telescope is now illuminated by the soft, golden light of a sunrise streaming through a window.", "path": None},
    #                 {"prompt": "The telescope is now lit by the harsh, direct beam of a single spotlight from above, creating deep shadows.", "path": None}
    #             ],
    #             "video_relighting_prompts": [
    #                 {"prompt": "A time-lapse of the telescope as the light changes from cool dawn to warm midday sun, with shadows shortening.", "path": None}
    #             ]
    #         }]
    #     })
    # else:
    #     return json.dumps({
    #         "t2v_prompts": [{
    #             "id": "t2v_001",
    #             "prompt": "cinematic time-lapse video of a glossy red apple on a table. The light from a window moves across it, causing the highlight to glide over its surface from morning to evening. Static camera.",
    #             "path": None
    #         }]
    #     })

def generate_image(prompt, output_path, model, device):
    """Placeholder for generating a single image."""
    logging.info(f"Generating image for: {prompt[:80]}...")
    # image = model(prompt=prompt, ...).images[0] # Your model call
    # image.save(output_path)
    # For testing, create a dummy file
    # Path(output_path).touch()
    model.to(device)
    image = model(prompt).images[0]
    image.save(output_path)
    return output_path

def relight_image(base_image_path, prompt, output_path, model, device):
    """Placeholder for relighting an image."""
    logging.info(f"Relighting '{base_image_path}' with: {prompt[:80]}...")
    # image = Image.open(base_image_path)
    # relit_image = model(prompt=prompt, image=image, ...).images[0]
    # relit_image.save(output_path)
    # For testing, create a dummy file
    Path(output_path).touch()
    return output_path

def generate_video(prompt, output_path, model, device, base_image_path=None):
    """Placeholder for T2V or I2V video generation."""
    task_type = "I2V" if base_image_path else "T2V"
    logging.info(f"Generating {task_type} video for: {prompt[:80]}...")
    # if task_type == 'I2V':
    #     image = Image.open(base_image_path)
    #     video_frames = model(prompt=prompt, image=image, ...).frames
    # else: # T2V
    #     video_frames = model(prompt=prompt, ...).frames
    # export_to_video(video_frames, output_path) # You need a utility for this
    # For testing, create a dummy file
    Path(output_path).touch()
    return output_path

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
                response_data["editing_sets"][i]["set_id"] = f"set_{counter + i}"
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
                response_data["t2v_prompts"][i]["id"] = f"video_{counter + i}"
            video_counter += len(response_data["t2v_prompts"])
            PROMPT_DATA["t2v_prompts"].extend(response_data.get("t2v_prompts", []))
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"Failed to parse LLM response for video prompts (Batch {i+1}): {e}")
            continue
    logging.info(f"Generated {video_counter} video prompts at all, required number: {args.num_videos}")

    save_progress()
    logging.info(f"Prompt generation complete. Saved to {JSON_SAVE_PATH}")
    return PROMPT_DATA


def media_generation_pipeline(args, prompts_dict):
    """Generates images and videos based on the prompt dictionary."""
    global PROMPT_DATA, JSON_SAVE_PATH
    PROMPT_DATA = prompts_dict
    JSON_SAVE_PATH = args.prompts_file

    # --- Model Loading (placeholders) ---
    # This is where you would load your models onto the specified GPUs.
    # We are using placeholders here.
    # For a real multi-gpu setup, you might create a pool of workers,
    # each with a model on a specific GPU.
    logging.info("Loading generation models (placeholder)...")
    # image_gen_model = AutoPipelineForText2Image.from_pretrained(args.image_generation_model_name, torch_dtype=torch.float16, variant="fp16")
    # relight_model = InstructPix2PixPipeline.from_pretrained(args.relighting_model_name, torch_dtype=torch.float16)
    # t2v_model = DiffusionPipeline.from_pretrained(args.T2V_model_name, torch_dtype=torch.float16)
    # i2v_model = DiffusionPipeline.from_pretrained(args.I2V_model_name, torch_dtype=torch.float16)
    image_gen_model = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
    models = { "image": image_gen_model, "relight": None, "t2v": None, "i2v": None }
    
    # Simple round-robin GPU assignment
    gpu_cycler = cycle([f"cuda:{g}" for g in args.gpus])

    # --- Task Execution ---
    base_output_dir = Path(args.prompts_file).stem

    # 1. Generate Base Images
    logging.info("--- Starting Task 1: Base Image Generation ---")
    for item in tqdm(PROMPT_DATA["editing_sets"], desc="Base Images"):
        output_dir = Path(args.prompts_dir) / base_output_dir / "editing" / item["set_id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        path = item.get("path")
        if path and Path(path).exists():
            logging.info(f"Skipping existing base image: {path}")
            continue

        output_path = output_dir / "base_img.png"
        device = next(gpu_cycler)
        generate_image(item["base_generation_prompt"], output_path, models["image"], device)
        item["path"] = str(output_path)
        save_progress() # Save after each successful generation

    # 2. Generate Relighted Images
    logging.info("--- Starting Task 2: Relighted Image Generation ---")
    for item in tqdm(PROMPT_DATA["editing_sets"], desc="Relighting"):
        base_img_path = item.get("path")
        if not base_img_path:
            logging.warning(f"Skipping relighting for {item['set_id']} as base image is missing.")
            continue
        
        output_dir = Path(base_img_path).parent / "relighting"
        output_dir.mkdir(exist_ok=True)
        
        for i, relight_prompt in enumerate(item["relighting_prompts"]):
            path = relight_prompt.get("path")
            if path and Path(path).exists():
                logging.info(f"Skipping existing relighted image: {path}")
                continue

            output_path = output_dir / f"variant_{i}.png"
            device = next(gpu_cycler)
            relight_image(base_img_path, relight_prompt["prompt"], output_path, models["relight"], device)
            relight_prompt["path"] = str(output_path)
            save_progress()

    # 3. Generate I2V Videos
    logging.info("--- Starting Task 3: Image-to-Video Generation ---")
    for item in tqdm(PROMPT_DATA["editing_sets"], desc="I2V Videos"):
        base_img_path = item.get("path")
        if not base_img_path:
            logging.warning(f"Skipping I2V for {item['set_id']} as base image is missing.")
            continue

        output_dir = Path(base_img_path).parent / "video"
        output_dir.mkdir(exist_ok=True)

        for i, video_prompt in enumerate(item["video_relighting_prompts"]):
            path = video_prompt.get("path")
            if path and Path(path).exists():
                logging.info(f"Skipping existing I2V video: {path}")
                continue
            
            output_path = output_dir / f"variant_{i}.mp4"
            device = next(gpu_cycler)
            generate_video(video_prompt["prompt"], output_path, models["i2v"], device, base_image_path=base_img_path)
            video_prompt["path"] = str(output_path)
            save_progress()

    # 4. Generate T2V Videos
    logging.info("--- Starting Task 4: Text-to-Video Generation ---")
    for item in tqdm(PROMPT_DATA["t2v_prompts"], desc="T2V Videos"):
        output_dir = Path(args.prompts_dir) / base_output_dir / "video"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        path = item.get("path")
        if path and Path(path).exists():
            logging.info(f"Skipping existing T2V video: {path}")
            continue

        output_path = output_dir / f"{item['id']}.mp4"
        device = next(gpu_cycler)
        generate_video(item["prompt"], output_path, models["t2v"], device)
        item["path"] = str(output_path)
        save_progress()
        
    logging.info("All media generation tasks completed.")


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