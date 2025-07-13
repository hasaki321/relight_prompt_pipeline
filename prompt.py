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
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key, base_url=os.getenv("BASE_URL"))

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
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
        background_prompt = Path(args.prompts_dir ,'background.txt').read_text()
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