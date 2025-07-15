#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# os.environ['HF_HOME'] 必须 在导入 huggingface 相关包之前设置
import os
os.environ['HF_HOME'] = '/mnt/data/sysu/Users/wangzh/ID-Relight/models'
# os.environ["HF_TOKEN"] = "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"

import sys
import logging
# This is safe at the global level as it only configures logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
from huggingface_hub import HfFolder, whoami, login
def hf_login(token: str, hf_home: str = None):
    if hf_home:
        os.environ["HF_HOME"] = hf_home

    # 检查是否已有登录信息
    stored_token = HfFolder.get_token()
    if stored_token is None:
        logging.info("🔑 未检测到已保存的 HuggingFace Token，正在登录...")
        login(token)
    else:
        try:
            user_info = whoami()
            logging.info(f"✅ 已登录 HuggingFace 用户: {user_info['name']}")
        except Exception:
            logging.info("⚠️ 存在过期或无效 Token，尝试重新登录...")
            login(token)

token = "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"  
hf_home = "/mnt/data/sysu/Users/wangzh/ID-Relight/models"
hf_login(token=token, hf_home=hf_home)

import argparse
import copy
import logging
import math
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxControlPipeline, FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import check_min_version, is_wandb_available, load_image, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from diffusers import DiffusionPipeline

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.35.0.dev0")

logger = get_logger(__name__)

NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)

if __name__ == '__main__':
    from controlnet_aux import OpenposeDetector
    from diffusers import FluxControlPipeline

    open_pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

    # prepare pose condition.
    url = "https://huggingface.co/Adapter/t2iadapter/resolve/main/people.jpg"
    image = load_image(url)
    image = open_pose(image, detect_resolution=512, image_resolution=1024)
    image = np.array(image)[:, :, ::-1]           
    image = Image.fromarray(np.uint8(image))
    print(type(image))
    image.save("pose_image.png")
    prompt = "A couple, 4k photo, highly detailed"

    # prepare flux-control pipeline.
    base_model = "black-forest-labs/FLUX.1-dev"
    pipe = FluxControlPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

    lora_repo = "/home/wangzhen/id-light-v2/lora/pose-control-lora"
    pipe.load_lora_weights(lora_repo)

    device = torch.device("cuda")
    pipe.to(device)

    # generating image.
    gen_images = pipe(
    prompt=prompt,
    control_image=image,
    num_inference_steps=50,
    joint_attention_kwargs={"scale": 0.9},
    guidance_scale=25., 
    ).images[0]
    gen_images.save("output.png")