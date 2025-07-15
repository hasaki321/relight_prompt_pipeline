import os
os.environ['HF_HOME'] = '/mnt/data/sysu/Users/wangzh/ID-Relight/models'

import torch
from diffusers import DiffusionPipeline

base_model = "black-forest-labs/FLUX.1-dev"
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

# repo from https://huggingface.co/prithivMLmods/Shadow-Projection-Flux-LoRA
lora_repo = "prithivMLmods/Shadow-Projection-Flux-LoRA"
trigger_word = "Shadow Projection"  
pipe.load_lora_weights(lora_repo)

device = torch.device("cuda")
pipe.to(device)

prompt = f"A portrait of a woman, {trigger_word}, cinematic lighting"
negative_prompt = "low quality, blurry, distorted face, watermark"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30, 
    guidance_scale=7.5,      
).images[0]

image.save("flux_shadow_projection.png")