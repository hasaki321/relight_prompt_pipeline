
import os
os.environ['HF_HOME'] = '/mnt/data/sysu/Users/wangzh/ID-Relight/models'

from huggingface_hub import model_info

import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

model_path = "lambdalabs/sd-pokemon-diffusers"

info = model_info(model_path)
model_base = info.cardData["base_model"]
print(model_base, type(info.cardData))  

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)  
pipe = pipe.to("cuda")

prompt = "Yoda"
scale = 10
n_samples = 4

# Sometimes the nsfw checker is confused by the Pokémon images, you can disable
# it at your own risk here
disable_safety = False

if disable_safety:
  def null_safety(images, **kwargs):
      return images, False
  pipe.safety_checker = null_safety

with autocast("cuda"):
  images = pipe(n_samples*[prompt], guidance_scale=scale).images

for idx, im in enumerate(images):
  im.save(f"{idx:06}.png")
