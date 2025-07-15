import os
os.environ['HF_HOME'] = '/mnt/data/sysu/Users/wangzh/ID-Relight/models'

from diffusers import StableDiffusionPipeline
import torch

model_path = 'stablediffusionapi/realistic-vision-v51'

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    safety_checker=None
).to("cuda")

# downloaded from https://sevenstyles.com/p/shadow-light-lora-for-stable-diffusion-1-5-7037689/?srsltid=AfmBOoqUotIYdw-TWwqPdEITIQaJpeW4lGdAX4a4w0fGZpDSCWDrCrkh
lora_path = '/mnt/data/sysu/Users/wangzh/output/sddata/lora/light_shadow/model.safetensors'
pipe.load_lora_weights(lora_path)

# inference
prompt = "a portrait of a man with rim light, sunlight coming from window, creating a shadow on the face" # "a portrait of a woman with dramatic rim light, cinematic lighting"
image = pipe(prompt).images[0]
image.save("rimlight_diffusers.png")


