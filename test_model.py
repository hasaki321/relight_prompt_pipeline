from diffusers import DiffusionPipeline
import os
from huggingface_hub import login

os.environ['HF_HOME'] = '/home/herongshen/relighting/models'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
login(token="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")

# ============================

pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]

# from diffusers import DiffusionPipeline

# ============================

# pipe = DiffusionPipeline.from_pretrained("timbrooks/instruct-pix2pix")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipe(prompt).images[0]

# ============================

# import torch
# from diffusers.utils import export_to_video
# from diffusers import AutoencoderKLWan, WanPipeline
# from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

# # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
# model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
# vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
# flow_shift = 5.0 # 5.0 for 720P, 3.0 for 480P
# scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
# pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
# pipe.scheduler = scheduler
# pipe.to("cuda")

# prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
# negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# output = pipe(
#      prompt=prompt,
#      negative_prompt=negative_prompt,
#      height=720,
#      width=1280,
#      num_frames=81,
#      guidance_scale=5.0,
#     ).frames[0]
# export_to_video(output, "output.mp4", fps=16)
