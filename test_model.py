from diffusers import DiffusionPipeline
import os
from huggingface_hub import login
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

os.environ['HF_HOME'] = '/home/herongshen/relighting/models'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
login(token="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")

model_id = "Wan-AI/Wan2.1-VACE-1.3B"
# pipe = DiffusionPipeline.from_pretrained("Wan-AI/Wan2.1-VACE-1.3B")

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-VACE-1.3B", torch_dtype=torch.bfloat16)
# pipe.to("cuda")

#============================

# model_id = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
# image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
# vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
# model = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
# model.to(device)
# model.to()

# ============================

# pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")

# prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
# image = pipe(prompt).images[0]
# pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16).to("cuda")

# ============================

# import PIL
# import torch
# from diffusers import FluxControlPipeline, FluxTransformer2DModel
# from diffusers.utils import load_image
# from image_gen_aux import DepthPreprocessor

# pipe = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda:6")
# pipe.load_lora_weights("black-forest-labs/FLUX.1-Depth-dev-lora", adapter_name="depth")
# pipe.set_adapters("depth", 0.85)

# prompt = "A robot made of exotic candies and chocolates of different kinds. The background is filled with confetti and celebratory gifts."
# control_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/robot.png")

# processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
# control_image = processor(control_image)[0].convert("RGB")

# image = pipe(
#     prompt=prompt,
#     control_image=control_image,
#     height=1024,
#     width=1024,
#     num_inference_steps=30,
#     guidance_scale=10.0,
#     generator=torch.Generator().manual_seed(42),
# ).images[0]
# image.save("output.png")

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
