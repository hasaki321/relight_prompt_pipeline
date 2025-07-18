import torch
import logging

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video, load_image
from diffusers import AutoencoderKLWan, WanPipeline, WanImageToVideoPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from transformers import CLIPVisionModel
from image_gen_aux import DepthPreprocessor


def load_model(model_name, device):
    """Loads a model onto a specific device."""
    logging.info(f"[{device}] Loading model: {model_name}...")
    if "FLUX.1-Depth-dev-lora" in model_name:
        model = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
        model.load_lora_weights("black-forest-labs/FLUX.1-Depth-dev-lora", adapter_name="depth")
        model.set_adapters("depth", 0.85)
        model.to(device)
        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        model = (model, processor)
    elif "FLUX.1-Depth-dev" in model_name:
        model = FluxControlPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16).to("cuda")
        model.to(device)
        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
        model = (model, processor)
    elif "FLUX" in model_name:
        model = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        model.to(device)
    elif "instruct-pix2pix" in model_name:
        model = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch.bfloat16, safety_checker=None)
        model.to(device)
        model.scheduler = EulerAncestralDiscreteScheduler.from_config(model.scheduler.config)
    elif "Wan2.1-T2V" in model_name:
        model_id = model_name
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", load_in_4bit=True)
        flow_shift = 3.0 # 5.0 for 720P, 3.0 for 480P
        scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
        model = WanPipeline.from_pretrained(model_id, vae=vae, load_in_4bit=True)
        model.scheduler = scheduler
        model.enable_model_cpu_offload(gpu_id=int(device.split(':')[-1]))
        # model.enable_xformers_memory_efficient_attention()
        # model.to(device)
    elif "Wan2.1-I2V" in model_name:
        model_id = model_name
        image_encoder = CLIPVisionModel.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=torch.float32)
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        model = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, image_encoder=image_encoder, torch_dtype=torch.bfloat16)
        model.enable_model_cpu_offload(gpu_id=int(device.split(':')[-1]))
        # model.to(device)
    ## import lbm, Modified by wangzh
    elif "lbm" in model_name or 'jasperai/LBM_relighting' in model_name:
        from lbm.inference import evaluate, get_model, extract_object, resize_and_center_crop 
        
        sys.path.insert(0, str(MODEL_ROOT / 'LBM' / 'src'))
        lbm_model = get_model(
                "jasperai/LBM_relighting",
                torch_dtype=torch.bfloat16,
                )
        lbm_model.to(device)

        # pipe to generate background
        lbm_t2i_pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        lbm_t2i_pipe.to(device)
        
        ## loading birefnet
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        birefnet.to(device)

        logging.info(f"Loading lbm model, flux-dev and birefnet successfully...")
        model = (lbm_t2i_pipe, birefnet)

    ## import IC-Light, Modified by wangzh
    elif "iclight" in model_name or 'lllyasviel/IC-Light' in model_name:
        sys.path.insert(0, str(MODEL_ROOT / 'IC-Light'))
        from gradio_demo import process_relight ## process_relight is a function defined in gradio_demo.py
        model = process_relight
    else:
        time.sleep(1)  # Simulate model loading time
        logging.info(f"[{device}] Model {model_name} loaded.")
        return {"name": model_name, "device": device} # Dummy object
    return model

def unload_model(model):
    """Unloads a model to free up GPU memory."""
    if model:
        logging.info(f"Unloading model: {model}...")
        del model
        torch.cuda.empty_cache()
        logging.info(f"[Memory cleared.]")