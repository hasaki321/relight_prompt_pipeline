安装conda环境
```sh
conda create --name lora python=3.10
conda activate lora
pip install -r requirements.txt
```

sd lora 启动命令：[🤗sdlora](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
```bash
python  train_sd_lora.py --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --push_to_hub \
  --report_to=wandb \
  --checkpointing_steps=500 \
  --validation_prompt="Totoro" \
  --seed=1337

```

flux control lora 启动命令：[🤗flux control lora](https://github.com/huggingface/diffusers/tree/main/examples/flux-control)
```bash
accelerate launch --main_process_port 25901 train_control_lora_flux.py \
  --dataset_name="raulc0399/open_pose_controlnet" \
  --output_dir="pose-control-lora" \
  --mixed_precision="bf16" \
  --train_batch_size=1 \
  --rank=64 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --offload \
  --seed="0" \
  --push_to_hub
```