# python pipeline.py --mode all --num_img_sets 20 --num_videos 10 --num_relight_vairant 3 --num_video_vairant 1 --prompts_batch_size 10 --gpus 1 2 3 --prompts_dir ./demo_prompts

# python pipeline.py --mode prompt-only --num_img_sets 20 --num_videos 10 --num_relight_vairant 3 --num_video_vairant 1 --prompts_batch_size 10 --gpus 1 --prompts_dir ./demo_prompts

# python pipeline.py --mode generation-only --gpus 1 2 3 --prompts_dir ./demo_prompts --prompts_file ./demo_prompts/prompts_1.json --batch_size 2


# accelerate launch --config_file ./accelerate_config.yaml pipeline.py --mode all --num_img_sets 20 --num_videos 10 --num_relight_vairant 3 --num_video_vairant 1 --prompts_batch_size 10 --gpus 1 2 3 --prompts_dir ./demo_prompts

# accelerate launch --config_file ./accelerate_config.yaml pipeline.py --mode prompt-only --num_img_sets 20 --num_videos 10 --num_relight_vairant 3 --num_video_vairant 1 --prompts_batch_size 10 --gpus 1 --prompts_dir ./demo_prompts

accelerate launch --config_file ./accelerate_config.yaml pipeline.py --mode generation-only --gpus 1 2 --prompts_dir ./demo_prompts --prompts_file ./demo_prompts/prompts_1.json --batch_size 2