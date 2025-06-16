# 将会创建 prompts/prompts_1.json (如果不存在)
python pipeline.py --mode prompt-only --num_img_sets 50 --num_videos 25 --num_relight_vairant 3 --num_video_vairant 1 --prompts_batch_size 25

# 生成prompts，然后立即开始生成媒体文件
python pipeline_async.py --mode generation-only --prompts_file prompt/prompts_3.json --gpus 6 7