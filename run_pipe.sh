# 将会创建 prompts/prompts_1.json (如果不存在)
python pipeline.py --mode prompt-only --num_img_sets 20 --num_videos 20 --prompts_batch_size 10

# 必须指定一个存在的json文件
python pipeline.py --mode generation-only --prompts_file prompts/prompts_1.json --gpus 0 1

# 生成prompts，然后立即开始生成媒体文件
python pipeline.py --mode all --num_img_sets 10 --num_videos 10 --gpus 0