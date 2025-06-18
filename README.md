## 流程

1. 如果是选定的图像，则喂给多模态打标大模型让其输出描述dsec；否则直接用以下规则生成图片描述dsec。【文本-图像对】
2. 造不同光照/环境/视角下的图像:
    1. 通过文本-图像编辑模型直接给定图像与文本进行编辑（利用以下造prompt规则生成编辑prompt） **[TODO: 寻找更多开源实现]**  【新 文本-图像 / 图像对】
    2. 通过重光照模型给定其他信息进行编辑 （HDR图像，球谐函数，新环境等） **[TODO: 寻找更多开源实现]**  【新 环境-图像 / 图像对】
    3. 如果我们仍然需要文本图像对，我们可以继续让模型对新图片打标
    4. 如果我们需要更细致的信息，我们可以通过分解模型 **[TODO: 寻找可靠分解模型实现]** 将 a,b 中已经打好光的模型分解新造出来的图像 → 本征态和其他信息？
3. 造文本视频对：
    1. 通过给定的图像或者生成的图像，利用其描述desc丢给大模型让其生成视频变化prompt
    2. 直接通过prompt生成 （T2V， 此方案可用模型较少，因此我们一般考虑下面的方案）
    3. 通过给定图像结合AI给出的prompt生成视频 （I2V， 此方案可用模型更多，更实惠）**[TODO: 寻找更多开源实现]**
    4. 已经造好的视频继续分解？不太清楚流程

---
## 运行

### prompt
提供了prompts示例，详见 `example_prompts`, 完整的造数据prompt包含
- 系统级 背景提示： background.txt
- 图像编辑&I2V生成： editing_task.txt
- T2V生成： video_task.txt
- 任务数量要求： 要求生成任务数量指引（位于代码中，可通过参数调节）

### 运行
分为获取prompt的pipeline `pipeline_prompt.py` 以及通过prompt执行推理的pipeline `pipeline_async.py`

pipeline_prompt:
- 目前使用ds-v3的apikey，可以替换成gpt或者gemini等更优模型
- prompts_batch_size 控制一次请求要求生成的prompt集合数
- num_img_sets： 控制图像生成prompt集合数量
    - num_relight_vairant： 控制图像编辑重打光prompt数量
    - num_video_vairant： 控制I2V prompt数量
- num_videos： 控制生成T2V prompt数量

pipeline_async:
- batch_size： 图像生成bs， 视频为 ceil(batch_size / 5)
- prompts_dir： 包含prompt txt的目录
- prompts_file： 先前生成的prompt json文件
- gpus： 使用的gpu id

---
## 模型

### I2V
- [x] Van2.1-14B-720p: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers (81frames/ 30min)
- [x] Van2.1-14B-480p: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers (81frames/ 16min)

### T2V
- [x] Van2.1-14B-720p: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-720P-Diffusers (81frames/ 30min)
- [x] Van2.1-14B-480p: https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-480P-Diffusers (81frames/ 16min)
- [] Vace-1.3B： https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers
- [] Vace-14B: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers

### 通过prompt生成图像
- [x] FLUX: https://huggingface.co/black-forest-labs/FLUX.1-dev

### 通过 图像+prompt 重打光
- [x] FLUX-depth-lora: https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev-lora 
- [x] FLUX-depth: https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev
- [x] Instruct-pix2pix: https://huggingface.co/timbrooks/instruct-pix2pix 