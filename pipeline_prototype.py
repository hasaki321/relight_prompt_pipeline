
argparser:
mode: prompt only, generation only, all

#============
prompts_dir: './prompt'
"""
structured like:
./prompt
|-background.txt
|-editing.txt
|-video.txt
|-prompts_{i}.json
|-prompts_{i}
    |-editing
        |-set_{i}
            |-base_img.png
            |-relighting
            |   |-vairant_{i}.png
            |-video
                |-vairant_{i}.mp4
    |-video
        |-video_{i}.mp4
"""
prompts_file: None
"""prompts_file be like: prompts_1.json, number increase if one exist"""

if generation only: assert prompts_file is not None, 'prompts_file must be specify in generationg mode'
if prompt only or all: auto generate prompts_file name

#============
num_img_sets: 100
num_editing: 3
num_videos: 100
prompts_batch_size: 10 #request in batch to prevent long context
prompts_batch_size = min(num_img_sets, prompts_batch_size)
assert num_img_sets // prompts_batch_size == 0
assert num_videos // prompts_batch_size == 0

img_gen_batch_size: 3
img_editing_batch_size: 3
I2V_batch_size: 1
T2V_batch_size: 1

#============
LLM_model_name: "..."
LLM_api_token: "..."
image_generation_model_name: "black-forest-labs/FLUX.1-dev"
relighting_model_name: "timbrooks/instruct-pix2pix"
T2V_model_name: "Wan-AI/Wan2.1-T2V-14B-Diffusers"
I2V_model_name: "Wan-AI/Wan2.1-T2V-14B-Diffusers"

#============
gpus: [5,6,7,8] mode:+


json_parser:
json_path = prompts_path + prompts_file 
prompts = json parse(json_path)
"""
prompts structure:
    {
        "task"{
            "editing":[
                        {
                            "set_id": "1",
                            "description": "A brief, human-readable description of the scene, e.g., 'A ceramic vase by a window during different times of day.'",
                            "base_generation_prompt": "The detailed prompt to generate the initial image.",
                            "path": None
                            "relighting_prompts": [
                                {"prompt": "A prompt describing the first lighting variation.", "comma_prompt":"short prompts seperated by comma", "path": None},
                                {"prompt": "A prompt describing the second lighting variation.", "comma_prompt":"short prompts seperated by comma",  "path": None},
                                {"prompt": "A prompt describing the third lighting variation.", "comma_prompt":"short prompts seperated by comma", "path": None}
                            ],
                            "video_relighting_prompts": [
                                {"prompt": "A prompt describing the first video lighting variation.", "path": None},
                                {"prompt": "A prompt describing the second video lighting variation.", "path": None},
                                {"prompt": "A prompt describing the third video lighting variation.", "path": None}
                            ],
                        },
                        {
                            "set_id": "2",
                            ...
                        }
            ],
            "video_gen":[
                    {"id":1, "prompt": "A complete, detailed prompt for the first video.", "path": None},
                    {"id":2, "prompt": "A complete, detailed prompt for the second video.", "path": None},
                    {"id":3, "prompt": "A complete, detailed prompt for the third video.", "path": None},
                    ...
            ]
        }
    }
"""
return prased prompts in dict format



prompt_pipeline:
"""
we need three editable files as api prompt instruction
background.txt: give a detailed general background and requirements(including json format guide)

editing.txt: detailed text guide to tell the model to generate 1. prompt for de-novo img generation 2. image editing prompt for relighting 3. I2V video generation prompt 
video.txt: detailed text guide to tell the model to generate relightning videos from text

then add a instruct prompt follow by those prompts deatiling numbers of tasks described in args 
"""

background_prompt = (open(os.path.join(prompts_path, 'background.txt'))).read()
editing_prompt = (open(os.path.join(prompts_path, 'editing.txt'))).read()
video_prompt = (open(os.path.join(prompts_path, 'video.txt'))).read()
context_prompt = ""
instruct_prompt = ""

api_class = openai context

prompt_dict = {
        "task"{
            "editing":[],
            "video_gen":[]
        }
    }

# === editing task ===
editing_counter = 0
editing_system_prompt = background_prompt + editing_prompt 
for i in range(num_img_sets // args.prompts_batch_size):
    instruct_prompt = f'**Instruction:** Now, generate **{args.prompts_batch_size}** unique sets following the `editing` JSON structure above.'
    editing_json = json_check(api_class.send(editing_system_prompt, instruct_prompt))
    # error handle
    editing_dict = parse_json(editing_json)
    for i in range(len(editing_dict["editing"])):
        editing_dict["editing"][i]["set_id"] = f"set_{counter + i}"
    editing_counter += len(editing_dict["editing"])
    prompt_dict["task"]["editing"].extend(editing_dict)

# === video task ===
video_counter = 0
video_system_prompt = background_prompt + video_prompt
for i in range(num_videos // args.prompts_batch_size):
    instruct_prompt = f'**Instruction:** Now, generate **{args.prompts_batch_size}** unique prompts following the `video_gen` JSON structure above.'
    video_json = json_check(api_class.send(instruct_prompt))
    # error handle
    video_dict = parse_json(video_json)
    for i in range(len(video_dict["video_gen"])):
        video_dict["video_gen"][i]["id"] = f"video_{counter + i}"
    video_counter += len(video_dict["video_gen"])
    prompt_dict["task"]["video_gen"].extend(video_dict)

prompts_file.save(json(prompt_dict))
return prompt_dict


image_generation_pipeline:
"""
denovo image generation
input model_name, prompts_dict -> prompt list, 
return image list, save the images and fill the path in the prompts_dict
"""

relighting_pipeline:
"""
relighting image generation
input model_name, prompts_dict -> (image path , prompt/comma) list,
return image list, save the images and fill the path in the prompts_dict
"""

T2V_pipeline:
"""
T2V video generation
input model_name, prompts_dict -> prompt list, 
return video list, save the videos and fill the path in the prompts_dict
"""

I2V_pipeline:
"""
I2V video generation
input model_name, prompts_dict -> (image path , prompt) list 
return video list, save the videos and fill the path in the prompts_dict
"""

model_pipeline:
"""
Inputs:
prompt_dict
args:
    # call their model using hf
    image_generation_model_name: "black-forest-labs/FLUX.1-dev"
    relighting_model_name: "timbrooks/instruct-pix2pix"
    T2V_model_name: "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    I2V_model_name: "Wan-AI/Wan2.1-T2V-14B-Diffusers"

Finish the following task:
All the tasks should be allocated on multiple GPUS using mp or torch.mp or other libraries
1. generate image using base_generation_prompt and set its path to the generated path
2. relighting image using relighting_prompts/prompt and set its path to the generated path
3. (I2V) generate video using the de-novo image path and I2V prompt video_relighting_prompts/prompt and set its path to the generated path
4. generate video using video_relighting_prompts/prompt and set its path to the generated path
"""


main:
args = argparser()

if all or prompt only:
    prompt_dict = prompt_pipeline
else:
    prompt_dict = json_parser

if all or generation only:
model_pipeline(prompt_dict)