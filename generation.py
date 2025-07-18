import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
from PIL import Image

from transformers import AutoModelForImageSegmentation

from copy import deepcopy
from torchvision.transforms import ToPILImage, ToTensor

from model_utils import load_model, unload_model
from generation_core import *



# --- The Worker Process ---
def worker(task_queue, shared_data_proxy, data_lock, args, device):
    """A worker process that runs on a single GPU."""
    process_name = mp.current_process().name
    logging.info(f"[{process_name}] assigned to {device}.")

    current_model = None
    current_model_name = None

    while True:
        try:
            task_batch = task_queue.get()
            if task_batch is None:
                logging.info(f"[{process_name}] Received sentinel. Shutting down.")
                break
            task_type = task_batch[0]['type']
            model_name = task_batch[0]['model_name']
            batch_id = task_batch[0].get('batch_id', 'N/A')
            task_ids = [task.get('task_id', 'N/A') for task in task_batch]
            logging.info(f"[{process_name}] Processing batch_id={batch_id}, task_ids={task_ids}, type={task_type}")

            if model_name != current_model_name:
                unload_model(current_model)
                current_model = load_model(model_name, device)
                current_model_name = model_name

            if task_type == 'generate_image':
                batch_generate_images(task_batch, current_model, device)
            elif task_type == 'relight_image':
                batch_relight_images(task_batch, current_model, device)
            elif task_type == 'i2v':
                batch_relight_videos(task_batch, current_model, device)
            elif task_type == 't2v':
                batch_generate_videos(task_batch, current_model, device)
            elif task_type == 'iclight': 
                batch_iclight_images(task_batch, current_model, device)
            elif task_type == 'lbm': 
                batch_lbm_images(task_batch, current_model, device)

            with data_lock:
                # To modify the shared dict proxy, we need to get it, modify it, and set it back
                # This is a nuance of multiprocessing.Manager.dict
                current_data = dict(shared_data_proxy)
                for task in task_batch:
                    if task['type'] == 'generate_image':
                        current_data['editing_sets'][task['set_idx']]['path'] = task['output_path']
                    elif task['type'] == 'relight_image':
                        current_data['editing_sets'][task['set_idx']]['relighting_prompts'][task['relight_idx']]['path'] = task['output_path']
                    elif task['type'] == 'i2v':
                        current_data['editing_sets'][task['set_idx']]['video_relighting_prompts'][task['video_idx']]['path'] = task['output_path']
                    elif task['type'] == 't2v':
                        current_data['t2v_prompts'][task['video_idx']]['path'] = task['output_path']
                    elif task['type'] == 'iclight':
                        current_data['editing_sets'][task['set_idx']]['relighting_prompts'][task['relight_idx']]['iclight_path'] = task['output_path']
                    elif task['type'] == 'lbm':
                        current_data['editing_sets'][task['set_idx']]['relighting_prompts'][task['relight_idx']]['lbm_path'] = task['output_path']

                # Update the shared proxy
                shared_data_proxy.update(current_data)

                # Save progress to disk after each batch
                with open(args.prompts_file, 'w', encoding='utf-8') as f:
                    json.dump(current_data, f, indent=4, ensure_ascii=False)
            
            logging.info(f"[{process_name}] Completed batch of {len(task_batch)} {task_type} tasks. Progress saved.")
            logging.info(f"[{process_name}] Completed batch_id={batch_id}, type={task_type}, task_ids={task_ids}")

        except queue.Empty:
            # This should not happen with a blocking get(), but is good practice
            logging.info(f"[{process_name}] Queue empty. This should not be reached.")
            continue
        except Exception:
            logging.error(f"[{process_name}] Unhandled exception in worker loop:")
            logging.error(traceback.format_exc())
            break
            
    unload_model(current_model)
    logging.info(f"[{process_name}] Worker finished.")

# --- The Main Media Generation Pipeline ---
def advanced_media_generation_pipeline(args, prompts_dict):
    """Manages the media generation process using a multi-process worker pool."""
    # The Manager should be created here, inside the function that uses it.
    manager = Manager()
    shared_data = manager.dict(prompts_dict)
    data_lock = manager.Lock()
    task_queue = manager.Queue()

    all_tasks = defaultdict(list)
    base_output_dir = Path(args.prompts_file).stem

    # Task 1: Base Image Generation
    # for i, item in enumerate(shared_data["editing_sets"]):
    #     if not item.get("path") or not Path(item.get("path")).exists():
    #         output_dir = Path(args.prompts_dir) / base_output_dir / "editing" / item["set_id"]
    #         output_dir.mkdir(parents=True, exist_ok=True)
    #         all_tasks[args.image_generation_model_name].append({
    #             'type': 'generate_image', 'model_name': args.image_generation_model_name,
    #             'set_idx': i, 'prompt': item["base_generation_prompt"],
    #             'output_path': str(output_dir / "base_img.png")})

    # Task 2: Religting images with IC-Light and LBM
    # for i, item in enumerate(shared_data["editing_sets"]):
    #     base_path = item.get("path")
    #     if not base_path or not Path(base_path).exists():
    #         continue
    #     # Relighting
    #     relight_dir = Path(base_path).parent / "relighting"
    #     relight_dir.mkdir(exist_ok=True)
    #     for j, p in enumerate(item["relighting_prompts"]):
    #         if not p.get("iclight_path") or not Path(p.get("iclight_path")).exists():
    #             all_tasks[args.iclight_model].append({
    #                 'type': 'iclight', 'model_name': args.iclight_model,
    #                 'set_idx': i,
    #                 'relight_idx': j, 
    #                 'prompt': p["prompt"],
    #                 'background_prompt': p["background_prompt"],
    #                 'lighting_prompt': p['lighting_prompt'],
    #                 'base_image_path': base_path,
    #                 'output_path': str(relight_dir / f"iclight_{j}.png")})
                
    #         if not p.get("lbm_path") or not Path(p.get("lbm_path")).exists():
    #             all_tasks[args.lbm_model].append({
    #                 'type': 'lbm', 'model_name': args.lbm_model,
    #                 'set_idx': i,
    #                 'relight_idx': j,  
    #                 'prompt': p["prompt"],
    #                 'background_prompt': p["background_prompt"],
    #                 'lighting_prompt': p['lighting_prompt'],
    #                 'base_image_path': base_path,
    #                 'output_path': str(relight_dir / f"lbm_{j}.png")})

    # Task 2, 3: Relighting and I2V
    # for i, item in enumerate(shared_data["editing_sets"]):
    #     base_path = item.get("path")
    #     if not base_path or not Path(base_path).exists():
    #         continue
    #     # Relighting
    #     relight_dir = Path(base_path).parent / "relighting"
    #     relight_dir.mkdir(exist_ok=True)
    #     for j, p in enumerate(item["relighting_prompts"]):
    #         if not p.get("path") or not Path(p.get("path")).exists():
    #              all_tasks[args.relighting_model_name].append({
    #                 'type': 'relight_image', 'model_name': args.relighting_model_name,
    #                 'set_idx': i, 'relight_idx': j, 'prompt': p["prompt"],
    #                 'base_image_path': base_path, 'output_path': str(relight_dir / f"variant_{j}.png")})

    for i, item in enumerate(shared_data["editing_sets"]):
        base_path = item.get("path")
        if not base_path or not Path(base_path).exists():
            continue

        # I2V
        video_dir = Path(base_path).parent / "video"
        video_dir.mkdir(exist_ok=True)
        for j, p in enumerate(item["video_relighting_prompts"]):
            if not p.get("path") or not Path(p.get("path")).exists():
                all_tasks[args.I2V_model_name].append({
                    'type': 'i2v', 'model_name': args.I2V_model_name,
                    'set_idx': i, 'video_idx': j, 'prompt': p["prompt"],
                    'base_image_path': base_path, 'output_path': str(video_dir / f"variant_i2v_{j}.mp4")})
                 
    # Task 4: T2V Video Generation
    # for i, item in enumerate(shared_data["t2v_prompts"]):
    #     if not item.get("path") or not Path(item.get("path")).exists():
    #         output_dir = Path(args.prompts_dir) / base_output_dir / "video"
    #         output_dir.mkdir(parents=True, exist_ok=True)
    #         all_tasks[args.T2V_model_name].append({
    #             'type': 't2v', 'model_name': args.T2V_model_name,
    #             'video_idx': i, 'prompt': item["prompt"],
    #             'output_path': str(output_dir / f"{item['id']}.mp4")})

    total_batches = 0
    task_id = 0
    for model_name, tasks in all_tasks.items():
        batch_size = args.batch_size
        if ("Wan" in model_name) or ("Vace" in model_name): batch_size = 1 #math.ceil(batch_size / 3)
        logging.info(f"Creating {len(tasks)} tasks for model {model_name} with batch size {batch_size}")
        for i in range(0, len(tasks), batch_size):
            task_batch = tasks[i:i + batch_size]

            for task in task_batch:
                task['task_id'] = task_id
                task['batch_id'] = total_batches
                task_id += 1

            task_queue.put(task_batch)
            total_batches += 1
    
    if total_batches == 0:
        logging.info("No new tasks to process. Everything is up to date.")
        return

    logging.info(f"Populated queue with {total_batches} batches of tasks.")

    ## Adding sentinel values to signal workers to exit
    for _ in range(len(args.gpus)):
        task_queue.put(None)

    processes = []

    for i in range(len(args.gpus)):
        p = mp.Process(target=worker, name=f"Worker-GPU-{args.gpus[i]}", args=(task_queue, shared_data, data_lock, args, f"cuda:{args.gpus[i]}"))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    logging.info("All worker processes have finished.")
