import os
import argparse
from time import time
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusion3Pipeline # for SD3
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler # for SD2.1
from diffusers import DiffusionPipeline # for SDXL


def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def concat_images(image_path_list, direction='horizontal'):
    # Open all images
    images = [Image.open(image_path) for image_path in image_path_list]

    # Determine the size of the final image
    if direction == 'horizontal':
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        final_image = Image.new('RGB', (total_width, max_height))
        
        current_x = 0
        for img in images:
            final_image.paste(img, (current_x, 0))
            current_x += img.width

    elif direction == 'vertical':
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        final_image = Image.new('RGB', (max_width, total_height))

        current_y = 0
        for img in images:
            final_image.paste(img, (0, current_y))
            current_y += img.height

    else:
        raise ValueError("Direction should be 'horizontal' or 'vertical'.")

    return final_image


def concat_images_with_resize(image_path_list, direction='horizontal'):
    # Open all images
    images = [Image.open(image_path) for image_path in image_path_list]
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)
    # Determine the size of the final image
    if direction == 'horizontal':
        total_width = len(images) * max_width
        final_image = Image.new('RGB', (total_width, max_height))
        
        current_x = 0
        for img in images:
            if img.width != max_width:
                img = img.resize((max_width, max_height))
            final_image.paste(img, (current_x, 0))
            current_x += img.width

    elif direction == 'vertical':
        total_height = len(images) * max_height
        final_image = Image.new('RGB', (max_width, total_height))

        current_y = 0
        for img in images:
            if img.height != max_height:
                img = img.resize((max_width, max_height))
            final_image.paste(img, (0, current_y))
            current_y += img.height

    else:
        raise ValueError("Direction should be 'horizontal' or 'vertical'.")

    return final_image


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
    args.add_argument("--seed", type=int, default=2024)
    
    seed_everything(args.seed)
    prompt = args.prompt
    
    # SD3
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    start_time = time()
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    
    start_time = time()
    image = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    end_time = time()
    print(f"SD3 Elapsed time: {end_time - start_time:.2f}s")

    image.save("sd3.png")

    del pipe


    # SD2.1
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    start_time = time()
    image = pipe(prompt).images[0]
    end_time = time()
    print(f"SD2.1 Elapsed time: {end_time - start_time:.2f}s")

    image.save("sd21.png")

    del pipe


    # SDXL
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    start_time = time()
    image = pipe(prompt=prompt).images[0]
    end_time = time()
    print(f"SDXL Elapsed time: {end_time - start_time:.2f}s")

    image.save("sdxl.png")

    del pipe

    # Concat images
    img_path_list = ["sd21.png", "sdxl.png", "sd3.png"]
    concat_image = concat_images_with_resize(img_path_list, direction="horizontal")
    prompt_full = "_".join(prompt.replace(",", " ").split(" "))
    concat_image.save(f"{prompt_full}.png")
    
    