import json
import os
import PIL
import cv2
import math
import numpy as np
import torch
import torchvision
import imageio
from pandas import read_csv
from einops import rearrange
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def save_videos_grid(videos, path=None, rescale=True, n_rows=4, fps=8, discardN=0):
    videos = rearrange(videos, "b c t h w -> t b c h w").cpu()
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x / 2.0 + 0.5).clamp(0, 1)  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        #x = adjust_gamma(x, 0.5)
        outputs.append(x)

    outputs = outputs[discardN:]

    if path is not None:
        #os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, outputs, duration=1000/fps, loop=0)

    return outputs

def convert_image_to_fn(img_type, minsize, image, eps=0.02):
    width, height = image.size
    if min(width, height) < minsize:
        scale = minsize/min(width, height) + eps
        image = image.resize((math.ceil(width*scale), math.ceil(height*scale)))

    if image.mode != img_type:
        return image.convert(img_type)
    return image

def csv_to_dataset_format(csv_file):
    csv_data = read_csv(csv_file)
    caption = csv_data.columns.values
    output_data = {}
    print("*"*20,"dataset information","*"*20)
    for item in caption:
        output_data[item] = csv_data[item].to_list()
        print(f"Row {item} contains {len(output_data[item])} items")
    print("*"*20,"Finish load dataset dictionary","*"*20)
    return output_data

def meta_to_dataset_format(meta_file,root_path):
    with open(meta_file,"r") as f:
        meta_info = json.load(f)
    output_data = {}
    caption = ["input_image","edited_image","edit_prompt"]
    output_data['input_image'] = []
    output_data['edited_image'] = []
    output_data['edit_prompt'] = []
    output_data['mask'] = []
    print("*" * 20, "dataset information", "*" * 20)
    for item in meta_info:
        output_data["input_image"].append(os.path.join(root_path,item["gt"]))
        output_data["edited_image"].append(os.path.join(root_path,item["inpaint"]))
        output_data["edit_prompt"].append(item["text"]['label']['name'])
        output_data["mask"].append(os.path.join(root_path,item["mask"]))
    length_cases = len(output_data["input_image"])
    print("*"*20,f"Finish load dataset dictionary, total {length_cases} train samples","*"*20)
    return output_data

def meta_to_inpaint_dataset_format(meta_file,root_path):
    output_data = {}
    caption = ["input_image","edited_image","edit_prompt"]
    output_data['input_image'] = []
    output_data['edited_image'] = []
    output_data['edit_prompt'] = []
    output_data['mask'] = []
    print("*" * 20, "dataset information", "*" * 20)
    if not isinstance(meta_file,list):
        print(f"Load from {meta_file}")
        with open(meta_file,"r") as f:
            meta_info = json.load(f)
        for item in meta_info:
            output_data["input_image"].append(os.path.join(root_path,item["source"]))
            output_data["edited_image"].append(os.path.join(root_path,item["GT"]))
            output_data["edit_prompt"].append("")
            output_data["mask"].append(os.path.join(root_path,item["mask"]))
    else:
        for meta_path in meta_file:
            root_path = os.path.dirname(meta_path)
            print(f"Load from {meta_path}")
            with open(meta_path, "r") as f:
                meta_info = json.load(f)
            for item in meta_info:
                output_data["input_image"].append(os.path.join(root_path, item["source"]))
                output_data["edited_image"].append(os.path.join(root_path, item["GT"]))
                output_data["edit_prompt"].append("")
                output_data["mask"].append(os.path.join(root_path, item["mask"]))
    length_cases = len(output_data["input_image"])
    print("*"*20,f"Finish load dataset dictionary, total {length_cases} train samples","*"*20)
    return output_data

def im_crop_center(img, crop_size=512):
    img_width, img_height = img.size
    left, right = (img_width - crop_size) / 2, (img_width + crop_size) / 2
    top, bottom = (img_height - crop_size) / 2, (img_height + crop_size) / 2
    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(img_width - 0, right)), round(min(img_height - 0, bottom))
    return img.crop((left, top, right, bottom))

def reshape_PIL(pil_image,target_size=512):
    original_h,original_w = pil_image.size
    # if original_h >= target_size and original_w >= target_size:
    #     return pil_image
    if original_h > original_w:
        new_w = target_size
        rate = new_w / original_w
        new_h = int(original_h * rate)
    else:
        new_h = target_size
        rate = new_h / original_h
        new_w = int(original_w * rate)
    new_pil = pil_image.resize((new_h,new_w))
    new_pil = im_crop_center(new_pil)
    return new_pil


