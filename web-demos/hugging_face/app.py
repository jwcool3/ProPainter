import sys
sys.path.append("../../")

import os
import json
import argparse
import shutil

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from tools.painter import mask_painter
from track_anything import TrackingAnything

from model.misc import get_device
from utils.download_util import load_file_from_url

def parse_augment():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--sam_model_type', type=str, default="vit_h")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--mask_save', default=False)
    args = parser.parse_args()
    
    if not args.device:
        args.device = str(get_device())

    return args 

def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type": ["click"],
        "input_point": click_state[0],
        "input_label": click_state[1],
        "multimask_output": "True",
    }
    return prompt

def clear_output_directory():
    output_dir = "output_images"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(os.path.join(output_dir, "masks"))

def initialize_images(file_paths):
    clear_output_directory()  # Clear previous outputs

    if not file_paths:
        return [], None  # Return empty states and no image if no files are uploaded

    images = []
    for file in file_paths:
        with Image.open(file) as img:
            img = img.convert("RGB")
            images.append(np.array(img))
    
    image_states = []
    for image in images:
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        model.samcontroler.sam_controler.reset_image()
        model.samcontroler.sam_controler.set_image(image)
        image_states.append({
            "origin_image": image,
            "painted_image": image.copy(),
            "mask": mask,
        })
    
    return image_states, images[0]

def sam_refine(image_states, index, point_prompt, click_state, evt: gr.SelectData):
    if index < 0 or index >= len(image_states):
        return None, image_states

    image_state = image_states[index]
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
    
    prompt = get_prompt(click_state=click_state, click_input=coordinate)
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(image_state["origin_image"])

    mask, logit, painted_image = model.first_frame_click(
        image=image_state["origin_image"], 
        points=np.array(prompt["input_point"]),
        labels=np.array(prompt["input_label"]),
        multimask=prompt["multimask_output"],
    )

    image_state["mask"] = mask
    image_state["painted_image"] = painted_image

    return painted_image, image_states

def save_current_image_and_mask(image_states, file_paths, current_index, click_state):
    output_dir = "output_images"
    mask_dir = os.path.join(output_dir, "masks")

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_state = image_states[current_index]
    file_path = file_paths[current_index]

    # Save original image as PNG
    image_filename = os.path.splitext(os.path.basename(file_path))[0] + ".png"
    image_path = os.path.join(output_dir, image_filename)
    Image.fromarray(image_state["origin_image"]).save(image_path)

    # Save mask as PNG
    mask_filename = os.path.splitext(os.path.basename(file_path))[0] + "_mask.png"
    mask_path = os.path.join(mask_dir, mask_filename)
    mask = (image_state["mask"] * 255).astype(np.uint8)
    Image.fromarray(mask).save(mask_path)

    # Clear clicks automatically
    click_state = [[], []]

    # Advance to the next image
    next_index = min(current_index + 1, len(image_states) - 1)

    return image_states[next_index]["painted_image"], next_index, click_state

def prepare_download():
    # Zip the output directory
    zip_filename = "images_and_masks.zip"
    shutil.make_archive("images_and_masks", 'zip', "output_images")
    return zip_filename

def clear_click(image_states, index, click_state):
    if index < 0 or index >= len(image_states):
        return None, click_state

    click_state = [[], []]
    template_frame = image_states[index]["origin_image"]
    return template_frame, click_state

def navigate_images(image_states, current_index, direction):
    new_index = current_index + direction
    if new_index < 0 or new_index >= len(image_states):
        new_index = current_index  # Keep the index within bounds
    return image_states[new_index]["painted_image"], new_index

args = parse_augment()
pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
sam_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
checkpoint_folder = os.path.join('..', '..', 'weights')

sam_checkpoint = load_file_from_url(sam_checkpoint_url_dict[args.sam_model_type], checkpoint_folder)
cutie_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'cutie-base-mega.pth'), checkpoint_folder)
propainter_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'ProPainter.pth'), checkpoint_folder)
raft_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'raft-things.pth'), checkpoint_folder)
flow_completion_checkpoint = load_file_from_url(os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), checkpoint_folder)

model = TrackingAnything(sam_checkpoint, cutie_checkpoint, propainter_checkpoint, raft_checkpoint, flow_completion_checkpoint, args)

with gr.Blocks() as iface:
    click_state = gr.State([[], []])
    image_states = gr.State([])
    current_index = gr.State(0)

    gr.Markdown("<h1 align='center'>ProPainter Image Masking</h1>")
    gr.Markdown("Upload one or more images, select points to create a mask, and save the masks and images as black and white PNGs.")

    with gr.Row():
        painted_image = gr.Image(type="pil", label="Painted Image", interactive=True)
    
    with gr.Row():
        point_prompt = gr.Radio(
            choices=["Positive", "Negative"],
            value="Positive",
            label="Point prompt",
            interactive=True,
        )
    
    with gr.Row():
        prev_button = gr.Button("Previous Image")
        next_button = gr.Button("Next Image")
        clear_button_click = gr.Button("Clear clicks")
        save_button = gr.Button("Download All Masks and Images")
    
    with gr.Row():
        image_input = gr.File(label="Upload Images", file_count="multiple", type="filepath")

    image_input.change(
        fn=initialize_images,
        inputs=image_input,
        outputs=[image_states, painted_image]
    )

    painted_image.select(
        fn=sam_refine,
        inputs=[image_states, current_index, point_prompt, click_state],
        outputs=[painted_image, image_states]
    )

    next_button.click(
        fn=save_current_image_and_mask,
        inputs=[image_states, image_input, current_index, click_state],
        outputs=[painted_image, current_index, click_state]
    )

    save_button.click(
        fn=prepare_download,
        inputs=None,
        outputs=gr.File()
    )

    clear_button_click.click(
        fn=clear_click,
        inputs=[image_states, current_index, click_state],
        outputs=[painted_image, click_state]
    )

    prev_button.click(
        fn=navigate_images,
        inputs=[image_states, current_index, gr.State(-1)],
        outputs=[painted_image, current_index]
    )

iface.launch(debug=True)
