import base64
from PIL import Image, ImageDraw, ImageFont
import textwrap
from tqdm import tqdm
import numpy as np
import json
import copy
from os.path import join
from config import run_config


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
 
def write_text_on_image(image_path, text, output_path):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Define a font (use a default font for simplicity)
    # You can specify a TTF font file if you have one
    font = ImageFont.load_default()

    # Set the position and wrap the text (to avoid overflowing the image width)
    max_width = 50  # Define max characters per line (adjust based on your image size)
    wrapped_text = textwrap.fill(text, width=max_width)
    
    # Position to start the text
    text_position = (10, 10)  # Top-left corner with some padding
    
    # Add text to the image
    draw.text(text_position, wrapped_text, font=font, fill="white")
    
    # Save the image with the text overlay
    image.save(output_path)

def obs_to_json(obs, run_obs_dir, step):
    obs_copy = copy.deepcopy(obs)
    for i, key_1 in enumerate(obs_copy.keys()):
        if i == 0:
            continue
        for key_2 in obs_copy[key_1].keys():
            if isinstance(obs_copy[key_1][key_2], np.ndarray):
                obs_copy[key_1][key_2] = str(obs_copy[key_1][key_2].tolist())#.replace('\n', '')
    del obs_copy["rgb"]
    with open(join(run_obs_dir, f"obs_step_{step}.json"), "w") as f:
        json.dump(obs_copy, f, indent=4)

def task_to_str(task_dict):
    task_dic_copy = copy.deepcopy(task_dict)
    for i, key in enumerate(task_dic_copy.keys()):
        task_dic_copy[key] = str(task_dic_copy[key])
    return task_dic_copy

def calculate_distance(first_pos, second_pos):
    return np.linalg.norm(first_pos-second_pos).astype(np.float32)

def check_distance(total_distance, step):
    if (step != 0) and (step % run_config["check_distance_interval"] == run_config["check_distance_interval"] - 1):
        if total_distance > run_config["min_distance"]:
            total_distance = 0.0
            return total_distance, False
        else:
            print(f"Agent moved less than {run_config['min_distance']} in the last {run_config['check_distance_interval']} steps, ending the run.")
            return total_distance, True
    return total_distance, False

def trivial_action_generator(obs, step, task_prompt, task_guidance, error_count):
    predicted_actions = []
    success = 1
    error_count = 0
    parsing_success = 1
    # 5x np.array([1, 0, 0, 12, 12, 0, 0, 0]), 1x np.array([0, 0, 0, 15, 12, 0, 0, 0]), 2x np.array([0, 0, 0, 12, 12, 1, 0, 0]), 1x np.array([0, 0, 0, 12, 12, 0, 0, 0])
    for i in range(5):
        predicted_actions.append(np.array([1, 0, 0, 12, 12, 0, 0, 0]))
    predicted_actions.append(np.array([0, 0, 0, 15, 12, 0, 0, 0]))
    for i in range(10):
        predicted_actions.append(np.array([0, 0, 0, 12, 12, 1, 0, 0]))
    for i in range(10):
        predicted_actions.append(np.array([0, 0, 0, 12, 12, 0, 0, 0]))
    return predicted_actions, error_count, parsing_success, success

def empty_action_generator(obs, step, task_prompt, task_guidance, error_count):
    predicted_actions = []
    success = 1
    error_count = 0
    parsing_success = 1
    for i in range(40):
        predicted_actions.append(np.array([0, 0, 0, 12, 12, 0, 0, 0]))
    return predicted_actions, error_count, parsing_success, success
