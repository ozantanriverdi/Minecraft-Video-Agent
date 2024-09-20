import base64
from PIL import Image, ImageDraw, ImageFont
import textwrap
from tqdm import tqdm
import numpy as np
import json

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

def obs_to_json(obs):
    
    for i, key_1 in enumerate(obs.keys()):
        if i == 0:
            continue
        for key_2 in obs[key_1].keys():
            if isinstance(obs[key_1][key_2], np.ndarray):
                obs[key_1][key_2] = str(obs[key_1][key_2])
    del obs["rgb"]
    with open("obs_1.json", "w") as f:
        json.dump(obs, f, indent=4)

