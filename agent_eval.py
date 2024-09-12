import base64
import datetime
import os
import re
import time
import json
from os.path import join
import textwrap
from PIL import Image, ImageDraw, ImageFont
import minedojo
import numpy as np
import openai
from PIL import Image
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key)

cwd = os.getcwd()
rgb_obs_dir = join(cwd, "rgb_frames")
obs_dir = join(cwd, "obs")
run_rgb_obs_dir = join(rgb_obs_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
run_obs_dir = join(obs_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")


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

def predict_action(obs, step, task_prompt, task_guidance, error_count):
    predicted_actions = []
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"{step}.jpg"))
    # Can delete previous images as well
    encoded_obs_image = encode_image(join(run_rgb_obs_dir, f"{step}.jpg"))
    image_url = f"data:image/jpeg;base64,{encoded_obs_image}"

    with open("prompt.json", "r") as f:
        prompt_data = json.load(f)
    with open("action_desc.json", "r") as f:
        action_desc = json.load(f)
    prompt_text = prompt_data["prompt"][0]["text"].format(
        actions=action_desc,
        task=task_prompt,
        guidance=task_guidance)
    #prompt_data["prompt"][0]["text"] = prompt_text
    #prompt_data["prompt"][1]["image_url"]["url"] = image_url
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Replace with the appropriate model name
            messages=messages,
            max_tokens=300)
        start = time.time()
        llm_output = response.choices[0].message.content
        print(llm_output)
        end = time.time()
        print("Request Time: ", end-start)
        output_image_path = join(run_rgb_obs_dir, f"annotated_step_{step}.jpg")
        write_text_on_image(join(run_rgb_obs_dir, f"{step}.jpg"), llm_output, output_image_path)
        # Create an empty action and put in the actions from the response
        # action = env.action_space.no_op()
        # action[0] = 1
        # return action
        action_vec = extract_action_vector(llm_output)
        if isinstance(action_vec, list):
            predicted_actions.extend(action_vec)
        elif isinstance(action_vec, np.ndarray):
            predicted_actions.append(action_vec)
        print(action_vec)
        error_count = 0
        for action in predicted_actions:
            if not (isinstance(action, np.ndarray) and action.shape == (8,) and issubclass(action.dtype.type, np.integer)):
                raise ValueError("The action is not a valid 8-element integer numpy array.")
        return predicted_actions, error_count
    except openai.InternalServerError:
        print("The LLM API service is temporarily unavailable. Please try again later.")

    except openai.AuthenticationError:
        print("There was an issue with API authentication. Please check your API key.")

    except openai.RateLimitError:
        print("You have exceeded your rate limit or run out of credits. Please check your usage.")

    except openai.APITimeoutError:
        print(f"Request timed out.")

    except openai.APIConnectionError:
        print("Network error: Unable to connect to the OpenAI API. Please check your internet connection.")

    except ValueError as e:
        print(f"Invalid action predicted: {e}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # Return a default action vector in case of any error
    error_count += 1
    predicted_actions.append(np.array([0, 0, 0, 12, 12, 0, 0, 0]))
    return predicted_actions, error_count



def parse_action_vector(api_response):
    start = api_response.find('[')
    end = api_response.find(']')
    action_vec_str = api_response[start:end+1]

    action_vec_str = action_vec_str.lstrip('[').rstrip(']').replace(' ', '')
    action_vec = []
    for i in range(8):
        if i == 7:
            action = int(action_vec_str)
            action_vec.append(action)
        else:
            end_index = action_vec_str.find(',')
            action = int(action_vec_str[:end_index])
            action_vec.append(action)
            action_vec_str = action_vec_str[end_index+1:]

    action_vec = np.array(action_vec)
    return action_vec

def extract_action_vector(llm_output):
    # Regex pattern to match exactly 8 elements inside square brackets [] or parentheses ()
    pattern_single = r"\[((\d+,\s*){7}\d+)\]|\(((\d+,\s*){7}\d+)\)"

    # Regex pattern to match a list of action vectors, each with exactly 8 elements
    pattern_list = r"\[\[((\d+,\s*){7}\d+)(\],\s*\[((\d+,\s*){7}\d+))*\]\]"

    try:
        # Search for the pattern in the LLM output
        match_list = re.search(pattern_list, llm_output)
        if match_list:
            # Extract the list of action vectors as a string
            vector_list_str = match_list.group(0)[2:-2] # Remove outer [[ and ]]
            # Split by "], [" to separate individual vectors
            vector_list_str = vector_list_str.replace(' ', '')
            vector_list = vector_list_str.split("],[")
            # Convert each vector string to a numpy array of integers
            action_vectors = [np.array([int(x) for x in vec.split(',')]) for vec in vector_list]
            return action_vectors

        match_single = re.search(pattern_single, llm_output)
        if match_single:
            # Extract the single vector
            vector_str = match_single.group(1) if match_single.group(1) else match_single.group(3)
            # Convert the string into a list of integers and return it as a numpy array
            action_vector = np.array([int(x) for x in vector_str.split(',')])
            return action_vector
        # Raise ValueError if no match was found
        raise ValueError("No valid action vector or list of vectors found in the output, expecting exactly 8 elements.")
    
    except ValueError as e:
        print(e)
        # Return default vector if ValueError is raised
        return np.array([0, 0, 0, 12, 12, 0, 0, 0])

def single_action_agent():
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"{step}.jpg"))
    action = np.array([0, 0, 0, 12, 11, 0, 0, 0])
    return action


def check_if_same_actions(action_buffer, step_count):
    """
    Checks if all the actions in the buffer have been the same for the last 'step_count' steps.

    Parameters:
    - action_buffer: List or buffer of actions taken.
    - step_count: The number of steps to check.
    
    Returns:
    - True if all the actions for the last 'step_count' steps are the same, False otherwise.
    """
    # Ensure there are enough steps in the buffer to check
    if len(action_buffer) < step_count:
        return False
    
    # Get the last 'step_count' actions
    recent_actions = action_buffer[-step_count:]

    # Check if all actions in 'recent_actions' are the same as the first one
    return all(np.array_equal(action, recent_actions[0]) for action in recent_actions)

if __name__ == '__main__':
    os.makedirs(rgb_obs_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(run_rgb_obs_dir, exist_ok=True)
    os.makedirs(run_obs_dir, exist_ok=True)
    env = minedojo.make(task_id="harvest_milk", image_size=(480, 768))
    print(env.task_prompt)
    print(env.task_guidance)
    obs = env.reset()
    action_buffer = [] # To check if same actions predicted repeatedly
    error_count = 0 # Number of consecutive
    step = 0

    while step < 10:
        start = time.time()
        predicted_actions, error_count = predict_action(obs=obs,
                                step=step,
                                task_prompt=env.task_prompt,
                                task_guidance=env.task_guidance,
                                error_count=error_count)
        if error_count == 5:
            print("API requests failed for 5 times, ending the run.")
            break

        action_buffer.extend(predicted_actions)
        if check_if_same_actions(action_buffer, 10):
            print("Received the same action for 10 steps, ending the run.")
            break
        end = time.time()
        print("Predict Action Time: ", end-start)
        while predicted_actions:
            action = predicted_actions.pop(0)
            obs, reward, done, info = env.step(action)
            print("Reward: ", reward, type(reward))
            with open(join(run_obs_dir, f"info_step_{step}.json"), "w") as f:
                json.dump(info, f, indent=4)
            if done:
                break
            step += 1
        if done:
            break
    env.close()
