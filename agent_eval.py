import datetime
import os
import re
import time
import json
from os.path import join
from PIL import Image
import minedojo
import numpy as np
import openai
from PIL import Image
from openai import OpenAI
from utils import encode_image, write_text_on_image, obs_to_json, calculate_distance, check_distance
from config import run_config, task_config

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
client = OpenAI(api_key=api_key)

cwd = os.getcwd()
rgb_obs_dir = join(cwd, "rgb_frames")
info_dir = join(cwd, "info")
obs_dir = join(cwd, "obs")
run_history_dir = join(cwd, "run_history")
run_start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_rgb_obs_dir = join(rgb_obs_dir, f"{run_start_time}")
run_info_dir = join(info_dir, f"{run_start_time}")
run_obs_dir = join(obs_dir, f"{run_start_time}")


def predict_action(obs, step, task_prompt, task_guidance, error_count):
    success = 1
    predicted_actions = []
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"{step}.jpg"))
    # Can delete previous images as well
    encoded_obs_image = encode_image(join(run_rgb_obs_dir, f"{step}.jpg"))
    image_url = f"data:image/jpeg;base64,{encoded_obs_image}"

    with open("prompt_3.txt", "r") as f:
        prompt_text_raw = f.read()
    with open("action_desc.json", "r") as f:
        action_desc = json.load(f)
    prompt_text = prompt_text_raw.format(
        #actions=action_desc,
        task=task_prompt,
        guidance=task_guidance)
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
        action_vec, parsing_success = extract_action_vector(llm_output)
        if isinstance(action_vec, list):
            predicted_actions.extend(action_vec)
        elif isinstance(action_vec, np.ndarray):
            predicted_actions.append(action_vec)
        print(action_vec)
        error_count = 0
        for predicted_action in predicted_actions:
            if not (isinstance(predicted_action, np.ndarray) and predicted_action.shape == (8,) and issubclass(predicted_action.dtype.type, np.integer)):
                raise ValueError("The action is not a valid 8-element integer numpy array.")
        return predicted_actions, error_count, parsing_success, success
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
    success = 0
    predicted_actions.append(np.array([0, 0, 0, 12, 12, 0, 0, 0]))
    return predicted_actions, error_count, parsing_success, success

def extract_action_vector(llm_output):
    success = 1
    llm_output = llm_output.replace(' ', '').replace('\n', '')
    llm_output = llm_output.replace(r'\[', '[').replace(r'\]', ']')
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
            #vector_list_str = vector_list_str.replace(' ', '').replace('\n', '')
            vector_list = vector_list_str.split("],[")
            # Convert each vector string to a numpy array of integers
            action_vectors = [np.array([int(x) for x in vec.split(',')]) for vec in vector_list]
            return action_vectors, success

        match_single = re.search(pattern_single, llm_output)
        if match_single:
            # Extract the single vector
            vector_str = match_single.group(1) if match_single.group(1) else match_single.group(3)
            # Convert the string into a list of integers and return it as a numpy array
            action_vector = np.array([int(x) for x in vector_str.split(',')])
            return action_vector, success
        # Raise ValueError if no match was found
        raise ValueError("No valid action vector or list of vectors found in the output, expecting exactly 8 elements.")
    
    except ValueError as e:
        print(e)
        success = 0
        # Return default vector if ValueError is raised
        return np.array([0, 0, 0, 12, 12, 0, 0, 0]), success

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
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(run_history_dir, exist_ok=True)
    os.makedirs(run_rgb_obs_dir, exist_ok=True)
    os.makedirs(run_info_dir, exist_ok=True)
    os.makedirs(run_obs_dir, exist_ok=True)
    env = minedojo.make(**task_config["easy_1"])
    print(env.task_prompt)
    print(env.task_guidance)
    obs = env.reset()
    first_pos = obs["location_stats"]["pos"]
    sent_actions = []
    # action_buffer = [] # To check if same actions predicted repeatedly
    error_count = 0 # Number of consecutive
    step = 0
    api_calls = 0
    total_distance = 0
    run_history = {}
    run_start = time.time()

    while step < run_config["step_count"]:
        start = time.time()
        predicted_actions, error_count, parsing_success, api_success = predict_action(obs=obs,
                                                                                    step=step,
                                                                                    task_prompt=env.task_prompt,
                                                                                    task_guidance=env.task_guidance,
                                                                                    error_count=error_count)
        api_calls += 1
        if error_count == run_config["error_limit"]:
            print("API requests failed for 5 times, ending the run.")
            break

        # action_buffer.extend(predicted_actions)
        # if check_if_same_actions(action_buffer, run_config["same_action_limit"]):
        #     print(f"Received the same action for {run_config['same_action_limit']} steps, ending the run.")
        #     break
        end = time.time()
        print("Predict Action Time: ", end-start)
        while predicted_actions and step < run_config["step_count"]:
            action = predicted_actions.pop(0)
            obs, reward, done, info = env.step(action)

            # Calculate the distance between each following step and add it to the 'total_distance'
            second_pos = obs["location_stats"]["pos"]
            total_distance += calculate_distance(first_pos, second_pos)

            # Check if the 'total_distance' in the last n steps is greater than a certain value (otherwise: agent stuck!)
            total_distance, done = check_distance(total_distance, step)

            # Logging the actions of the run with a possible error suffix
            sent_actions.append(str(action.tolist()) 
                                + (" - Parsing Error" if not parsing_success else "")
                                + (" - API Error" if not api_success else ""))

            # Logging the 'info' and the 'obs' returned by the environment
            with open(join(run_info_dir, f"info_step_{step}.json"), "w") as f:
                json.dump(info, f, indent=4)
            obs_to_json(obs, run_obs_dir, step)

            if done:
                break
            step += 1
            first_pos = second_pos
        if done:
            break
    
    run_finish = time.time()

    run_history["actions_count"] = step
    run_history["api_calls_count"] = api_calls
    run_history["run_duration"] = run_finish - run_start
    run_history["final_reward"] = reward
    run_history["final_done"] = done
    run_history["actions"] = sent_actions
    with open(join(run_history_dir, f"run_{run_start_time}.json"), "w") as f:
        json.dump(run_history, f, indent=4)
    env.close()
