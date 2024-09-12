import os
import datetime
import time
import base64
import requests
import json
import minedojo
import re
import numpy as np
import openai
from PIL import Image
from os.path import join
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key="api_key")

cwd = os.getcwd()
rgb_obs_dir = join(cwd, "rgb_frames")
obs_dir = join(cwd, "obs")
run_rgb_obs_dir = join(rgb_obs_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
run_obs_dir = join(obs_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def predict_action(obs, step, task_prompt, task_guidance):
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
        # Create an empty action and put in the actions from the response
        # action = env.action_space.no_op()
        # action[0] = 1
        # return action
        action_vec = extract_action_vector(llm_output)
        print(action_vec)
        return action_vec
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

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    # Return a default action vector in case of any error
    return np.array([0, 0, 0, 12, 12, 0, 0, 0])



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
    # Regex pattern to match vectors inside square brackets [] or parentheses ()
    pattern = r"\[(\d[\d,\s]*)\]|\((\d[\d,\s]*)\)"
    
    try:
        # Search for the pattern in the LLM output
        match = re.search(pattern, llm_output)
        
        if match:
            # Extract the vector, match.group(1) will capture the first group (square brackets)
            # match.group(2) will capture the second group (parentheses)
            vector_str = match.group(1) if match.group(1) else match.group(2)
            
            # Convert the string into a list of integers
            action_vector = np.array([int(x) for x in vector_str.split(',')])
            
            return action_vector
        else:
            # Raise ValueError if no match was found
            raise ValueError("No valid action vector found in the output, sending an empty action vector.")
    
    except ValueError as e:
        print(e)
        # Return default vector if ValueError is raised
        return np.array([0, 0, 0, 12, 12, 0, 0, 0])

def single_action_agent():
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"{step}.jpg"))
    action = np.array([0, 0, 0, 12, 11, 0, 0, 0])
    return action


if __name__ == '__main__':
    os.makedirs(rgb_obs_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(run_rgb_obs_dir, exist_ok=True)
    os.makedirs(run_obs_dir, exist_ok=True)
    env = minedojo.make(task_id="harvest_milk", image_size=(480, 768))
    print(env.task_prompt)
    print(env.task_guidance)
    obs = env.reset()

    for step in range(10):
        start = time.time()
        try:
            action = predict_action(obs=obs,
                                    step=step, 
                                    task_prompt=env.task_prompt, 
                                    task_guidance=env.task_guidance)
            if isinstance(action, np.ndarray) and action.shape == (8,) and issubclass(action.dtype.type, np.integer):
                print("Valid action:", action)
            else:
                print("Invalid action predicted")
        except ValueError as e:
            print(f"Invalid action predicted: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        end = time.time()
        print("Predict Action Time: ", end-start)
        obs, reward, done, info = env.step(action)
    
        print("Reward: ", reward, type(reward))

        # Doesn't work because: "TypeError: Object of type ndarray is not JSON serializable"
        # with open(join(obs_dir, "obs.json"), "w") as f:
        #     json.dump(obs["equipment"], f, indent=4)
        with open(join(run_obs_dir, f"info_step_{step}.json"), "w") as f:
            json.dump(info, f, indent=4)
        #print(obs.keys())
        if done:
            break


    env.close()