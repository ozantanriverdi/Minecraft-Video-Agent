import re
import numpy as np
import json
import minedojo
import time
from minedojo.sim import InventoryItem
from PIL import Image
from utils import obs_to_json, calculate_distance
from config import run_config
#from config import task_config
#from config import easy_task_parameters

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
    

if __name__ == '__main__':
    text = """
        Based on the current observation, it seems you are in a forested area with a lot of trees and foliage. There are no visible cows in this image.

        To proceed with the task of obtaining milk from a cow, the first action will involve moving forward to search for a cow. Here are a couple of actions to take:

        1. Move forward to explore the area.
        2. Continue moving forward to potentially find a cow.

        Here are the action vectors:

        ```python
        [[1, 0, 0, 12, 12, 0, 0, 0], 
        [1, 0, 0, 12, 12, 0, 0, 0]]
        ```"""
    text_2 = """
        Based on the current observation, it seems you are in a forested area with a lot of trees and foliage. There are no visible cows in this image.

        To proceed with the task of obtaining milk from a cow, the first action will involve moving forward to search for a cow. Here are a couple of actions to take:

        1. Move forward to explore the area.
        2. Continue moving forward to potentially find a cow.

        Here are the action vectors:

        ```python
        \[1, 5, 0, 12, 12, 0, 0, 0\]
        ```"""
    text_3 = """
        Based on the current observation, it seems you are in a forested area with a lot of trees and foliage. There are no visible cows in this image.

        To proceed with the task of obtaining milk from a cow, the first action will involve moving forward to search for a cow. Here are a couple of actions to take:

        1. Move forward to explore the area.
        2. Continue moving forward to potentially find a cow.

        Here are the action vectors:

        ```python
        [\[1, 4, 0, 12, 12, 0, 0, 0], [1, 6, 0, 12, 12, 0, 0, 0], [1, 7, 0, 12, 12, 0, 0, 0]\]
        ```"""
    text_4 = """
        To milk the cow in front of you, follow this predefined action sequence:

        1. **Move Forward** (five times)
        2. **Bend Forward** to aim the camera at the cow
        3. **Use the Bucket** (three times)

        Here are the actions formatted as a list of action vectors:

        ```
        [[5, 0, 0, 12, 12, 0, 0, 0],
        [1, 0, 0, 12, 12, 0, 0, 0],
        [1, 0, 0, 12, 12, 0, 0, 0],
        [1, 0, 0, 12, 12, 0, 0, 0],
        [1, 0, 0, 12, 12, 0, 0, 0],
        [0, 0, 0, 15, 12, 0, 0, 0],
        [0, 0, 0, 12, 12, 1, 0, 0],
        [0, 0, 0, 12, 12, 1, 0, 0],
        [0, 0, 0, 12, 12, 1, 0, 0],
        [0, 0, 0, 12, 12, 1,
    """


    task = {
        "task_id": "harvest",
        "target_names": "milk_bucket",    # Items to harvest
        "target_quantities": 1,          # Quantities to harvest
        "specified_biome": "plains",           # Biome where task takes place
        # "spawn_rate": 0.99,
        # "spawn_range_low": (0, 0, 1),
        # "spawn_range_high": (0, 0, 1),
        "initial_mobs": "cow",
        "initial_mob_spawn_range_low": (0, 0, 1),
        "initial_mob_spawn_range_high": (0, 0, 1),
        "image_size": (480, 768),
        "seed": 1,
        "world_seed": 40,
        "initial_inventory": [
            InventoryItem(slot="mainhand", name="bucket", variant=None, quantity=1)
        ],
        "initial_weather": "clear",
        #"start_position": {"x": 190.5, "y": 69, "z": 248.5, "pitch": 0, "yaw": 0} # x: negative values mean right
    }



    # Create the environment with task_id="harvest" and your custom parameters
    env = minedojo.make(**task)
    print("************")
    print(env.task_prompt)
    print(env.task_guidance)
    print("************2")
    # Now you can use the environment as usual
    obs = env.reset()
    print("************3")
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(f"seed_{task['world_seed']}.jpg")
    print("Initial Inventory:", obs["inventory"]["name"])
    print(obs["location_stats"]["pos"], obs["location_stats"]["yaw"], obs["location_stats"]["pitch"])
    for step in range(5):  # Example loop
        action = np.array([1, 0, 0, 12, 12, 0, 0, 0])  # Replace with your agent's action
        obs, reward, done, info = env.step(action)

    
    action = np.array([0, 0, 0, 15, 12, 0, 0, 0])
    obs, reward, done, info = env.step(action)
    for step in range(3):  # Example loop
        if step == 0:
            action = np.array([0, 0, 0, 12, 12, 1, 0, 0])  # Replace with your agent's action
            obs, reward, done, info = env.step(action)
            print("***" + str(step) + "***")
            print("Done: ", done)
            print("Reward: ", reward)
        else:
            action = np.array([0, 0, 0, 12, 12, 0, 0, 0])
            obs, reward, done, info = env.step(action)
            print("***" + str(step) + "***")
            print("Done: ", done)
            print("Reward: ", reward)
        if done:
            print("Task completed!")
            #break
        #print(obs["location_stats"]["pos"], obs["location_stats"]["yaw"], obs["location_stats"]["pitch"])
        print("Inventory:", obs["inventory"]["name"])

    
    env.close()

    
    #print(extract_action_vector(text_4))