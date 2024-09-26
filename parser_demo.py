import re
import numpy as np
import json
import minedojo
import time
from minedojo.sim import InventoryItem
from PIL import Image
from utils import obs_to_json, calculate_distance
from config import task_config
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
    # print(extract_action_vector(text))
    # print("****************")
    # print(extract_action_vector(text_2))
    # print("****************")
    # print(extract_action_vector(text_3))
    # with open("prompt.txt", "r") as f:
    #     prompt_text_raw = f.read()
    
    # print(prompt_text_raw)

    # with open("obs/20240919_052722/info_step_0.json", "r") as f:
    #     obs = json.load(f)
    
    # print(obs.keys())
    
    # env = minedojo.make(task_id="harvest_milk", image_size=(480, 768))
    # print(env.task_prompt)
    # print(env.task_guidance)
    # obs = env.reset()
    # first_pos = obs["location_stats"]["pos"]
    # action = np.array([1, 0, 0, 12, 12, 0, 0, 0])
    # total_distance = 0
    # for i in range(10):
    #     obs, reward, done, info = env.step(action)
    #     second_pos = obs["location_stats"]["pos"]
    #     total_distance += calculate_distance(first_pos, second_pos)
    #     first_pos = second_pos
    # env.close()
    # print("Total Covered Distance: ", total_distance)
    
    
    # print(type(obs['equipment']['name']))
    # print(type(obs['equipment']['name'][0]))
    #print(type(obs), obs.keys())
    #print(type(info), info.keys())
    # for i, key in enumerate(obs.keys()):
    #     if i == 0:
    #         continue
    #     print(key)
    #     print(type(obs[key]))
    #     print(obs[key].keys())
    #     print("**********************")
    #obs_to_json(obs)


    custom_task_params = {
        "target_names": ["log", "sapling"],    # Items to harvest
        "target_quantities": [5, 10],          # Quantities to harvest
        "reward_weights": {"log": 1.0, "sapling": 0.5},  # Reward weights
        "specified_biome": "forest",           # Biome where task takes place
        "start_health": 20.0,                  # Agent's starting health
        "start_food": 20,                      # Agent's starting food level
        "image_size": (480, 768),              # Set image size
        "use_voxel": False,                    # Whether to include voxel observations
        "use_lidar": True,                     # Whether to include lidar observations
        # Add any other parameters you wish to customize
    }

    initial_inventory = [
        InventoryItem(slot="mainhand", name="bucket", variant=None, quantity=1)
    ]

    easy_task_parameters = {
    "task_id": "harvest",
    "target_names": "milk_bucket",    # Items to harvest
    "target_quantities": 1,          # Quantities to harvest
    "specified_biome": "plains",           # Biome where task takes place
    #"spawn_rate": 0.99,
    #"spawn_range_low": (0, 0, 1),
    #"spawn_range_high": (0, 0, 1),
    "initial_mobs": "cow",
    "initial_mob_spawn_range_low": (0, 0, 1),
    "initial_mob_spawn_range_high": (0, 0, 1),
    "image_size": (1080, 1920),              # Set image size
    "seed": 1,
    "world_seed": 3,
    "initial_inventory": initial_inventory,
    "initial_weather": "clear"
    }
    easy_task_parameters["start_position"] = {
    "x": 5,
    "y": 64,  # Typical ground level in Minecraft
    "z": 0,
    "yaw": 0,  # Facing south (positive Z)
    "pitch": 0
    }



    # Create the environment with task_id="harvest" and your custom parameters
    env = minedojo.make(**task_config["easy_1"])
    print(env.task_prompt)
    print(env.task_guidance)
    # Now you can use the environment as usual
    obs = env.reset()
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(f"3.jpg")
    print("Initial Inventory:", obs["inventory"]["name"])
    #print(obs["location_stats"]["pos"], obs["location_stats"]["yaw"], obs["location_stats"]["pitch"])
    for step in range(5):  # Example loop
        action = np.array([1, 0, 0, 12, 12, 0, 0, 0])  # Replace with your agent's action
        obs, reward, done, info = env.step(action)
        #print(obs["location_stats"]["pos"], obs["location_stats"]["yaw"], obs["location_stats"]["pitch"])
        #time.sleep(0.5)
        print("***" + str(step) + "***")
        print("Done: ", done)
        print("Reward: ", reward)
        
        if done:
            print("Task completed!")
            break
    # print("Using!")
    # for step in range(50):  # Example loop
    #     action = np.array([0, 0, 0, 12, 12, 1, 0, 0])  # Replace with your agent's action
    #     obs, reward, done, info = env.step(action)
    #     #print(obs["location_stats"]["pos"], obs["location_stats"]["yaw"], obs["location_stats"]["pitch"])
    #     #time.sleep(0.5)
    #     print("***" + str(step) + "***")
    #     print("Done: ", done)
    #     print("Reward: ", reward)
    #     if done:
    #         print("Task completed!")
    #         break
    action = np.array([0, 0, 0, 15, 12, 1, 0, 0])
    obs, reward, done, info = env.step(action)
    print("Attacking!")
    for step in range(50):  # Example loop
        action = np.array([0, 0, 0, 12, 12, 1, 0, 0])  # Replace with your agent's action

        obs, reward, done, info = env.step(action)
        #print(obs["location_stats"]["pos"], obs["location_stats"]["yaw"], obs["location_stats"]["pitch"])
        #time.sleep(0.5)
        print("***" + str(step) + "***")
        print("Done: ", done)
        print("Reward: ", reward)
        if done:
            print("Task completed!")
            break
    print("Final Inventory:", obs["inventory"]["name"])
    env.close()