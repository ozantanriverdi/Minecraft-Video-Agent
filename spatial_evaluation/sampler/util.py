import datetime
import os
import ast
import math
import copy
import json
import random
from pathlib import Path
import numpy as np
from os.path import join
from env_data import *
import math


# def create_folders():
#     # Get the parent directory (spatial_evaluation)
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Moves up one level from "sampler"
    
#     # Define the directory where samples should be stored
#     samples_dir = os.path.join(base_dir, "samples")
    
#     # Create a timestamped subdirectory
#     run_start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#     run_rgb_obs_dir = os.path.join(samples_dir, "rgb_frames", run_start_time)
#     run_info_dir = os.path.join(samples_dir, "info", run_start_time)
#     run_obs_dir = os.path.join(samples_dir, "obs", run_start_time)

#     # Create directories recursively
#     os.makedirs(run_rgb_obs_dir, exist_ok=True)
#     os.makedirs(run_info_dir, exist_ok=True)
#     os.makedirs(run_obs_dir, exist_ok=True)

#     return run_rgb_obs_dir, run_info_dir, run_obs_dir

def create_folders():
    # Get the parent directory (spatial_evaluation)
    base_dir = Path(__file__).parent.parent  # Moves up one level from "sampler"
    
    # Define the directory where samples should be stored
    samples_dir = base_dir / "samples"
    
    # Create a timestamped subdirectory
    run_start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_rgb_obs_dir = samples_dir / "rgb_frames" / run_start_time
    run_info_dir = samples_dir / "info" / run_start_time
    run_obs_dir = samples_dir / "obs" / run_start_time

    # Create directories recursively
    run_rgb_obs_dir.mkdir(parents=True, exist_ok=True)
    run_info_dir.mkdir(parents=True, exist_ok=True)
    run_obs_dir.mkdir(parents=True, exist_ok=True)

    return run_rgb_obs_dir, run_info_dir, run_obs_dir

def entity_random_location(obs):
    x_agent, y_agent, z_agent = obs["location_stats"]["pos"]
    print(f"Agent Coordinate: ({x_agent}, {y_agent}, {z_agent})")
    x_coord = obs["rays"]["traced_block_x"]
    y_coord = obs["rays"]["traced_block_y"]
    z_coord = obs["rays"]["traced_block_z"]
    is_solid = obs["rays"]["is_solid"]

    max_tries = 50
    chosen_x_value_rel, chosen_z_value_rel = None, None
    spawn_y = y_agent  # default spawn if no valid block is found

    for attempt in range(max_tries):
        x_value_rel = np.random.randint(-5, 5)
        z_value_rel = np.random.randint(4, 10)
        x = math.floor(x_agent + x_value_rel)
        z = math.floor(z_agent + z_value_rel)

        idx, idz = np.where(x_coord == x), np.where(z_coord == z)
        common_idx = np.intersect1d(idx[0], idz[0]) # gives all blocks that share the same 'X' and 'Z'.
        #print(f"Common idx: {common_idx}")
        solid_idx = np.where(is_solid[common_idx])[0] # only want solid blocks for the ground -> all solid blocks at (X,Z)
        #print(f"Solid idx: {solid_idx}")
        valid_idx = common_idx[solid_idx] # only pick the solid indices from the common ones
        #print(f"Valid idx: {valid_idx}")

        if valid_idx.size != 0:
            print(f"X-Coordinate: {x_value_rel}")
            print(f"Z-Coordinate: {z_value_rel}")
            print(f"Y-Coordinate candidates: {y_coord[valid_idx]}")
            y_block = min(y_coord[valid_idx])
            spawn_y = y_block + 1
            chosen_x_value_rel = x_value_rel
            chosen_z_value_rel = z_value_rel
            print(f"Found valid location after {attempt+1} attempt(s).")
            break
    else:
        print(f"No suitable location found after {max_tries} attempts, spawning at agent level")

    # If we found a valid location, chosen_x_value_rel and chosen_z_value_rel won't be None
    if chosen_x_value_rel is None:
        chosen_x_value_rel = 0
    if chosen_z_value_rel is None:
        chosen_z_value_rel = 0

    y_value_rel = spawn_y - y_agent
    return np.array([chosen_x_value_rel, y_value_rel, chosen_z_value_rel])


def entity_random_location_1(obs):
    x_agent, y_agent, z_agent = obs["location_stats"]["pos"]
    print(f"Agent Coordinate: ({x_agent}, {y_agent}, {z_agent})")
    x_coord = obs["rays"]["traced_block_x"]
    y_coord = obs["rays"]["traced_block_y"]
    z_coord = obs["rays"]["traced_block_z"]
    is_solid = obs["rays"]["is_solid"]

    candidates = [(x, z) for x in range(-5, 5) for z in range(4, 10)]
    random.shuffle(candidates)

    chosen_x_value_rel, chosen_z_value_rel = None, None
    spawn_y = y_agent

    for x_value_rel, z_value_rel in candidates:
        x_candidate = math.floor(x_agent + x_value_rel)
        z_candidate = math.floor(z_agent + z_value_rel)

        idx_x = np.where(x_coord == x_candidate)
        idx_z = np.where(z_coord == z_candidate)
        common_idx = np.intersect1d(idx_x[0], idx_z[0])

        solid_idx = np.where(is_solid[common_idx])[0]
        valid_idx = common_idx[solid_idx]

        if valid_idx.size != 0:
            print(f"X-Coordinate: {x_value_rel}")
            print(f"Z-Coordinate: {z_value_rel}")
            print(f"Y-Coordinate candidates: {y_coord[valid_idx]}")
            y_block = min(y_coord[valid_idx])
            spawn_y = y_block + 1
            chosen_x_value_rel = x_value_rel
            chosen_z_value_rel = z_value_rel
            break
    else:
        print(f"No suitable location found, spawning at agent level")

    # If we found a valid location, chosen_x_value_rel and chosen_z_value_rel won't be None
    if chosen_x_value_rel is None:
        chosen_x_value_rel = 0
    if chosen_z_value_rel is None:
        chosen_z_value_rel = 0

    y_value_rel = spawn_y - y_agent
    return np.array([chosen_x_value_rel, y_value_rel, chosen_z_value_rel])

def entity_random_location_simple():
    x_value_1 = np.random.randint(-5, 5)
    z_value_1 = np.random.randint(4, 10)

    x_value_2 = np.random.randint(-5, 5)
    z_value_2 = np.random.randint(4, 10)

    if x_value_1 == x_value_2 and z_value_1 == z_value_2:
        x_value_1 = np.random.randint(-5, 5)
        z_value_1 = np.random.randint(4, 10)

    return [np.array([x_value_1, 3, z_value_1]), np.array([x_value_2, 3, z_value_2])]


def entity_deterministic_location(biome):
    locations = []
    length = len(entities[biome])
    start_loc = -math.floor(5/2)
    # More space between Nether monsters since they're bigger
    if biome == "nether":
        for i in range(length):
            locations.append([start_loc+3*i, 0, 10])
    else:
        for i in range(length):
            locations.append([start_loc+i, 2, 6])
    return locations

def agent_random_location():
    """
    x: positive values mean left
    y: 4 is the ground level
    z: positive values mean front
    """
    x_value = np.random.randint(-1000, 1000)
    z_value = np.random.randint(-1000, 1000)
    return x_value, 4, z_value

def obs_to_json(obs, run_obs_dir, step, biome_id, frame):
    """
    Convert the numpy arrays in textual observations to string and save as json
    """
    obs_copy = copy.deepcopy(obs)
    for i, key_1 in enumerate(obs_copy.keys()):
        if i == 0:
            continue
        for key_2 in obs_copy[key_1].keys():
            if isinstance(obs_copy[key_1][key_2], np.ndarray):
                obs_copy[key_1][key_2] = str(obs_copy[key_1][key_2].tolist())#.replace('\n', '')
    del obs_copy["rgb"]
    with open(join(run_obs_dir, f"obs_step_{biome_id}_{step}_{frame}.json"), "w") as f:
        json.dump(obs_copy, f, indent=4)

def sample_entities(biome):
    if biome not in entities:
        raise ValueError(f"Biome '{biome}' not found in entity list.")
    biome_entities = entities[biome]
    if len(biome_entities) < 2:
        raise ValueError(f"Not enough entities in biome '{biome}' to sample two.")
    
    return random.sample(biome_entities, 2)


def random_action_generator(obs):
    
    forward_back = random.randint(0, 2)  # 0=noop, 1=forward, 2=back
    left_right = random.randint(0, 2)    # 0=noop, 1=left,   2=right
    #jump_sneak_sprint = random.randint(0, 3)  # 0=noop, 1=jump, 2=sneak, 3=sprint
    # --- 2) Get current pitch and yaw from obs ---
    # These are stored as strings like "[0.0]" representing degrees in [-180..180]
    pitch_degs = float(obs["location_stats"]["pitch"][0])
    yaw_degs   = float(obs["location_stats"]["yaw"][0])

    # Convert from [-180..180] -> discrete [0..24]
    # 0 => -180°, 24 => 180°, 12 => 0° (no movement)
    def angle_to_discrete(angle_degs):
        # Shift angle from [-180..180] to [0..360]
        shifted = angle_degs + 180.0
        # Scale into 25 bins
        discrete = int(round((shifted / 360.0) * 24))
        # Clamp to [0..24]
        return max(0, min(discrete, 24))

    pitch_index = angle_to_discrete(pitch_degs)
    yaw_index   = angle_to_discrete(yaw_degs)

    # --- 3) Random small offset ([-1..1]) so camera changes are small ---
    def random_small_offset(value):
        # pick from value-1, value, value+1
        candidates = [max(0, value-1), value, min(24, value+1)]
        return random.choice(candidates)

    pitch_index = random_small_offset(pitch_index)
    yaw_index   = random_small_offset(yaw_index)
    
    dimension_pick = random.randint(0, 3)
    if dimension_pick == 0:
        action = [forward_back,0,0,12,12,0,0,0]
    elif dimension_pick == 1:
        action = [0,left_right,0,12,12,0,0,0]
    elif dimension_pick == 2:
        action = [0,0,0,pitch_index,12,0,0,0]
    elif dimension_pick == 3:
        action = [0,0,0,12,yaw_index,0,0,0]

    return np.array(action)


def random_action_sampler(last_action, pitch_delta, yaw_delta):
    forward_back = random.randint(0, 2)  # 0=noop, 1=forward, 2=back
    left_right = random.randint(0, 2)    # 0=noop, 1=left,   2=right

    # pitch_candidates = [max(0, last_action[3]-1), min(24, last_action[3]+1)]
    # yaw_candidates = [max(0, last_action[4]-1), min(24, last_action[4]+1)]

    pitch_candidates = [11, 13]
    yaw_candidates = [11, 13]

    pitch_index = random.choice(pitch_candidates)
    yaw_index = random.choice(yaw_candidates)

    dimension_pick = random.randint(0, 3)
    if dimension_pick == 0:
        action = [forward_back,0,0,12,12,0,0,0]
    elif dimension_pick == 1:
        action = [0,left_right,0,12,12,0,0,0]
    elif dimension_pick == 2:
        action = [0,0,0,pitch_index,12,0,0,0]
        pitch_change = pitch_index - 12
        pitch_delta += pitch_change
    elif dimension_pick == 3:
        action = [0,0,0,12,yaw_index,0,0,0]
        yaw_change = yaw_index - 12
        yaw_delta += yaw_change

    return np.array(action), pitch_delta, yaw_delta

def pitch_yaw_corrector(last_action):
    # 17 - 12 = 5 -> 12 - 5 = 7
    # 7 - 12 = -5 -> 12 - (-5) = 17
    pitch_index = last_action[3] - 12
    correct_pitch = 12 - pitch_index
    yaw_index = last_action[4] - 12
    correct_yaw = 12 - yaw_index
    correcting_action = np.array([0,0,0,correct_pitch,correct_yaw,0,0,0])

    return correcting_action

if __name__ == "__main__":
    # with open("../obs/20250302_201247/obs_step_0_17.json", "r") as f:
    #     obs_1 = json.load(f)
    # with open("../obs/20250302_183200/obs_step_0_9.json", "r") as f:
    #     obs_2 = json.load(f)
    # with open("../obs/20250302_183200/obs_step_1_5.json", "r") as f:
    #     obs_3 = json.load(f)
    

    # print(entity_random_location(obs_3))

    print(sample_entities("forest"))