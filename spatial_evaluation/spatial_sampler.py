import datetime
import copy
import os
import time
import json
import minedojo
import numpy as np
from PIL import Image
from os.path import join

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

def agent_random_location():
    """
    x: positive values mean left
    y: 4 is the ground level
    z: positive values mean front
    """
    x_value = np.random.randint(-1000, 1000)
    z_value = np.random.randint(-1000, 1000)
    return x_value, 4, z_value

def animal_random_location():
    # If one animal fully blocks the other from the vision, lidar doesn't detect it
    # -> Check if both entities are found before passing the image to VLM
    x_value = np.random.randint(-5, 5)
    z_value = np.random.randint(4, 10)
    return np.array([x_value, 0, z_value])

vradius = 5

cwd = os.getcwd()
rgb_obs_dir = join(cwd, "rgb_frames")
info_dir = join(cwd, "info")
obs_dir = join(cwd, "obs")
run_history_dir = join(cwd, "run_history")
run_start_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
run_rgb_obs_dir = join(rgb_obs_dir, f"{run_start_time}")
run_info_dir = join(info_dir, f"{run_start_time}")
run_obs_dir = join(obs_dir, f"{run_start_time}")

if __name__ == '__main__':
    os.makedirs(rgb_obs_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)
    os.makedirs(run_history_dir, exist_ok=True)
    os.makedirs(run_rgb_obs_dir, exist_ok=True)
    os.makedirs(run_info_dir, exist_ok=True)
    os.makedirs(run_obs_dir, exist_ok=True)

    env = minedojo.make(
        "open-ended",
        image_size=(480, 768),
        generate_world_type="flat",
        flat_world_seed_string="",
        use_voxel = True,
        voxel_size=dict(xmin=-vradius, ymin=-vradius, zmin=-vradius, xmax=vradius, ymax=vradius, zmax=vradius),
        use_lidar=True,
        lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 12) # ALERT: lidar range is now 5
                for pitch in np.arange(-90, 60, 5)
                for yaw in np.arange(-90, 90, 5)
        ]
    )

    env.reset()
    for i in range(100):
        x, y, z = agent_random_location()
        env.teleport_agent(x, y, z, 0.0, 0.0)
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        env.spawn_mobs(["horse", "pig"], [animal_random_location(), animal_random_location()])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        obs, reward, done, info = env.step([0,0,0,12,12,0,0,0])

        Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"{i}.jpg"))

        with open(join(run_info_dir, f"info_step_{i}.json"), "w") as f:
            json.dump(info, f, indent=4)

        obs_to_json(obs, run_obs_dir, i)

    env.close()

