import json
import minedojo
import math
import numpy as np
from PIL import Image
from os.path import join
from scipy.stats import mode
from util import *
from env_data import *

def wrap_to_pi(a_deg):
    return ((a_deg + 180) % 360) - 180

def get_mode_coords(x, y, z, indices):
    if len(indices) == 0:
        return None
    coords = np.stack([x[indices], y[indices], z[indices]], axis=1)
    coord_tuples = [tuple(coord) for coord in coords]
    mode_coord, _ = mode(coord_tuples, axis=0, keepdims=False)
    
    return mode_coord

def detect_entity_loc(obs, entity_name):
    entities = np.array(obs["rays"]["entity_name"])
    traced_block_x = np.array(obs["rays"]["traced_block_x"])
    traced_block_y = np.array(obs["rays"]["traced_block_y"])
    traced_block_z = np.array(obs["rays"]["traced_block_z"])

    entity_idx = np.where(entities == entity_name)[0]
    entity_coords = get_mode_coords(traced_block_x, traced_block_y, traced_block_z, entity_idx)
    if entity_coords is None:
        return False
    
    return entity_coords

def check_within_fov(obs, entity_name):
    x_entity, _, z_entity = detect_entity_loc(obs, entity_name)
    x_agent, _, z_agent = np.array(obs["location_stats"]["pos"])

    delta_x = x_entity - x_agent
    delta_z = z_entity - z_agent

    bearing = math.atan2(delta_x, delta_z)
    bearing_deg = math.degrees(bearing)
    agent_yaw = obs["location_stats"]["yaw"]
    delta = wrap_to_pi(bearing_deg - agent_yaw)
    print(f"Delta Value: {delta}")

    if abs(delta) <= 52:
        return True
    else:
        return False




if __name__ == '__main__':
    run_rgb_obs_dir, run_info_dir, run_obs_dir = create_folders()

    trajectory_count = 1
    frame_count = 2
    entity_name = "cow"
    vradius = 5

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
                for pitch in np.arange(-55, 55, 5)
                for yaw in np.arange(-55, 55, 5)
        ]
    )


    obs_init = env.reset()
    for k in range(4):
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    
    env.spawn_mobs(entity_name, [5, 1, 5])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    entity_in_fov = check_within_fov(obs, entity_name)
    print(entity_in_fov)
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save("1.jpg")
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    obs, _, _, _ = env.step([0,0,0,12,11,0,0,0])
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save("2.jpg")
    print(obs["location_stats"]["yaw"])



    

    env.close()