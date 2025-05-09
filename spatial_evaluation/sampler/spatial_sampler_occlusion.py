import json
import minedojo
import numpy as np
from PIL import Image
from os.path import join
from util import *
from env_data import *

if __name__ == '__main__':
    run_rgb_obs_dir, run_info_dir, run_obs_dir = create_folders()

    trajectory_count = 1
    frame_count = 16
    
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
                for pitch in np.arange(-90, 60, 5)
                for yaw in np.arange(-90, 90, 5)
        ]
    )


    obs_init = env.reset()
    for k in range(4):
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])

    env.spawn_mobs("cow", [4, 1, 8])

    env.set_block("diamond_ore", [1, 0, 5])
    env.set_block("diamond_ore", [1, 1, 5])
    env.set_block("diamond_ore", [0, 0, 5])
    env.set_block("diamond_ore", [0, 1, 5])
    env.set_block("diamond_ore", [-1, 0, 5])
    env.set_block("diamond_ore", [-1, 1, 5])

    for frame in range(frame_count):
        if frame == 15:
            _, _, _, _ = env.step([0,2,0,12,10,0,0,0])
            _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
            obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
            Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"0_0_{frame}.jpg"))
            continue
        _, _, _, _ = env.step([0,2,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    
        Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"0_0_{frame}.jpg"))

    env.close()