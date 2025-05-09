import json
import time
import minedojo
import numpy as np
from PIL import Image
from os.path import join

from util import *
from env_data import *

if __name__ == '__main__':

    vradius = 5

    env = minedojo.make(
    "open-ended",
    image_size=(480, 768),
    generate_world_type="flat",
    flat_world_seed_string="1;7,2x3,2,6:5;1",
    use_voxel=True,
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
    env.set_block("diamond_ore", [-2, 0, 7])
    #env.set_block("diamond_ore", [-2, 1, 3])
    env.set_block("diamond_ore", [1, 0, 3])
    env.set_block("diamond_ore", [1, 1, 3])
    env.spawn_mobs("cow", [-2, 1, 7])
    env.spawn_mobs("horse", [1, 2, 3])
    
    for k in range(4):
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
        obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save("1.jpg")
    env.close()
    # with open("obs_init.json", "w") as f:
    #     json.dump(f, obs_init)