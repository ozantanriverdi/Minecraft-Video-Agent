import numpy as np
import minedojo
import os
import datetime
from os.path import join
from PIL import Image
from minedojo.sim import InventoryItem


def main ():
    cwd = os.getcwd()
    samples_dir = join(cwd, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    vradius = 5
    seed = 17

    env = minedojo.make(
        task_id="harvest", target_names="diamond",
        image_size=(512, 820),
        target_quantities=100, seed = 1,
        specified_biome = "plains",
        break_speed_multiplier = 100.0,
        world_seed = seed,
        use_voxel = True,
        voxel_size=dict(xmin=-vradius, ymin=-vradius, zmin=-vradius, xmax=vradius, ymax=vradius, zmax=vradius),
        use_lidar=True,
        lidar_rays=[
                (np.pi * pitch / 180, np.pi * yaw / 180, 10) # ALERT: lidar range is now 10
                for pitch in np.arange(-60, 60, 5)
                for yaw in np.arange(-60, 60, 5)
        ],
        start_position= {"x": 190.5, "y": 69, "z": 248.5, "pitch": 0, "yaw": 0},
        initial_weather= "clear"
    )

    env.reset()
    env.spawn_mobs("sheep", [0, 0, 4])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    file_path = join(samples_dir, "sheep.jpg")
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(file_path)

    env.spawn_mobs("horse", [4, 0, 0])
    _, _, _, _ = env.step([0,0,0,12,6,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    file_path = join(samples_dir, "horse.jpg")
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(file_path)

    env.spawn_mobs("cow", [0, 0, -4])
    _, _, _, _ = env.step([0,0,0,12,6,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    file_path = join(samples_dir, "cow.jpg")
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(file_path)

    env.spawn_mobs("pig", [-4, 0, 0])
    _, _, _, _ = env.step([0,0,0,12,6,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
    file_path = join(samples_dir, "pig.jpg")
    Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(file_path)

if __name__ == '__main__':
    main()