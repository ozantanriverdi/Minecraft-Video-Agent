import json
import time
import minedojo
import numpy as np
from PIL import Image
from os.path import join
from util import *
from env_data import *


if __name__ == '__main__':
    run_rgb_obs_dir, run_info_dir, run_obs_dir = create_folders()
    trajectory_count = 20
    frame_count = 15

    pitch_delta = 0
    yaw_delta = 0
    trajectory_actions = []

    vradius = 5

    # Start Episode
    for biome_id, biome in enumerate(biomes):
        # env = minedojo.make(
        #     "open-ended",
        #     image_size=(480, 768),
        #     generate_world_type="flat",
        #     flat_world_seed_string="",
        #     use_voxel = True,
        #     voxel_size=dict(xmin=-vradius, ymin=-vradius, zmin=-vradius, xmax=vradius, ymax=vradius, zmax=vradius),
        #     use_lidar=True,
        #     lidar_rays=[
        #             (np.pi * pitch / 180, np.pi * yaw / 180, 12) # ALERT: lidar range is now 5
        #             for pitch in np.arange(-90, 60, 5)
        #             for yaw in np.arange(-90, 90, 5)
        #     ]
        # )
        env = minedojo.make(
            "open-ended",
            image_size=(480, 768),
            generate_world_type="specified_biome",
            specified_biome=biome,
            allow_mob_spawn = False,
            use_voxel=True,
            start_time = 6000,
            initial_weather="clear",
            voxel_size=dict(xmin=-vradius, ymin=-vradius, zmin=-vradius, xmax=vradius, ymax=vradius, zmax=vradius),
            use_lidar=True,
            lidar_rays=[
                    (np.pi * pitch / 180, np.pi * yaw / 180, 12) # ALERT: lidar range is now 5
                    for pitch in np.arange(-90, 60, 5)
                    for yaw in np.arange(-90, 90, 5)
            ]
        )
        env.reset()
        
        for trajectory in range(trajectory_count):

            env.random_teleport(1000)
            if trajectory == 0:
                obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                print(f'Starting Pitch and Yaw: {obs["location_stats"]["pitch"]}, {obs["location_stats"]["yaw"]}')
            
            # Wait till the rendering finishes
            _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
            _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
            _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
            _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
            
            entities = sample_entities(biome, 2)
            entities_spawn_locations = entity_random_location_simple()
            env.spawn_mobs(entities, entities_spawn_locations)

            # Wait till the entities drop to the ground
            for k in range(4):
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
            obs_init, _, _, _ = env.step([0,0,0,12,12,0,0,0])
            Image.fromarray(obs_init["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"{biome_id}_{trajectory}_0.jpg"))
            obs_to_json(obs_init, run_obs_dir, trajectory, biome_id, "0")

            # Start Trajectory
            for frame in range(frame_count):

                random_action, pitch_delta, yaw_delta = random_action_sampler(pitch_delta, yaw_delta)
                trajectory_actions.append(random_action)
                print(random_action)
                # Accounting for action delay by 2 steps
                _, _, _, _ = env.step(random_action)
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])

                Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"{biome_id}_{trajectory}_{frame+1}.jpg"))
                obs_to_json(obs, run_obs_dir, trajectory, biome_id, frame+1)

            print(f"Ending Trajectory: {trajectory}")
            print(f'Pitch-Yaw before correction: {obs["location_stats"]["pitch"]}, {obs["location_stats"]["yaw"]}')
            print(f"Pitch Delta: {pitch_delta}")
            print(f"Yaw Delta: {yaw_delta}")

            # TODO: Can use modulo here
            pitch = 12 - pitch_delta
            yaw = 12 - yaw_delta
            # print(pitch)
            # print(yaw)
            camera_corrector_action = np.array([0,0,0,pitch,yaw,0,0,0])

            # Accounting for action delay by 2 steps
            _, _, _, _ = env.step(camera_corrector_action)
            _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
            obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])

            pitch_delta = 0
            yaw_delta = 0
            print(f'Obs after correction: {obs["location_stats"]["pitch"]}, {obs["location_stats"]["yaw"]}')
            if (float(obs["location_stats"]["pitch"]) != 0.0) or (float(obs["location_stats"]["yaw"]) != 0.0):
                print(f"Camera Correction Error at the End of Biome: {biome} Trajectory: {trajectory}")
            
            
            
            with open(join(run_info_dir, f"info_step_{biome_id}_{trajectory}.json"), "w") as f:
                trajectory_info = dict()
                trajectory_info["agent_location"] = str(obs_init["location_stats"]["pos"].tolist()) 
                trajectory_info["entities_spawned"] = entities
                trajectory_info["entities_spawn_locations"] = str([loc.tolist() for loc in entities_spawn_locations])
                trajectory_info["actions"] = str([action.tolist() for action in trajectory_actions])
                json.dump(trajectory_info, f, indent=4)
            
            trajectory_actions = []

        env.close()

