import json
import minedojo
import numpy as np
from PIL import Image
from os.path import join
from util import *
from env_data import *

if __name__ == '__main__':
    # Sampling parameters
    trajectory_count = 10
    frame_count = 15
    entity_counts = [2, 3, 4]
    N_MISS_REQUIRED = 3
    # Initialize Yaw/Pitch corrections parameters
    pitch_delta = 0
    yaw_delta = 0

    trajectory_actions = []
    # Voxel observation range
    vradius = 7
    for entity_count in entity_counts:

        # Create the directories for the samples images and textual observations
        paths = create_folders(sample_set="sample_set_1", entity_count=entity_count)
        
        rgb_dir_normal = paths["normal"]["rgb"]
        obs_dir_normal = paths["normal"]["obs"]
        info_dir_normal = paths["normal"]["info"]
        rgb_dir_occlusion  = paths["occluded"]["rgb"]
        obs_dir_occlusion  = paths["occluded"]["obs"]
        info_dir_occlusion  = paths["occluded"]["info"]

        for biome_id, biome in enumerate(biomes):
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
                        (np.pi * pitch / 180, np.pi * yaw / 180, 15)
                        for pitch in np.arange(-43, 44, 2) 
                        for yaw in np.arange(-52, 53, 2)
                ]
            )
            env.reset()
            # Start the trajectory
            valid_traj_count = 0
            valid_normal_traj_count = 0
            valid_occluded_traj_count = 0
            attempted_traj = 0
            while valid_normal_traj_count + valid_occluded_traj_count < trajectory_count:
                
                traj_rgb_frames = []
                traj_obs = []
                trajectory_actions = []

                attempted_traj += 1
                env.random_teleport(1000)
                if attempted_traj == 0:
                    obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    print(f'Starting Pitch and Yaw: {obs["location_stats"]["pitch"]}, {obs["location_stats"]["yaw"]}')
                
                # Wait till the rendering finishes
                obs_before, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])

                entities = sample_entities(biome, entity_count)

                last_seen     = {e: None  for e in entities}
                missing_count = {e: 0     for e in entities}
                occluded_entities: dict[str, list[int]] = {}
                occlusion_happened = False

                entities_spawn_locations = sample_entity_locations(entity_count)
                env.spawn_mobs(entities, entities_spawn_locations)

                # Wait till the entities drop to the ground
                for k in range(4):
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    # _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                obs_init, _, _, _ = env.step([0,0,0,12,12,0,0,0])

                # Validate that the agent is not under water
                if validate_not_under_water(obs_before, obs_init):
                    print(f"Skipping trajectory {attempted_traj}: underwater.")
                    continue
                # Validate all the spawned entities are visible
                is_visible, missing = validate_entities_visible(obs_init, entities)
                if not is_visible:
                    print(f"Skipped trajectory: entities not visible -> {missing}")
                    continue
                
                traj_rgb_frames.append(obs_init["rgb"].transpose(1, 2, 0))
                traj_obs.append(obs_init)
                #Image.fromarray(obs_init["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"{biome_id}_{valid_traj_count}_0.jpg"))
                #obs_to_json(obs_init, run_obs_dir, valid_traj_count, biome_id, "0")
                
                # Start sending actions
                for frame in range(frame_count):
                    random_action, pitch_delta, yaw_delta = random_action_sampler(pitch_delta, yaw_delta)
                    trajectory_actions.append(random_action)
                    print(random_action)
                    # Accounting for action delay by 2 steps
                    _, _, _, _ = env.step(random_action)
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])

                    for e in entities:
                        is_visible, _ = validate_entities_visible(obs, [e])
                        if is_visible:
                            missing_count[e] = 0
                            last_seen[e] = detect_entity_loc(obs, e)
                        else:
                            # Not seen this frame
                            missing_count[e] += 1
                            if missing_count[e] == 1:
                                # this is the first frame of the gap; store previous pose already held
                                pass
                            if missing_count[e] >= N_MISS_REQUIRED:
                                pose = last_seen[e]
                                if pose is not None and check_pose_in_fov(pose, obs):
                                    occlusion_happened = True
                                    occluded_entities.setdefault(e, []).append(frame+1)

                    traj_rgb_frames.append(obs["rgb"].transpose(1, 2, 0))
                    traj_obs.append(obs)
                    # Image.fromarray(obs["rgb"].transpose(1, 2, 0)).save(join(run_rgb_obs_dir, f"{biome_id}_{valid_traj_count}_{frame+1}.jpg"))
                    # obs_to_json(obs, run_obs_dir, valid_traj_count, biome_id, frame+1)

                print(f"Ending Trajectory: {valid_traj_count}")
                print(f'Pitch-Yaw before correction: {obs["location_stats"]["pitch"]}, {obs["location_stats"]["yaw"]}')
                print(f"Pitch Delta: {pitch_delta}")
                print(f"Yaw Delta: {yaw_delta}")

                pitch = 12 - pitch_delta
                yaw = 12 - yaw_delta
                camera_corrector_action = np.array([0,0,0,pitch,yaw,0,0,0])

                _, _, _, _ = env.step(camera_corrector_action)
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])

                pitch_delta = 0
                yaw_delta = 0
                print(f'Obs after correction: {obs["location_stats"]["pitch"]}, {obs["location_stats"]["yaw"]}')
                if (float(obs["location_stats"]["pitch"]) != 0.0) or (float(obs["location_stats"]["yaw"]) != 0.0):
                    print(f"Camera Correction Error at the End of Biome: {biome} Trajectory: {valid_traj_count}")

                #occluded_entities = list(set(occluded_entities))

                if not occlusion_happened:
                    for idx, (rgb, ob) in enumerate(zip(traj_rgb_frames, traj_obs)):
                        Image.fromarray(rgb).save(rgb_dir_normal / f"{biome_id}_{valid_normal_traj_count}_{idx}.jpg")
                        obs_to_json(ob, obs_dir_normal, biome_id, valid_normal_traj_count, idx)

                    with open(join(info_dir_normal / f"info_step_{biome_id}_{valid_normal_traj_count}.json"), "w") as f:
                        trajectory_info = dict()
                        trajectory_info["agent_location"] = str(obs_init["location_stats"]["pos"].tolist()) 
                        trajectory_info["entities_spawned"] = entities
                        trajectory_info["entities_spawn_locations"] = str([loc.tolist() for loc in entities_spawn_locations])
                        trajectory_info["actions"] = str([action.tolist() for action in trajectory_actions])
                        trajectory_info["occluded_entities"] = occluded_entities
                        json.dump(trajectory_info, f, indent=4)
                    valid_normal_traj_count += 1
                
                elif occluded_entities:
                    for idx, (rgb, ob) in enumerate(zip(traj_rgb_frames, traj_obs)):
                        Image.fromarray(rgb).save(rgb_dir_occlusion / f"{biome_id}_{valid_occluded_traj_count}_{idx}.jpg")
                        obs_to_json(ob, obs_dir_occlusion, biome_id, valid_occluded_traj_count, idx)

                    with open(join(info_dir_occlusion / f"info_step_{biome_id}_{valid_occluded_traj_count}.json"), "w") as f:
                        trajectory_info = dict()
                        trajectory_info["agent_location"] = str(obs_init["location_stats"]["pos"].tolist()) 
                        trajectory_info["entities_spawned"] = entities
                        trajectory_info["entities_spawn_locations"] = str([loc.tolist() for loc in entities_spawn_locations])
                        trajectory_info["actions"] = str([action.tolist() for action in trajectory_actions])
                        trajectory_info["occluded_entities"] = occluded_entities
                        json.dump(trajectory_info, f, indent=4)
                    valid_occluded_traj_count += 1
                
                valid_traj_count += 1
            env.close()