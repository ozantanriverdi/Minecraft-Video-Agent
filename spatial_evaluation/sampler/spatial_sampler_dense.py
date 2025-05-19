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
    frame_count = 32
    entity_counts = [2, 3, 4]
    # Occlusion test constants
    N_MISS_REQUIRED = 3
    YAW_TOLERANCE = 10
    # Initialize Yaw/Pitch corrections parameters
    pitch_delta = 0
    yaw_delta = 0
    # Voxel observation range
    vradius = 7
    
    for entity_count in entity_counts:

        # Create the directories for the samples images and textual observations
        paths = create_folders(sample_set="sample_set_demo", entity_count=entity_count)
        
        rgb_dir = paths["rgb"]
        obs_dir = paths["obs"]
        info_dir = paths["info"]

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
            # valid_normal_traj_count = 0
            # valid_occluded_traj_count = 0
            attempted_traj = 0
            while valid_traj_count < trajectory_count:
                
                traj_rgb_frames = []
                traj_obs = []
                trajectory_actions = []
                attempted_traj += 1

                env.random_teleport(1000)
                
                # Wait till the rendering finishes
                obs_before, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                _, _, _, _ = env.step([0,0,0,12,12,0,0,0])

                # Sample entity names/locations and spawn them
                entities = sample_entities(biome, entity_count)
                entities_spawn_locations = sample_entity_locations(entity_count)
                env.spawn_mobs(entities, entities_spawn_locations)

                # Save the last seen detected location for each entity
                last_seen_loc = {e: loc for e, loc in zip(entities, entities_spawn_locations)}
                # Save the yaw of the agent when an entity was last seen
                yaw_when_visible = {e: None for e in entities}
                
                missing_count = {e: 0     for e in entities}
                occluded_entities: dict[str, list[int]] = {}
                frame_visibility = {frame_id: [] for frame_id in range(frame_count)}
                entity_visibility: dict[str, list[int]] = {}

                # Wait till the entities drop to the ground
                for k in range(4):
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                obs_init, _, _, _ = env.step([0,0,0,12,12,0,0,0])

                # Validate that the agent is not under water
                if validate_not_under_water(obs_before, obs_init):
                    print(f"Skipping trajectory {attempted_traj}: underwater.")
                    continue

                for e in entities:
                    is_visible, _ = validate_entities_visible(obs_init, [e])
                    if is_visible:
                        entity_visibility.setdefault(e, []).append(0)
                        frame_visibility[0].append(e)
                        last_seen_loc[e] = detect_entity_loc(obs_init, e)
                        yaw_when_visible[e] = obs_init["location_stats"]["yaw"]
                    else:
                        missing_count[e] += 1
                
                traj_rgb_frames.append(obs_init["rgb"].transpose(1, 2, 0))
                traj_obs.append(obs_init)
                
                # Start sending actions
                for frame in range(frame_count-1):
                    random_action, pitch_delta, yaw_delta = random_action_sampler(pitch_delta, yaw_delta)
                    trajectory_actions.append(random_action)
                    print(random_action)
                    # Accounting for action delay by 2 steps
                    _, _, _, _ = env.step(random_action)
                    _, _, _, _ = env.step([0,0,0,12,12,0,0,0])
                    obs, _, _, _ = env.step([0,0,0,12,12,0,0,0])

                    for e in entities:
                        # Check if entity is visible
                        is_visible, _ = validate_entities_visible(obs, [e])
                        if is_visible:
                            # Log entity is visible
                            entity_visibility.setdefault(e, []).append(frame+1)
                            frame_visibility[frame+1].append(e)
                            # Set consecutive missing frame count to 0
                            missing_count[e] = 0
                            # Get the last detected location of the entity
                            last_seen_loc[e] = detect_entity_loc(obs, e)
                            # Get the yaw of the agent when the entity was last visible
                            yaw_when_visible[e] = obs["location_stats"]["yaw"]
                        else:
                            # Not seen this frame
                            missing_count[e] += 1
                            if missing_count[e] >= N_MISS_REQUIRED:
                                yaw_now = obs["location_stats"]["yaw"]
                                yaw_prev = yaw_when_visible[e]

                                delta_ok = abs(wrap_to_pi(yaw_now - yaw_prev)) < YAW_TOLERANCE \
                                    if yaw_prev is not None else True

                                if delta_ok and check_pose_in_fov(last_seen_loc[e], obs):
                                    occluded_entities.setdefault(e, []).append(frame+1)

                    traj_rgb_frames.append(obs["rgb"].transpose(1, 2, 0))
                    traj_obs.append(obs)

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

                for idx, (rgb, ob) in enumerate(zip(traj_rgb_frames, traj_obs)):
                    Image.fromarray(rgb).save(rgb_dir / f"{biome_id}_{valid_traj_count}_{idx}.jpg")
                    obs_to_json(ob, obs_dir, biome_id, valid_traj_count, idx)

                with open(join(info_dir / f"info_step_{biome_id}_{valid_traj_count}.json"), "w") as f:
                    trajectory_info = dict()
                    trajectory_info["agent_location"] = str(obs_init["location_stats"]["pos"].tolist()) 
                    trajectory_info["entities_spawned"] = entities
                    trajectory_info["entities_spawn_locations"] = str([loc.tolist() for loc in entities_spawn_locations])
                    trajectory_info["actions"] = str([action.tolist() for action in trajectory_actions])
                    trajectory_info["occluded_entities"] = occluded_entities
                    trajectory_info["frame_visibility"] = frame_visibility
                    trajectory_info["entity_visibility"] = entity_visibility
                    json.dump(trajectory_info, f, indent=4)
 
                valid_traj_count += 1
            env.close()