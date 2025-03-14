import os
import json
import ast
import json
import ast
import base64
import re
import numpy as np
import math
from os.path import join
from pathlib import Path
from tqdm import tqdm
from .utils import *


class GroundTruthExtractor:
    def __init__(self, run_id, biomes_count=10, trajectories_count=20, frames_count=16):
        self.run_id = run_id

        base_dir = Path(__file__).parent.parent.parent
        samples_dir = base_dir / "samples"
        self.run_rgb_obs_dir = samples_dir / "rgb_frames" / run_id
        self.run_obs_dir = samples_dir / "obs" / run_id
        self.run_info_dir = samples_dir / "info" / run_id

        self.ground_truth_dir = create_folders(self.run_id)

        self.biomes_count = biomes_count
        self.trajectories_count = trajectories_count
        self.frames_count = frames_count

        num_obs = len([f for f in os.listdir(self.run_obs_dir) if os.path.isfile(join(self.run_obs_dir, f))])
        num_rgb = len([f for f in os.listdir(self.run_rgb_obs_dir) if os.path.isfile(join(self.run_rgb_obs_dir, f))])

        correct_obs_count = self.biomes_count * self.trajectories_count * self.frames_count

        if num_rgb != correct_obs_count or num_obs != correct_obs_count:
            raise Exception("GroundTruthExtractor: Mismatch between the number of obs and rgb files detected!")
        else:
            print(f"GroundTruthExtractor: Correct number of obs files found: {num_obs}")

        self.filtered_frames = {"biome": [], "trajectory": [], "frame": []}


    def filter_trajectories(self):
        for biome in tqdm(range(self.biomes_count)):
            for trajectory in range(self.trajectories_count):
                
                with open(join(self.run_info_dir, f"info_step_{biome}_{trajectory}.json"), "r") as f:
                    trajectory_info = json.load(f)
                
                entities_spawned = trajectory_info["entities_spawned"]

                for frame in range(self.frames_count):

                    with open(join(self.run_obs_dir, f"obs_step_{biome}_{trajectory}_{frame}.json"), "r") as f:
                        obs = json.load(f)
                    entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
                    entity_1_idx = np.where(entities == entities_spawned[0])[0]
                    entity_2_idx = np.where(entities == entities_spawned[1])[0]
                    if len(entity_1_idx) > 0 and len(entity_2_idx) > 0:
                        self.filtered_frames["biome"].append(biome)
                        self.filtered_frames["trajectory"].append(trajectory)
                        self.filtered_frames["frame"].append(frame)
        return self.filtered_frames


    def extract_absolute_distances(self):
        ground_truths = {}

        for biome, trajectory, frame in zip(self.filtered_frames["biome"], 
                                            self.filtered_frames["trajectory"], 
                                            self.filtered_frames["frame"]):

            with open(join(self.run_info_dir, f"info_step_{biome}_{trajectory}.json"), "r") as f:
                trajectory_info = json.load(f)
            
            entities_spawned = trajectory_info["entities_spawned"]

            with open(join(self.run_obs_dir, f"obs_step_{biome}_{trajectory}_{frame}.json"), "r") as f:
                obs = json.load(f)
            entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
            distances = np.array(ast.literal_eval(obs["rays"]["entity_distance"]))
            entity_idx_0 = np.where(entities == entities_spawned[0])[0]
            entity_idx_1 = np.where(entities == entities_spawned[1])[0]
            
            entity_0_distance = np.min(distances[entity_idx_0])
            entity_0_distance = round(entity_0_distance, 3)
            entity_1_distance = np.min(distances[entity_idx_1])
            entity_1_distance = round(entity_1_distance, 3)

            if biome not in ground_truths:
                ground_truths[biome] = {}
            if trajectory not in ground_truths[biome]:
                ground_truths[biome][trajectory] = {}
            
            ground_truths[biome][trajectory][frame] = [entity_0_distance, entity_1_distance]

        with open(join(self.ground_truth_dir, "absolute_distance.json"), "w") as f:
            json.dump(ground_truths, f, indent=4)

        return ground_truths


    def extract_relative_distances(self):
        ground_truths = {}

        for biome, trajectory, frame in zip(self.filtered_frames["biome"], 
                                            self.filtered_frames["trajectory"], 
                                            self.filtered_frames["frame"]):

            with open(join(self.run_info_dir, f"info_step_{biome}_{trajectory}.json"), "r") as f:
                trajectory_info = json.load(f)
            
            entities_spawned = trajectory_info["entities_spawned"]

            with open(join(self.run_obs_dir, f"obs_step_{biome}_{trajectory}_{frame}.json"), "r") as f:
                obs = json.load(f)
            
            entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
            traced_block_x = np.array(ast.literal_eval(obs["rays"]["traced_block_x"]))
            traced_block_y = np.array(ast.literal_eval(obs["rays"]["traced_block_y"]))
            traced_block_z = np.array(ast.literal_eval(obs["rays"]["traced_block_z"]))
            entity_1_idx = np.where(entities == entities_spawned[0])[0]
            entity_2_idx = np.where(entities == entities_spawned[1])[0]
            relative_distance = calculate_distance(traced_block_x, traced_block_y, traced_block_z, entity_1_idx, entity_2_idx)

            if biome not in ground_truths:
                ground_truths[biome] = {}
            if trajectory not in ground_truths[biome]:
                ground_truths[biome][trajectory] = {}
            
            ground_truths[biome][trajectory][frame] = relative_distance

        with open(join(self.ground_truth_dir, "relative_distance.json"), "w") as f:
            json.dump(ground_truths, f, indent=4)

        return ground_truths


    def extract_relative_directions(self):
        """
        Relative direction of 'entity_2' based on 'entity_1'
        left, right, in front, back, front-left, front-right, back-left, back-right
        """
        ground_truths = {}

        for biome, trajectory, frame in zip(self.filtered_frames["biome"], 
                                            self.filtered_frames["trajectory"], 
                                            self.filtered_frames["frame"]):

            with open(join(self.run_info_dir, f"info_step_{biome}_{trajectory}.json"), "r") as f:
                trajectory_info = json.load(f)
            
            entities_spawned = trajectory_info["entities_spawned"]

            with open(join(self.run_obs_dir, f"obs_step_{biome}_{trajectory}_{frame}.json"), "r") as f:
                obs = json.load(f)
            
            entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
            traced_block_x = np.array(ast.literal_eval(obs["rays"]["traced_block_x"]))
            traced_block_y = np.array(ast.literal_eval(obs["rays"]["traced_block_y"]))
            traced_block_z = np.array(ast.literal_eval(obs["rays"]["traced_block_z"]))
            entity_1_idx = np.where(entities == entities_spawned[0])[0]
            entity_2_idx = np.where(entities == entities_spawned[1])[0]
        
            #entity_1_coord = list(zip(traced_block_x[entity_1_idx], traced_block_y[entity_1_idx], traced_block_z[entity_1_idx]))
            #entity_2_coord = list(zip(traced_block_x[entity_2_idx], traced_block_y[entity_2_idx], traced_block_z[entity_2_idx]))

            x1, y1, z1 = np.min(traced_block_x[entity_1_idx]), np.min(traced_block_y[entity_1_idx]), np.min(traced_block_z[entity_1_idx])
            x2, y2, z2 = np.min(traced_block_x[entity_2_idx]), np.min(traced_block_y[entity_2_idx]), np.min(traced_block_z[entity_2_idx])
        
            # Compute direction differences
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            # Determine horizontal direction
            if dx == 0 and dz == 0:
                horizontal_direction = "same position"
            elif dx == 0:
                horizontal_direction = "in front" if dz > 0 else "back"
            elif dz == 0:
                horizontal_direction = "left" if dx > 0 else "right"
            elif dz > 0:  # Entity_2 is in front
                horizontal_direction = "back-left" if dx > 0 else "back-right"
            else:  # Entity_2 is behind
                horizontal_direction = "front-left" if dx > 0 else "front-right"

            # Determine vertical direction
            if dy > 0:
                vertical_direction = "above"
            elif dy < 0:
                vertical_direction = "below"
            else:
                vertical_direction = "same level"

            if biome not in ground_truths:
                ground_truths[biome] = {}
            if trajectory not in ground_truths[biome]:
                ground_truths[biome][trajectory] = {}
            
            ground_truths[biome][trajectory][frame] = f"{horizontal_direction}-{vertical_direction}"

        with open(join(self.ground_truth_dir, "relative_direction.json"), "w") as f:
            json.dump(ground_truths, f, indent=4)

        return ground_truths
    
    def extract_ground_truths(self):
        self.extract_absolute_distances()
        self.extract_relative_distances()
        self.extract_relative_directions()





if __name__ == '__main__':
    extractor = GroundTruthExtractor("20250304_000717", biomes_count=2, trajectories_count=5, frames_count=16)
    extractor.filter_trajectories()
    print(extractor.filtered_frames)
    extractor.extract_absolute_distances()
    extractor.extract_relative_distances()
    extractor.extract_relative_directions()

