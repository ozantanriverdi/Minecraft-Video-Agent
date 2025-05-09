import os
import json
import ast
import math
import numpy as np
from itertools import combinations
from scipy.stats import mode
from os.path import join
from pathlib import Path

class GroundTruthExtractor:
    def __init__(self, run_id, biomes_count=10, trajectories_count=20, frames_count=16):
        self.run_id = run_id

        base_dir = Path(__file__).parent.parent.parent
        samples_dir = base_dir / "samples"
        self.run_rgb_obs_dir = samples_dir / "rgb_frames" / run_id
        self.run_obs_dir = samples_dir / "obs" / run_id
        self.run_info_dir = samples_dir / "info" / run_id

        self.ground_truths_dir = self._create_gt_dirs()

        self.biomes_count = biomes_count
        self.trajectories_count = trajectories_count
        self.frames_count = frames_count

    def _create_gt_dirs(self):
        base_dir = Path(__file__).parent.parent.parent
        ground_truths_dir = base_dir / "ground_truths" / self.run_id
        ground_truths_dir.mkdir(parents=True, exist_ok=True)
        return ground_truths_dir
    
    def _calculate_distance(self, traced_block_x, traced_block_y, traced_block_z, entity_1_idx, entity_2_idx):
        distances = []
        for entity_1_id in entity_1_idx:
            for entity_2_id in entity_2_idx:
                distance = math.sqrt((traced_block_x[entity_1_id] - traced_block_x[entity_2_id]) ** 2 +
                                        (traced_block_y[entity_1_id] - traced_block_y[entity_2_id]) ** 2 +
                                        (traced_block_z[entity_1_id] - traced_block_z[entity_2_id]) ** 2)
                distances.append(distance)
        min_distance = np.min(distances)
        min_distance = round(min_distance, 3)
        return min_distance
    
    def _get_mode_coords(self, x, y, z, indices):
        if len(indices) == 0:
            return None
        coords = np.stack([x[indices], y[indices], z[indices]], axis=1)
        coord_tuples = [tuple(coord) for coord in coords]
        mode_coord, _ = mode(coord_tuples, axis=0, keepdims=False)
        return mode_coord

    def _distance_to_camera(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    def extract_absolute_distances(self):
        ground_truths = {}

        for biome in range(self.biomes_count):
            for trajectory in range(self.trajectories_count):
                for frame in range(self.frames_count):
                    with open(join(self.run_info_dir, f"info_step_{biome}_{trajectory}.json"), "r") as f:
                        trajectory_info = json.load(f)
                    entities_spawned = trajectory_info["entities_spawned"]

                    with open(join(self.run_obs_dir, f"obs_step_{biome}_{trajectory}_{frame}.json"), "r") as f:
                        obs = json.load(f)

                    entity_distances = {}
                    entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
                    distances = np.array(ast.literal_eval(obs["rays"]["entity_distance"]))
                    for entity_name in entities_spawned:
                        entity_idxs = np.where(entities == entity_name)[0]

                        if len(entity_idxs) == 0:
                            entity_distances[entity_name] = None  # or np.nan
                        else:
                            min_dist = np.min(distances[entity_idxs])
                            entity_distances[entity_name] = round(min_dist, 3)

                    if biome not in ground_truths:
                        ground_truths[biome] = {}
                    if trajectory not in ground_truths[biome]:
                        ground_truths[biome][trajectory] = {}
                    
                    ground_truths[biome][trajectory][frame] = entity_distances
        
        with open(join(self.ground_truths_dir, "absolute_distance.json"), "w") as f:
            json.dump(ground_truths, f, indent=4)
        
        return ground_truths
    
    def extract_relative_distances(self):
        ground_truths = {}
        
        for biome in range(self.biomes_count):
            for trajectory in range(self.trajectories_count):
                for frame in range(self.frames_count):
                    with open(join(self.run_info_dir, f"info_step_{biome}_{trajectory}.json"), "r") as f:
                        trajectory_info = json.load(f)
                    entities_spawned = trajectory_info["entities_spawned"]

                    with open(join(self.run_obs_dir, f"obs_step_{biome}_{trajectory}_{frame}.json"), "r") as f:
                        obs = json.load(f)

                    entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
                    traced_block_x = np.array(ast.literal_eval(obs["rays"]["traced_block_x"]))
                    traced_block_y = np.array(ast.literal_eval(obs["rays"]["traced_block_y"]))
                    traced_block_z = np.array(ast.literal_eval(obs["rays"]["traced_block_z"]))

                    relative_distances = {}
                    for entity_a, entity_b in combinations(entities_spawned, 2):
                        idx_a = np.where(entities == entity_a)[0]
                        idx_b = np.where(entities == entity_b)[0]

                        if len(idx_a) == 0 or len(idx_b) == 0:
                            continue  # or mark as None

                        dist = self._calculate_distance(traced_block_x, traced_block_y, traced_block_z, idx_a, idx_b)
                        if dist is not None:
                            # Add both directions for easy access
                            if entity_a not in relative_distances:
                                relative_distances[entity_a] = {}
                            if entity_b not in relative_distances:
                                relative_distances[entity_b] = {}

                            relative_distances[entity_a][entity_b] = dist
                            relative_distances[entity_b][entity_a] = dist

                    if biome not in ground_truths:
                        ground_truths[biome] = {}
                    if trajectory not in ground_truths[biome]:
                        ground_truths[biome][trajectory] = {}

                    ground_truths[biome][trajectory][frame] = relative_distances

        with open(join(self.ground_truths_dir, "relative_distance.json"), "w") as f:
            json.dump(ground_truths, f, indent=4)
        
        return ground_truths

    def extract_relative_directions(self):
        """
        Compute pairwise relative directions for all visible entities.
        Encodes direction from entity A to B as (dx, dy, dz):
        dx: -1=left, 0=centered, 1=right
        dy: -1=below, 0=same level, 1=above
        dz: -1=in front, 0=aligned, 1=behind (with dynamic threshold)
        """
        ground_truths = {}
        
        for biome in range(self.biomes_count):
            for trajectory in range(self.trajectories_count):
                for frame in range(self.frames_count):
            
                    with open(join(self.run_info_dir, f"info_step_{biome}_{trajectory}.json"), "r") as f:
                        trajectory_info = json.load(f)
                    
                    entities_spawned = trajectory_info["entities_spawned"]

                    with open(join(self.run_obs_dir, f"obs_step_{biome}_{trajectory}_{frame}.json"), "r") as f:
                        obs = json.load(f)

                    agent_pos = np.array(ast.literal_eval(obs["location_stats"]["pos"]))

                    entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
                    x = np.array(ast.literal_eval(obs["rays"]["traced_block_x"]))
                    y = np.array(ast.literal_eval(obs["rays"]["traced_block_y"]))
                    z = np.array(ast.literal_eval(obs["rays"]["traced_block_z"]))

                    rel_directions = {}

                    for ent_a, ent_b in combinations(entities_spawned, 2):
                        idx_a = np.where(entities == ent_a)[0]
                        idx_b = np.where(entities == ent_b)[0]

                        if len(idx_a) == 0 or len(idx_b) == 0:
                            continue  # Skip if either entity not visible

                        coord_a = self._get_mode_coords(x, y, z, idx_a)
                        coord_b = self._get_mode_coords(x, y, z, idx_b)

                        if coord_a is None or coord_b is None:
                            continue

                        x1, y1, z1 = coord_a
                        x2, y2, z2 = coord_b

                        dx = x2 - x1
                        dy = y2 - y1
                        dz = z2 - z1

                        direction = []

                        # X-axis (left/right)
                        if dx > 0:
                            direction.append(-1)  # B is left of A
                        elif dx < 0:
                            direction.append(1)  # B is right of A
                        else:
                            direction.append(0)

                        # Y-axis (above/below)
                        if dy > 0:
                            direction.append(1)  # B is above A
                        elif dy < 0:
                            direction.append(-1)  # B is below A
                        else:
                            direction.append(0)

                        # Z-axis (front/behind with tolerance)
                        depth_ref = min(abs(z1 - agent_pos[2]), abs(z2 - agent_pos[2]))
                        threshold = depth_ref * 0.2
                        if dz > threshold:
                            direction.append(1)  # B is behind A
                        elif dz < -threshold:
                            direction.append(-1)  # B is in front of A
                        else:
                            direction.append(0)

                        # Save to both A→B and B→A entries
                        for a, b, d in [(ent_a, ent_b, direction), (ent_b, ent_a, [-d for d in direction])]:
                            if a not in rel_directions:
                                rel_directions[a] = {}
                            rel_directions[a][b] = d

                    if biome not in ground_truths:
                        ground_truths[biome] = {}
                    if trajectory not in ground_truths[biome]:
                        ground_truths[biome][trajectory] = {}
                    
                    ground_truths[biome][trajectory][frame] = rel_directions

        with open(join(self.ground_truths_dir, "relative_direction.json"), "w") as f:
            json.dump(ground_truths, f, indent=4)

        return ground_truths
