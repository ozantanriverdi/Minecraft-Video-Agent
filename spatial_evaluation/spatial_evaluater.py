import os
import json
import ast
import math
import numpy as np
from os.path import join
from utils import calculate_distance


class Evaluator:
    def __init__(self, sample_id):
        self.sample_id = sample_id
        cwd = os.getcwd()
        obs_dir = join(cwd, "obs")
        self.directory = join(obs_dir, sample_id)
    
    def evaluate_absolute_distance(self, step, entity_name):
        with open(join(self.directory, f"obs_step_{step}.json"), "r") as f:
            obs = json.load(f)
        entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
        distances = np.array(ast.literal_eval(obs["rays"]["entity_distance"]))
        entity_idx = np.where(entities == entity_name)[0]
        #print(distances[entity_idx])
        if len(entity_idx) > 0:
            entity_distance = np.min(distances[entity_idx])
            entity_distance = round(entity_distance, 2)

        
        # x, y, z = [], [], []
        # x_obs = np.array(ast.literal_eval(obs["rays"]["traced_block_x"]))
        # y_obs = np.array(ast.literal_eval(obs["rays"]["traced_block_y"]))
        # z_obs = np.array(ast.literal_eval(obs["rays"]["traced_block_z"]))
        # for i in entity_idx:
        #     x.append(x_obs[i])
        #     y.append(y_obs[i])
        #     z.append(z_obs[i])
        #     print(f"({x_obs[i]}, {y_obs[i]}, {z_obs[i]})")
        # my_loc = np.array(ast.literal_eval(obs["location_stats"]["pos"]))

        # distances = []
        # for i in range(len(entity_idx)):
        #     distance = math.sqrt((my_loc[0] - x[i]) ** 2 + (my_loc[1] - y[i]) ** 2 + (my_loc[2] - z[i]) ** 2)
        #     distances.append(distance)

        # print(entity_idx, entity_distance, x, y, z, my_loc, distances)

        return entity_distance



    def evaluate_relative_distance(self, step, entity_names):
        with open(join(self.directory, f"obs_step_{step}.json"), "r") as f:
            obs = json.load(f)
        entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
        traced_block_x = np.array(ast.literal_eval(obs["rays"]["traced_block_x"]))
        traced_block_y = np.array(ast.literal_eval(obs["rays"]["traced_block_y"]))
        traced_block_z = np.array(ast.literal_eval(obs["rays"]["traced_block_z"]))
        entity_1_idx = np.where(entities == entity_names[0])[0]
        entity_2_idx = np.where(entities == entity_names[1])[0]
        if len(entity_1_idx) > 0 and len(entity_2_idx) > 0:
            distances = calculate_distance(traced_block_x, traced_block_y, traced_block_z, entity_1_idx, entity_2_idx)
            entities_relative_distance = np.min(distances)
            entities_relative_distance = round(entities_relative_distance, 2)
        
        return entities_relative_distance
        

    def evaluate_relative_direction(self, step, entity_1, entity_2):
        """
        Relative direction of 'entity_2' based on 'entity_1'
        left, right, in front, back, front-left, front-right, back-left, back-right
        """
        with open(join(self.directory, f"obs_step_{step}.json"), "r") as f:
            obs = json.load(f)
        entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
        traced_block_x = np.array(ast.literal_eval(obs["rays"]["traced_block_x"]))
        traced_block_y = np.array(ast.literal_eval(obs["rays"]["traced_block_y"]))
        traced_block_z = np.array(ast.literal_eval(obs["rays"]["traced_block_z"]))
        entity_1_idx = np.where(entities == entity_1)[0]
        entity_2_idx = np.where(entities == entity_2)[0]

        entity_1_coord = list(zip(traced_block_x[entity_1_idx], traced_block_y[entity_1_idx], traced_block_z[entity_1_idx]))
        entity_2_coord = list(zip(traced_block_x[entity_2_idx], traced_block_y[entity_2_idx], traced_block_z[entity_2_idx]))
        #print(entity_1_coord)
        #print(entity_2_coord)

        x1, z1 = np.min(traced_block_x[entity_1_idx]), np.min(traced_block_z[entity_1_idx])
        x2, z2 = np.min(traced_block_x[entity_2_idx]), np.min(traced_block_z[entity_2_idx])

        
        dx = x2 - x1
        dz = z2 - z1

        # TODO: Can add a buffer as absolute values like 'dx == 0' -> 'abs(dx) < 1'
        if dx == 0:
            return "in front" if dz > 0 else "back"
        elif dz == 0:
            return "left" if dx > 0 else "right"
        elif dz > 0:  # Entity_2 is in front
            return "back-left" if dx > 0 else "back-right"
        else:  # Entity_2 is behind
            return "front-left" if dx > 0 else "front-right"



if __name__ == '__main__':
    evaluator = Evaluator(sample_id="20250221_000412")
    print(evaluator.evaluate_absolute_distance(step=0, entity_name="horse"))
    #print(evaluator.evaluate_relative_distance(step=0, entity_names=["horse", "pig"]))
    #print(evaluator.evaluate_relative_direction(step=2, entity_1="pig", entity_2="horse"))
