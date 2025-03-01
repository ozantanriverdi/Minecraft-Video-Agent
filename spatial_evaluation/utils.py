import os
import json
import ast
import base64
import re
import numpy as np
import math
import matplotlib.pyplot as plt
from os.path import join


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def parse_llm_output_distance(response):
    match = re.search(r"```json\s*([\s\S]*?)```", response) or re.search(r"{.*}", response)

    if match:
        json_str = match.group(1)  # Extract JSON content
        try:
            parsed_json = json.loads(json_str)  # Parse JSON
            return parsed_json.get("distance", None)  # Get 'distance' value
        except json.JSONDecodeError:
            return None  # Return None if JSON is malformed
    return None  # Return None if no JSON found

def parse_llm_output_direction(response):
    match = re.search(r"```*([\s\S]*?)```", response) or re.search(r"{.*}", response)

    if match:
        json_str = match.group(1)  # Extract JSON content
        try:
            parsed_json = json.loads(json_str)  # Parse JSON
            return parsed_json.get("direction", None)  # Get 'distance' value
        except json.JSONDecodeError:
            return None  # Return None if JSON is malformed
    return None  # Return None if no JSON found
    
def filter_runs(sample_id):
    """
    Filter the indices which have entities 'horse' and 'pig' in their observations
    """
    cwd = os.getcwd()
    obs_dir = join(cwd, "obs")
    directory = join(obs_dir, sample_id)
    num_files = len([f for f in os.listdir(directory) if os.path.isfile(join(directory, f))])
    
    filtered_steps = []
    for i in range(num_files):
        with open(join(directory, f"obs_step_{i}.json"), "r") as f:
            obs = json.load(f)
        entities = np.array(ast.literal_eval(obs["rays"]["entity_name"]))
        #entities, distances = obs["rays"]["entity_name"], obs["rays"]["entity_distance"]
        horse_idx = np.where(entities == "horse")[0]
        pig_idx = np.where(entities == "pig")[0]
        if len(horse_idx) > 0 and len(pig_idx) > 0:
            filtered_steps.append(i)
    
    return filtered_steps

def calculate_distance(traced_block_x, traced_block_y, traced_block_z, entity_1_idx, entity_2_idx):
    distances = []
    for entity_1_id in entity_1_idx:
        for entity_2_id in entity_2_idx:
            distance = math.sqrt((traced_block_x[entity_1_id] - traced_block_x[entity_2_id]) ** 2 +
                                    (traced_block_y[entity_1_id] - traced_block_y[entity_2_id]) ** 2 +
                                    (traced_block_z[entity_1_id] - traced_block_z[entity_2_id]) ** 2)
            distances.append(distance)
    return distances

def distance_metric(predicted_distances, ground_truth_distances):
    """
    Calculates the Mean Absolute Error (MAE) between two lists of distances
    """
    if len(predicted_distances) != len(ground_truth_distances):
        raise ValueError("Both lists must have the same length.")
    mae = np.mean(np.abs(np.array(predicted_distances) - np.array(ground_truth_distances)))
    return round(mae, 2)

def direction_metric(predicted_directions, ground_truth_directions):
    """
    Calculates the accuracy of predicted directions compared to ground truth.
    """
    if len(predicted_directions) != len(ground_truth_directions):
        raise ValueError("Both lists must have the same length.")
    correct_predictions = sum(1 for pred, gt in zip(predicted_directions, ground_truth_directions) if pred == gt)
    total_predictions = len(predicted_directions)
    accuracy = (correct_predictions / total_predictions) * 100
    return round(accuracy, 2)

def save_results(sample_id, task, metric, sample_size, predictions, ground_truth_distances):
    if os.path.exists("results.json"):
        with open("results.json", "r") as f:
            results = json.load(f)
    else:
        results = {}
    
    try:
        last_run_id = int(list(results.keys())[-1])

        results[last_run_id+1] = {
            "sample_id": sample_id,
            "task": task,
            "metric": metric,
            "sample_size": sample_size,
            "predictions": predictions,
            "ground_truth_distances": ground_truth_distances
        }
    except:
        results[0] = {
            "sample_id": sample_id,
            "task": task,
            "metric": metric,
            "sample_size": sample_size,
            "predictions": predictions,
            "ground_truth_distances": ground_truth_distances
        }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

def plot_results(*run_ids):
    with open("results.json", "r") as f:
        results = json.load(f)
    
    experiments = []
    metrics = []
    for run_id in run_ids:
        experiments.append(f"{run_id} - {results[run_id]['task']} - {results[run_id]['sample_size']}")
        metrics.append(results[run_id]["metric"])

    plt.figure(figsize=(8, 5))
    plt.bar(experiments, metrics, color='skyblue')

    plt.xlabel("Experiment Run")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Comparison per Experiment")
    #plt.ylim(0.0, 1.0)
    plt.grid(axis="y")
    plt.show()

if __name__ == '__main__':
    print(filter_runs("20250221_000412"))
    plot_results("0")
