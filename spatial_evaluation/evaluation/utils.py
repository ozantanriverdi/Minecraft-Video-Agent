import os
import json
import ast
import base64
import re
import numpy as np
import math
import matplotlib.pyplot as plt
from os.path import join
from pathlib import Path
from tqdm import tqdm


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
        

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

# def save_results(run_id, task, metric, sample_size, predictions, ground_truth_distances):
#     if os.path.exists("results.json"):
#         with open("results.json", "r") as f:
#             results = json.load(f)
#     else:
#         results = {}
    
#     try:
#         last_run_id = int(list(results.keys())[-1])

#         results[last_run_id+1] = {
#             "run_id": run_id,
#             "task": task,
#             "metric": metric,
#             "sample_size": sample_size,
#             "predictions": predictions,
#             "ground_truth_distances": ground_truth_distances
#         }
#     except:
#         results[0] = {
#             "run_id": run_id,
#             "task": task,
#             "metric": metric,
#             "sample_size": sample_size,
#             "predictions": predictions,
#             "ground_truth_distances": ground_truth_distances
#         }

#     with open("results.json", "w") as f:
#         json.dump(results, f, indent=4)

def save_results(evaluation_result, predictions_dir, task):
    results_json = join(predictions_dir, "results.json")
    
    if os.path.exists(results_json):
        with open(results_json, "r") as f:
            results = json.load(f)
    else:
        results = {}

    if task == "absolute_distance":
        results["absolute_distance_error"] = evaluation_result
    elif task == "relative_distance":
        results["relative_distance_error"] = evaluation_result
    elif task == "relative_direction":
        results["relative_direction_error"] = evaluation_result

    with open(results_json, "w") as f:
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

def load_prompt(task, model):
    # TODO: Add error handling
    if model == "gpt":
        if task == "absolute_distance":
            with open("prompts/absolute_distance.txt") as f:
                prompt = f.read()
        elif task == "relative_distance":
            with open("prompts/relative_distance.txt") as f:
                prompt = f.read()
        elif task == "relative_direction":
            with open("prompts/relative_direction.txt") as f:
                prompt = f.read()
    elif model == "gpt_socratic":
        if task == "absolute_distance":
            with open("prompts/socratic_absolute_distance.txt") as f:
                prompt = f.read()
        elif task == "relative_distance":
            with open("prompts/socratic_relative_distance.txt") as f:
                prompt = f.read()
        elif task == "relative_direction":
            with open("prompts/socratic_relative_direction.txt") as f:
                prompt = f.read()
    
    return prompt


def format_prompt(prompt, task, entities):
    # absolute: 1, relative: 2, direction: 2
    if task == "absolute_distance":
        prompt = prompt.format(entity_1=entities[0])
    elif task == "relative_distance" or task == "relative_direction":
        prompt = prompt.format(entity_1=entities[0], entity_2=entities[1])
    
    return prompt


def prepare_image(image_dir, biome, trajectory, frame):
    frame = join(image_dir, f"{biome}_{trajectory}_{frame}.jpg")
    encoded_frame = encode_image(frame)
    image_url = f"data:image/jpeg;base64,{encoded_frame}"

    return image_url


def create_predictions_folder(run_id, model_type):
    # Get the parent directory (spatial_evaluation)
    base_dir = Path(__file__).parent  # Moves up one level from "sampler"
    
    # Define the directory where samples should be stored
    predictions = base_dir / "predictions" / f"{run_id}_{model_type}"

    predictions.mkdir(parents=True, exist_ok=True)

    return predictions