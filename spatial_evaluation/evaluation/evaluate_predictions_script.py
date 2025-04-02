import json

from evaluator import Evaluator
from ground_truth_extractor import GroundTruthExtractor
from utils import *

def get_prediction_frames(predictions):

    prediction_frames = {"biome": [], "trajectory": [], "frame": []}

    for biome, trajectories in predictions.items():
        for trajectory, frames in trajectories.items():
            for frame in frames.keys():
                prediction_frames["biome"].append(int(biome))
                prediction_frames["trajectory"].append(int(trajectory))
                prediction_frames["frame"].append(int(frame))
    return prediction_frames

def load_predictions(predictions_dir, task):
    with open(f"{predictions_dir}/{task}.json", "r") as f:
        predictions = json.load(f)
    return predictions

def load_ground_truths(run_id, task):
    with open(f"ground_truths/{run_id}/{task}.json", "r") as f:
        ground_truths = json.load(f)
    return ground_truths

if __name__ == '__main__':
    run_id = "20250303_231817"
    exp = "20250303_231817_gpt_v9"
    task = "relative_direction" # ["absolute_distance", "relative_distance", "relative_direction"]

    predictions_dir = f"predictions/{exp}"

    # groundTruthExtractor = GroundTruthExtractor(run_id, biomes_count=10, trajectories_count=20, frames_count=16)
    # filtered_frames = groundTruthExtractor.filter_trajectories()
    # groundTruthExtractor.extract_ground_truths()

    predictions = load_predictions(predictions_dir, task)
    print(type(predictions))
    prediction_frames = get_prediction_frames(predictions)
    print(prediction_frames)
    
    ground_truths = load_ground_truths(run_id, task)

    evaluator = Evaluator(task)
    evaluation_result, single_results = evaluator.evaluate_predictions(prediction_frames, predictions, ground_truths)
    save_results(evaluation_result, single_results, predictions_dir, task)