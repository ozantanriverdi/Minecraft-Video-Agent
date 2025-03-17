import os
from openai import OpenAI
from os.path import join
import json
import ast
import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils import *
from ground_truth_extractor import GroundTruthExtractor
from model import Model
from evaluator import Evaluator
from llm_parser import LLM_Parser


def main(run_id, model_type, task, dataset, groundTruthExtractor, model, evaluator, parser):
    
    
    base_dir = Path(__file__).parent.parent
    samples_dir = base_dir / "samples"
    run_rgb_obs_dir = samples_dir / "rgb_frames" / run_id
    run_info_dir = samples_dir / "info" / run_id

    if dataset == "custom":
        filtered_frames = evaluate_custom_frames()
    else:
        filtered_frames = groundTruthExtractor.filter_trajectories()

    print(f"Frames to be used in the evaluation: {filtered_frames}")
    print(f"Number of frames to be used: {len(filtered_frames['biome'])}")
    groundTruthExtractor.extract_ground_truths()
    with open(join("ground_truths", run_id, f"{task}.json"), "r") as f:
        ground_truths = json.load(f)

    prompt_raw = load_prompt(task, model_type)
    
    predictions_dir = create_predictions_folder(run_id, model_type)
    predictions = {}
    max_retries = 5
    
    for biome, trajectory, frame in tqdm(zip(filtered_frames["biome"], filtered_frames["trajectory"], filtered_frames["frame"])):
        
        with open(join(run_info_dir, f"info_step_{biome}_{trajectory}.json"), "r") as f:
            trajectory_info = json.load(f)
            
        entities_spawned = trajectory_info["entities_spawned"]

        prompt = format_prompt(prompt_raw, task, entities_spawned)
        image_url = prepare_image(run_rgb_obs_dir, biome, trajectory, frame)

        parsed_output = None
        attempts = 0

        while parsed_output is None and attempts < max_retries:

            output_raw = model.forward(prompt, image_url)
            print(output_raw)
            parsed_output = parser.parse(output_raw)
            print(parsed_output)
            if parsed_output is None:
                print(f"Parsing failed, retrying... ({attempts + 1}/{max_retries})")
            attempts += 1
        
        biome, trajectory, frame = str(biome), str(trajectory), str(frame)
        if biome not in predictions:
            predictions[biome] = {}
        if trajectory not in predictions[biome]:
            predictions[biome][trajectory] = {}
        
        predictions[biome][trajectory][frame] = parsed_output

    
    with open(join(predictions_dir, f"{task}.json"), "w") as f:
        json.dump(predictions, f, indent=4)
    print(predictions)
    evaluation_result = evaluator.evaluate_predictions(filtered_frames, predictions, ground_truths)
    print(evaluation_result)
    save_results(evaluation_result, predictions_dir, task)




if __name__ == '__main__':
    # TODO: Error handling for invalid run_id
    run_id = "20250304_000717"
    # TODO: Error handling for invalid model_type and task
    dataset = "custom"
    model_type = "gpt"
    task = "relative_direction"


    if dataset == "custom":
        filtered_frames = evaluate_custom_frames()
        groundTruthExtractor = GroundTruthExtractor(run_id, biomes_count=2, trajectories_count=5, frames_count=16, filtered_frames=filtered_frames)
    else:
        groundTruthExtractor = GroundTruthExtractor(run_id, biomes_count=2, trajectories_count=5, frames_count=16)
    model = Model(model_type)
    evaluator = Evaluator(task)
    parser = LLM_Parser(task)

    main(run_id, model_type, task, dataset, groundTruthExtractor, model, evaluator, parser)
    






