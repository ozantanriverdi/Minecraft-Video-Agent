import os
from openai import OpenAI
from os.path import join
import json
import ast
import numpy as np
from utils import filter_runs, encode_image, parse_llm_output_distance, parse_llm_output_direction, distance_metric, direction_metric, save_results, plot_results

from spatial_evaluator import Evaluator



def main(task_type, sample_id, filtered_steps):
    outputs = []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)

    cwd = os.getcwd()
    #obs_dir = join(cwd, "obs")
    rgb_dir = join(cwd, "rgb_frames")
    run_rgb_dir = join(rgb_dir, sample_id)

    # if task_type == "absolute":
    #     prompt = "prompts/absolute_distance_prompt.txt"
    # elif task_type == "relative":
    #     prompt = "prompts/relative_distance_prompt.txt"
    # elif task_type == "direction":
    #     prompt = "prompts/relative_direction_prompt.txt"
    socratic_prompt = "prompts/socratic_initial_prompt.txt"


    for i, step in enumerate(filtered_steps):
        # DEBUG
        # if i >= 2:
        #     break
        frame = join(run_rgb_dir, f"{step}.jpg")
        encoded_frame = encode_image(frame)
        image_url = f"data:image/jpeg;base64,{encoded_frame}"

        with open(socratic_prompt, "r") as f:
            socratic_prompt_text = f.read()

        socratic_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": socratic_prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ]
        max_retries = 5
        attempts = 0
        while attempts < max_retries:
            try:
                response_1 = client.chat.completions.create(
                    model="gpt-4o-mini",  # Replace with the appropriate model name
                    messages=socratic_messages,
                    max_tokens=300)
                socratic_llm_output = response_1.choices[0].message.content
                # print("*****************")
                # print("SOCRATIC:")
                # print(socratic_llm_output)
                # print("*****************")
                second_prompt_text = f"""
                Based on this description of the scene, estimate the distance from the camera to the horse:

                {socratic_llm_output}

                Considering your description of the horse’s apparent size, depth cues, and real-world knowledge, what is the most reasonable estimate in meters?
                Return your response in JSON format:

                ```
                {{
                    "distance": <calculated_distance>
                }}
                ```

                """
                print(second_prompt_text)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": second_prompt_text
                            }
                        ]
                    }
                ]
                #print(messages)
                response_2 = client.chat.completions.create(
                    model="gpt-4o-mini",  # Replace with the appropriate model name
                    messages=messages,
                    max_tokens=300)
                llm_output = response_2.choices[0].message.content
                print("*****************")
                print("SECOND")
                print(llm_output)
                print("*****************")

                if task_type in ["absolute", "relative"]:
                    output_value = parse_llm_output_distance(llm_output)
                    if isinstance(output_value, (int, float)):
                        outputs.append(output_value)
                        break
                    else:
                        print(f"Invalid output, retrying ({attempts+1}/{max_retries})...")
                elif task_type == "direction":
                    output_value = parse_llm_output_direction(llm_output)

            except Exception as e:
                print(f"Error occurred: {e}, retrying ({attempts+1}/{max_retries})...")
            attempts += 1
        if attempts == max_retries:
            print("Failed to get a valid output after multiple attempts.")
            break # Handle failure gracefully
    return outputs





if __name__ == '__main__':

    sample_id = "20250221_165846"
    task = "absolute"
    filtered_steps = filter_runs(sample_id)
    #print(filtered_steps)

    predictions = main(task_type=task, sample_id=sample_id, filtered_steps=filtered_steps)
    #print(predictions)

    evaluator = Evaluator(sample_id=sample_id)

    if task == "absolute":
        ground_truth_distances = []
        for obs in filtered_steps:
            ground_truth_distance = evaluator.evaluate_absolute_distance(step=obs, entity_name="horse")
            ground_truth_distances.append(ground_truth_distance)
        #print(ground_truth_distances)
    elif task == "relative":
        ground_truth_distances = []
        for obs in filtered_steps:
            ground_truth_distance = evaluator.evaluate_relative_distance(step=obs, entity_names=["horse", "pig"])
            ground_truth_distances.append(ground_truth_distance)
        #print(ground_truth_distances)
    elif task == "direction":
        ground_truth_directions = []
        for obs in filtered_steps:
            ground_truth_direction = evaluator.evaluate_relative_direction(step=obs, entity_1="pig", entity_2="horse")
            ground_truth_directions.append(ground_truth_direction)
        #print(ground_truth_directions)

    if task == "absolute" or task == "relative":
        mae = distance_metric(predictions, ground_truth_distances) #DEBUG
        print(f"Error: {mae}")
        save_results(sample_id, task, mae, len(filtered_steps), predictions, ground_truth_distances)
        #plot_results("20250221_000412", sample_id)
    elif task == "direction":
        accuracy = direction_metric(predictions, ground_truth_directions)
        print(f"Accuracy: {accuracy}")
        save_results(sample_id, task, accuracy, len(filtered_steps), predictions, ground_truth_distances)
        #plot_results(sample_id)






