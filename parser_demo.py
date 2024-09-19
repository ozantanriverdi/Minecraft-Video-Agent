import re
import numpy as np
import json


def extract_action_vector(llm_output):
    success = 1
    llm_output = llm_output.replace(' ', '').replace('\n', '')
    llm_output = llm_output.replace(r'\[', '[').replace(r'\]', ']')
    # Regex pattern to match exactly 8 elements inside square brackets [] or parentheses ()
    pattern_single = r"\[((\d+,\s*){7}\d+)\]|\(((\d+,\s*){7}\d+)\)"

    # Regex pattern to match a list of action vectors, each with exactly 8 elements
    pattern_list = r"\[\[((\d+,\s*){7}\d+)(\],\s*\[((\d+,\s*){7}\d+))*\]\]"

    try:
        # Search for the pattern in the LLM output
        match_list = re.search(pattern_list, llm_output)
        if match_list:
            # Extract the list of action vectors as a string
            vector_list_str = match_list.group(0)[2:-2] # Remove outer [[ and ]]
            # Split by "], [" to separate individual vectors
            #vector_list_str = vector_list_str.replace(' ', '').replace('\n', '')
            vector_list = vector_list_str.split("],[")
            # Convert each vector string to a numpy array of integers
            action_vectors = [np.array([int(x) for x in vec.split(',')]) for vec in vector_list]
            return action_vectors, success

        match_single = re.search(pattern_single, llm_output)
        if match_single:
            # Extract the single vector
            vector_str = match_single.group(1) if match_single.group(1) else match_single.group(3)
            # Convert the string into a list of integers and return it as a numpy array
            action_vector = np.array([int(x) for x in vector_str.split(',')])
            return action_vector, success
        # Raise ValueError if no match was found
        raise ValueError("No valid action vector or list of vectors found in the output, expecting exactly 8 elements.")
    
    except ValueError as e:
        print(e)
        success = 0
        # Return default vector if ValueError is raised
        return np.array([0, 0, 0, 12, 12, 0, 0, 0]), success
    

if __name__ == '__main__':
    text = """
        Based on the current observation, it seems you are in a forested area with a lot of trees and foliage. There are no visible cows in this image.

        To proceed with the task of obtaining milk from a cow, the first action will involve moving forward to search for a cow. Here are a couple of actions to take:

        1. Move forward to explore the area.
        2. Continue moving forward to potentially find a cow.

        Here are the action vectors:

        ```python
        [[1, 0, 0, 12, 12, 0, 0, 0], 
        [1, 0, 0, 12, 12, 0, 0, 0]]
        ```"""
    text_2 = """
        Based on the current observation, it seems you are in a forested area with a lot of trees and foliage. There are no visible cows in this image.

        To proceed with the task of obtaining milk from a cow, the first action will involve moving forward to search for a cow. Here are a couple of actions to take:

        1. Move forward to explore the area.
        2. Continue moving forward to potentially find a cow.

        Here are the action vectors:

        ```python
        \[1, 5, 0, 12, 12, 0, 0, 0\]
        ```"""
    text_3 = """
        Based on the current observation, it seems you are in a forested area with a lot of trees and foliage. There are no visible cows in this image.

        To proceed with the task of obtaining milk from a cow, the first action will involve moving forward to search for a cow. Here are a couple of actions to take:

        1. Move forward to explore the area.
        2. Continue moving forward to potentially find a cow.

        Here are the action vectors:

        ```python
        [\[1, 4, 0, 12, 12, 0, 0, 0], [1, 6, 0, 12, 12, 0, 0, 0], [1, 7, 0, 12, 12, 0, 0, 0]\]
        ```"""
    # print(extract_action_vector(text))
    # print("****************")
    # print(extract_action_vector(text_2))
    # print("****************")
    # print(extract_action_vector(text_3))
    with open("prompt.txt", "r") as f:
        prompt_text_raw = f.read()
    
    print(prompt_text_raw)