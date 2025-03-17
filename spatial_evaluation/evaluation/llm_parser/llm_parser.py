import json
import re
import ast


class LLM_Parser:
    def __init__(self, task):
        self.task = task

    def parse(self, response):
        if self.task == "absolute_distance" or self.task == "relative_distance":
            return self.parse_llm_output_distance(response)
        elif self.task == "relative_direction":
            return self.parse_llm_output_direction(response)

    def parse_llm_output_distance(self, response):
        match = re.search(r"```json\s*([\s\S]*?)```", response) or re.search(r"{.*}", response)

        if match:
            json_str = match.group(1)  # Extract JSON content
            try:
                parsed_json = json.loads(json_str)  # Parse JSON
                distance = parsed_json.get("distance", None)
                
                if isinstance(distance, (int, float)):
                    return distance
                else:
                    print(f"Warning: Invalid distance value '{distance}'")
                    return None  # Return None if the value is not numeric

            except (json.JSONDecodeError, TypeError) as e:
                print(f"JSON Decode Error: {e}")
                return None  # Return None if JSON is malformed
        return None  # Return None if no JSON found
    
    def parse_llm_output_direction(self, response):
        """
        Parses LLM output to extract the relative direction response in the expected format: [-1, 0, 1].
        
        :param response: The raw response from the LLM containing JSON with "direction".
        :return: A list of 3 integers representing relative directions, or None if parsing fails.
        """
        # Match JSON response within triple backticks or raw JSON
        match = re.search(r"```json\s*([\s\S]*?)```", response) or re.search(r"{.*}", response)

        if match:
            json_str = match.group(1)  # Extract JSON content
            try:
                parsed_json = json.loads(json_str)  # Parse JSON
                direction = parsed_json.get("direction", None)  # Extract the "direction" field

                # ✅ Fix: Convert direction from string to a list if necessary
                if isinstance(direction, str):  
                    try:
                        direction = ast.literal_eval(direction)  # Safely convert string list to actual list
                    except (SyntaxError, ValueError):
                        print(f"Warning: Failed to convert string to list: {direction}")
                        return None
                
                # Ensure direction is a list of exactly 3 values, each being -1, 0, or 1
                if isinstance(direction, list) and len(direction) == 3 and all(d in [-1, 0, 1] for d in direction):
                    return direction
                else:
                    print(f"Warning: Invalid direction format: {direction}")
                    return None  # Return None if the format is incorrect

            except json.JSONDecodeError as e:
                print(f"JSON Decode Error: {e}")
                return None  # Return None if JSON is malformed

        print("Warning: No JSON found in response.")
        return None  # Return None if no JSON found