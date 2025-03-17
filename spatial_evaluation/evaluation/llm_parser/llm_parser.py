import json
import re

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
        match = re.search(r"```*([\s\S]*?)```", response) or re.search(r"{.*}", response)

        if match:
            json_str = match.group(1)  # Extract JSON content
            try:
                parsed_json = json.loads(json_str)  # Parse JSON
                return parsed_json.get("direction", None)  # Get 'distance' value
            except json.JSONDecodeError:
                return None  # Return None if JSON is malformed
        return None  # Return None if no JSON found