import json
import re
import ast


class LLM_Parser:
    def __init__(self, task, model_type):
        self.task = task
        self.model = model_type

    def parse(self, response):
        if self.task in ("absolute_distance", "relative_distance") and self.model in ("gpt", "gpt_socratic"):
            return self.parse_gpt_output_distance(response)
        elif self.task in ("absolute_distance", "relative_distance") and self.model in ("llava"):
            return self.parse_llava_output_distance(response)
        elif self.task == "relative_direction" and self.model in ("gpt", "gpt_socratic"):
            return self.parse_gpt_output_direction(response)
        elif self.task == "relative_direction" and self.model in ("llava"):
            return self.parse_llava_output_direction(response)

    def parse_gpt_output_distance(self, response):
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
    
    def parse_gpt_output_direction(self, response):
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
    
    def parse_llava_output_distance(self, response: str):
        """
        Extracts the numeric 'distance' from an assistant's response block in a raw string.
        Handles both JSON markdown and plain JSON formats.
        """

        # Focus only on the assistant's response block
        assistant_block = re.search(r"ASSISTANT:\s*({[\s\S]*?})", response)
        if not assistant_block:
            print("Warning: No ASSISTANT block with JSON found.")
            return None

        json_str = assistant_block.group(1)

        try:
            parsed_json = json.loads(json_str)
            distance = parsed_json.get("distance", None)

            if isinstance(distance, (int, float)):
                return distance
            else:
                print(f"Warning: Invalid distance value '{distance}' (non-numeric)")
                return None

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return None
        
    def parse_llava_output_direction(self, response):
        """
        Parses an LLM ASSISTANT-style output to extract the relative direction in format: [-1, 0, 1].

        Handles various quirks such as:
        - direction as string: "[-1, 0, 1]"
        - direction as list with string inside: ["-1, 0, 1"]
        - direction with wrong quotes: ['-1, 0, 1']
        """

        # Find ASSISTANT block containing JSON
        match = re.search(r"ASSISTANT:\s*({[\s\S]*?})", response)
        if not match:
            print("Warning: No ASSISTANT JSON block found.")
            return None

        json_str = match.group(1)

        try:
            parsed_json = json.loads(json_str)
            direction = parsed_json.get("direction", None)

            # Step 1: If it's a string, try to eval directly
            if isinstance(direction, str):
                try:
                    direction = ast.literal_eval(direction)
                except Exception:
                    pass  # If literal_eval fails, we handle below

            # Step 2: If it's a list with one string element, clean it manually
            if isinstance(direction, list) and len(direction) == 1 and isinstance(direction[0], str):
                cleaned = direction[0].replace("'", "").replace('"', '').strip()
                direction = [int(x.strip()) for x in cleaned.split(",")]

            # Final validation
            if isinstance(direction, list) and len(direction) == 3 and all(d in [-1, 0, 1] for d in direction):
                return direction
            else:
                print(f"Warning: Invalid direction format after cleaning: {direction}")
                return None

        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            print(f"JSON Decode Error or Format Error: {e}")
            return None