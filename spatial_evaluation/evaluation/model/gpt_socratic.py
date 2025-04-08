import openai
import time
from openai import OpenAI
from pathlib import Path
from os.path import join


class GPT_Socratic_Model:
    def __init__(self, api_key, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
 
    def socratic_prompt_loader(self):
        base_dir = Path(__file__).parent.parent
        prompts_dir = base_dir / "prompts"
        socratic_prompt = join(prompts_dir, "socratic_initial_prompt.txt")
        
        with open(socratic_prompt, "r") as f:
            socratic_prompt_text = f.read()
        
        return socratic_prompt_text

    def get_socratic_description(self, image, max_tries=3):
        socratic_prompt_text = self.socratic_prompt_loader()
        messages = [
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
                            "url": image
                        }
                    }
                ]
            }
        ]
        attempt = 0

        while attempt < max_tries:
            try:
                # Send the request to OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=1000) # Limit response length
                # print response to view token usage
                # print(response)
                # Return the generated output
                return response.choices[0].message.content
            
            # Handle different API-related errors
            except openai.InternalServerError:
                print("OpenAI API service is temporarily unavailable. Please try again later.")
                
            except openai.AuthenticationError:
                print("There was an issue with API authentication. Please check your API key.")
                break # Stop retrying if API key is invalid
            except openai.RateLimitError:
                print("You have exceeded your rate limit or run out of credits. Please check your usage.")
                break # Stop retrying if out of credits
            except openai.APITimeoutError:
                print("Request timed out.")

            except openai.APIConnectionError:
                print("Network error: Unable to connect to the OpenAI API. Please check your internet connection.")

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            
            attempt += 1
            time.sleep(5)

        print("Max retries reached. Socratic Description API call failed.")
        return None # Return None if all attempts fail

    def forward(self, prompt, image, max_tries=5):
        socratic_description = self.get_socratic_description(image)
        prompt = prompt.replace("{description}", socratic_description)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]

        attempt = 0
        time.sleep(5)
        while attempt < max_tries:
            try:
                # Send the request to OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=300) # Limit response length
                # print response to view token usage
                # print(response)
                # Return the generated output
                return response.choices[0].message.content, socratic_description
            
            # Handle different API-related errors
            except openai.InternalServerError:
                print("OpenAI API service is temporarily unavailable. Please try again later.")
                
            except openai.AuthenticationError:
                print("There was an issue with API authentication. Please check your API key.")
                break # Stop retrying if API key is invalid
            except openai.RateLimitError:
                print("You have exceeded your rate limit or run out of credits. Please check your usage.")
                break # Stop retrying if out of credits
            except openai.APITimeoutError:
                print("Request timed out.")

            except openai.APIConnectionError:
                print("Network error: Unable to connect to the OpenAI API. Please check your internet connection.")

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            
            attempt += 1
            time.sleep(5)

        print("Max retries reached. API call failed.")
        return None, socratic_description # Return None if all attempts fail

