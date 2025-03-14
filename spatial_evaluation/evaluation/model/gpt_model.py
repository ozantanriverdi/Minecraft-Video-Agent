import openai
from openai import OpenAI


class GPT_Model:
    def __init__(self, api_key, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
    
    def forward(self, prompt, image, max_tries=5):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
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
                    max_tokens=300) # Limit response length
                
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

        print("Max retries reached. API call failed.")
        return None # Return None if all attempts fail
