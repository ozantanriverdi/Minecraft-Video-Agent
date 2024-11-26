import json
import os
from os.path import join
import openai
from openai import OpenAI
from tqdm import tqdm
import time

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

if __name__ == '__main__':
    client = OpenAI(api_key="")
    
    cwd = os.getcwd()
    gpt_filtered = join(cwd, "gpt_filtered")
    os.makedirs(gpt_filtered, exist_ok=True)

    with open("prompt.txt", "r", encoding="utf-8") as f:
        prompt_text_raw = f.read()

    # request_again = [2, 22, 23, 25, 50, 51, 61, 69, 98, 101, 125, 145, 150, 161, 165, 172, 176, 178, 181, 184, 191, 192, 218]
    request_again = [45]

    for i in tqdm(request_again):
        with open(f"chunks/titles_chunk_{i}.txt", "r", encoding="utf-8") as f_2:
            current_chunk = f_2.read()
        
        prompt_text = prompt_text_raw.format(articles=current_chunk)

        #print(prompt_text)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Send request to OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You filter article titles by relevance to LLM agents in open-world settings, avoiding unrelated reinforcement learning."
                        },
                        {
                            "role": "user", 
                            "content": prompt_text
                        }
                    ]
                )

                # Process the response text, handling JSON format markers
                response_text = response.choices[0].message.content
                if response_text.startswith("```json"):
                    response_text = response_text[7:].lstrip()  # Remove '```json' prefix
                if response_text.endswith("```"):
                    response_text = response_text[:-3]  # Remove trailing '```'

                # Save the filtered output to a JSON file
                with open(join(gpt_filtered, f"gpt_response_{i}.json"), "w", encoding="utf-8") as f_3:
                    f_3.write(response_text)
                
                # Break out of the retry loop if successful
                break

            except (openai.InternalServerError, openai.APITimeoutError, openai.APIConnectionError) as e:
                print(f"Attempt {attempt} - Temporary API error: {e}. Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            except openai.RateLimitError:
                print("Rate limit exceeded or credits exhausted. Please check usage.")
                break
            except openai.AuthenticationError:
                print("Authentication error: Check your API key.")
                break
            except json.JSONDecodeError:
                print("Error decoding JSON response from the API.")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break

        else:
            # Log to a file if all attempts fail
            with open("error_log.txt", "a") as error_file:
                error_file.write(f"Failed to process chunk {i} after {MAX_RETRIES} attempts.\n")