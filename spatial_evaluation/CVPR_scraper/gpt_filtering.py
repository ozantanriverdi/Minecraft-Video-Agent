import json
import os
from os.path import join
import openai
from openai import OpenAI
from tqdm import tqdm
import time

MAX_RETRIES = 3
RETRY_DELAY = 5

num_chunks = 59

client = OpenAI(api_key="")

cwd = os.getcwd()
gpt_filtered = join(cwd, "gpt_filtered")
os.makedirs(gpt_filtered, exist_ok=True)

with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt_text_raw = f.read()

for i in tqdm(range(1, num_chunks+1)):
    with open(f"chunks/titles_chunk_{i}.txt", "r", encoding="utf-8") as f:
        current_chunk = f.read()

    prompt_text = prompt_text_raw.format(articles=current_chunk)

    #print(prompt_text)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant who filters article titles by relevance to spatial understanding and evaluation."
                    },
                    {
                        "role": "user", 
                        "content": prompt_text
                    }
                ]
            )

            response_text = response.choices[0].message.content
            #print(response_text)

            if response_text.startswith("```json"):
                response_text = response_text[7:].lstrip()  # Remove '```json' prefix
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing '```'

            # Save the filtered output to a JSON file
            with open(join(gpt_filtered, f"gpt_response_{i}.json"), "w", encoding="utf-8") as f:
                f.write(response_text)

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