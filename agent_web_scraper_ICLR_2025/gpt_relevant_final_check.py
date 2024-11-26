import json
import os
import re
from os.path import join
import openai
from openai import OpenAI
from tqdm import tqdm
import time

client = OpenAI(api_key="")

def create_prompt(articles_chunk):
    
    with open("prompt_2.txt", "r", encoding="utf-8") as f:
        prompt_text_raw = f.read()

    
    # Format articles for the prompt
    articles_str = json.dumps(articles_chunk, indent=4, ensure_ascii=False)

    prompt_text = prompt_text_raw.format(articles=articles_str)

    # Combine into a full prompt
    return prompt_text

def extract_json_content(response_text):
    """
    Extracts and returns only the JSON array from the GPT response,
    ignoring any additional explanations or formatting.
    """
    try:
        # Use regex to extract the JSON array part only
        json_array_match = re.search(r'\[\s*{.*}\s*\]', response_text, re.DOTALL)
        if json_array_match:
            json_content = json_array_match.group(0)
            return json.loads(json_content)  # Parse as JSON
        else:
            print("No JSON array found in response.")
            return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

if __name__ == '__main__':

    with open("relevant_papers_with_abstracts.json", "r", encoding="utf-8" ) as f:
        articles = json.load(f)

    chunk_size = 5
    chunks = [articles[i:i + chunk_size] for i in range(0, len(articles), chunk_size)]

    output_dir = "gpt_relevance_responses"
    os.makedirs(output_dir, exist_ok=True)

    for idx, chunk in tqdm(enumerate(chunks, 1), total=len(chunks)):
        prompt = create_prompt(chunk)

        try:
            # Send to GPT
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = response.choices[0].message.content
            json_data = extract_json_content(response_text)

            # with open(f"{output_dir}/gpt_response_{idx}.json", "w", encoding="utf-8") as f:
            #     if response_text.startswith("```json"):
            #         response_text = response_text[7:].lstrip()
            #     if response_text.endswith("```"):
            #         response_text = response_text[:-3]
            #     f.write(response_text)
                        # Save only if JSON extraction was successful
            if json_data is not None:
                with open(f"{output_dir}/gpt_response_{idx}.json", "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=4)
            else:
                print(f"Error: No JSON content found in response for chunk {idx}.")
        except Exception as e:
            print(f"Error processing chunk {idx}: {e}")
