import json

num_responses = 59

relevant_papers = []

for response_count in range(1, num_responses+1):
    with open(f"gpt_filtered/gpt_response_{response_count}.json", "r", encoding="utf-8") as f:
        try:
            gpt_response = json.load(f)
            for article in gpt_response:
                if article.get("relevant") is True:
                    relevant_papers.append(str(article.get("title", "")))
        except Exception as e:
            print(f"Error in {response_count}: {e}")

with open("relevant_papers.txt", "w", encoding="utf-8") as f:
    for article in relevant_papers:
        f.write(article + "\n")