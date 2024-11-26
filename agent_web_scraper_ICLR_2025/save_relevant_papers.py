import json

if __name__ == '__main__':
   
    relevant_papers = []
    for i in range(1, 219):
        with open(f"gpt_filtered/gpt_response_{i}.json", "r", encoding="utf-8") as f:
            try:
                gpt_response = json.load(f)
                for article in gpt_response:
                    if article.get("relevant") is True:
                        relevant_papers.append(str(article.get("title", "")))
            except Exception as e:
                print(f"Error in {i}: {e}")

    with open("relevant_papers.txt", "w", encoding="utf-8") as f:
        for article in relevant_papers:
           f.write(article + "\n")