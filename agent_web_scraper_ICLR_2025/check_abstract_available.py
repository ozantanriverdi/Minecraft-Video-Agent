import json

if __name__ == '__main__':
    relevant_present_counter = 0
    relevant_papers_with_abstracts = []  # List to store matching titles and abstracts
    not_found_titles = []  # List to keep track of lines not found

    # Load relevant papers and abstracts
    with open("relevant_papers.txt", "r", encoding="utf-8") as f:
        relevant_titles = {line.strip() for line in f}  # Set of relevant titles for fast lookup
    
    with open("articles_with_abstracts.json", "r", encoding="utf-8") as f_2:
        abstracts = json.load(f_2)

    # Match titles in relevant papers with articles_with_abstracts
    for abstract in abstracts:
        title = abstract.get("title", "").strip()
        if title in relevant_titles:
            relevant_papers_with_abstracts.append({
                "title": title,
                "abstract": abstract.get("abstract", "")
            })
            relevant_present_counter += 1

    # Save the matched relevant papers with abstracts to a new JSON file
    with open("relevant_papers_with_abstracts.json", "w", encoding="utf-8") as f_out:
        json.dump(relevant_papers_with_abstracts, f_out, indent=4, ensure_ascii=False)

    # Print the count of matching lines
    print("Number of relevant papers found:", relevant_present_counter)

    # Print the lines that were not found
    # print("Titles not found in abstracts:")
    # for title in not_found_titles:
    #     print(title)

if __name__ == '__main__':
    missing_articles = []  # List to store missing articles

    # Load relevant papers and abstracts
    with open("relevant_papers.txt", "r", encoding="utf-8") as f:
        relevant_titles = {line.strip() for line in f}  # Set of relevant titles for fast lookup

    with open("articles_with_abstracts.json", "r", encoding="utf-8") as f_2:
        abstracts = json.load(f_2)

    # Extract all titles from the abstracts
    abstract_titles = {abstract.get("title", "").strip() for abstract in abstracts}

    # Find titles in relevant_papers.txt that are not in articles_with_abstracts.json
    missing_articles = list(relevant_titles - abstract_titles)

    # Save the missing articles to a JSON file
    with open("missing_relevant_papers.json", "w", encoding="utf-8") as f_out:
        json.dump(missing_articles, f_out, indent=4, ensure_ascii=False)

    # Print count of missing articles
    print(f"Number of missing articles: {len(missing_articles)}")
