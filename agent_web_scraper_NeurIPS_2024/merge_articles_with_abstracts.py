import json

all_articles = []


with open("articles_with_abstracts_0.json", "r", encoding="utf-8") as f_0:
    articles_0 = json.load(f_0)

with open("articles_with_abstracts_1.json", "r", encoding="utf-8") as f_1:
    articles_1 = json.load(f_1)

with open("articles_with_abstracts_2.json", "r", encoding="utf-8") as f_2:
    articles_2 = json.load(f_2)

all_articles.extend(articles_0)
all_articles.extend(articles_1)
all_articles.extend(articles_2)

with open("articles_with_abstracts.json", "w", encoding="utf-8") as f_3:
    json.dump(all_articles, f_3, indent=4, ensure_ascii=False)