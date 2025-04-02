import os
from os.path import join
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
url = r'https://cvpr.thecvf.com/Conferences/2025/AcceptedPapers'
driver.get(url)
time.sleep(1)

WebDriverWait(driver, 10).until(
    EC.presence_of_all_elements_located((By.XPATH, '//*[@id="main"]/div[2]/div/div/div/div/table/tbody/tr/td[1]/a'))
)


articles_a = []
articles_a_elem = driver.find_elements(By.XPATH, '//*[@id="main"]/div[2]/div/div/div/div/table/tbody/tr/td[1]/a')
for article in articles_a_elem:
    soup = BeautifulSoup(article.get_attribute("innerHTML"), "html.parser")
    title = soup.get_text(separator=" ").strip()
    articles_a.append(title)


articles_strong = []
articles_strong_elem = driver.find_elements(By.XPATH, '//*[@id="main"]/div[2]/div/div/div/div/table/tbody/tr/td[1]/strong')
for article in articles_strong_elem:
    soup = BeautifulSoup(article.get_attribute("innerHTML"), "html.parser")
    title = soup.get_text(separator=" ").strip()
    articles_strong.append(title)


articles = []
articles.extend(articles_a)
articles.extend(articles_strong)

cwd = os.getcwd()
chunks_dir = join(cwd, "chunks")
os.makedirs(chunks_dir, exist_ok=True)

chunk_articles = []
article_counter = 1
file_counter = 1

print(len(articles))
print("************")


for article in articles:
    #print(article.text)
    chunk_articles.append(article)
    if article_counter == 50:
        with open(join(chunks_dir, f"titles_chunk_{file_counter}.txt"), "w", encoding="utf-8") as f:
            for article in chunk_articles:
                f.write(article.strip() + "\n")
        chunk_articles = []
        article_counter = 0
        file_counter += 1
    article_counter += 1

if chunk_articles:
    with open(join(chunks_dir, f"titles_chunk_{file_counter}.txt"), "w", encoding="utf-8") as file_2:
        file_2.writelines(chunk_articles)

