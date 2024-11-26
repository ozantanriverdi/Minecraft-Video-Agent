import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver = webdriver.Chrome()
url = r'https://openreview.net/group?id=ICLR.cc/2025/Conference&referrer=%5BHomepage%5D(%2F)#tab-active-submissions'
driver.get(url)
time.sleep(1)

page_counter = 1
with open("article_titles.txt", "w", encoding="utf-8") as file:
    with open("page_counter.txt", "w", encoding="utf-8") as file_2:
        while True:
            try:
                # Wait until the list of articles is present
                WebDriverWait(driver, 10).until(
                    EC.presence_of_all_elements_located((By.XPATH, '//*[@id="active-submissions"]/div/div/ul/li/div/h4/a[1]'))
                )
                articles = driver.find_elements(By.XPATH, '//*[@id="active-submissions"]/div/div/ul/li/div/h4/a[1]')

                for article in articles:
                    soup = BeautifulSoup(article.get_attribute("innerHTML"), "html.parser")
                    title = soup.get_text(separator=" ").strip()
                    file.write(title + "\n")
                file_2.write(str(page_counter) + "\n")
                page_counter += 1

                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, '//*[@id="active-submissions"]/div/div/nav/ul/li[13]/a'))
                )
                next_button.click()
                time.sleep(2)
            except Exception as e:
                print("No more pages or an error occurred:", e)
                break

driver.quit()
