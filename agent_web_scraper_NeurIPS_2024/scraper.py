import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


driver = webdriver.Chrome()

urls = {
    'url_0': r'https://openreview.net/group?id=NeurIPS.cc/2024/Conference#tab-accept-oral',
    'url_1': r'https://openreview.net/group?id=NeurIPS.cc/2024/Conference#tab-accept-spotlight',
    'url_2': r'https://openreview.net/group?id=NeurIPS.cc/2024/Conference#tab-accept-poster'
}

xpaths = {
    'xpath_0': '//*[@id="accept-oral"]/div/div/ul/li/div/h4/a[1]',
    'xpath_1': '//*[@id="accept-spotlight"]/div/div/ul/li/div/h4/a[1]',
    'xpath_2': '//*[@id="accept-poster"]/div/div/ul/li/div/h4/a[1]'
}

buttons = {
    'button_0': '//*[@id="accept-oral"]/div/div/nav/ul/li[6]/a',
    'button_1': '//*[@id="accept-spotlight"]/div/div/nav/ul/li[13]/a',
    'button_2': '//*[@id="accept-poster"]/div/div/nav/ul/li[13]/a'
}

# Open files in write mode to start fresh
# with open("article_titles.txt", "w", encoding="utf-8") as _:
#     pass
# with open("page_counter.txt", "w", encoding="utf-8") as _:
#     pass

i = 2

# for i in range(3):
#     try:
page_counter = 1  # Reset counter for each tab
driver.get(urls[f'url_{i}'])
time.sleep(1)

with open("article_titles.txt", "a", encoding="utf-8") as file, open("page_counter.txt", "a", encoding="utf-8") as file_2:
    while True:
        try:
            # Wait until the list of articles is present
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, xpaths[f'xpath_{i}']))
            )
            articles = driver.find_elements(By.XPATH, xpaths[f'xpath_{i}'])

            for article in articles:
                soup = BeautifulSoup(article.get_attribute("innerHTML"), "html.parser")
                title = soup.get_text(separator=" ").strip()
                file.write(title + "\n")
            file_2.write(f"Tab {i}, Page {page_counter}\n")
            page_counter += 1

            # Click the next button
            next_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, buttons[f'button_{i}']))
            )
            next_btn.click()
            time.sleep(2)

        except Exception as e:
            print(f"No more pages or an error occurred in Tab {i}, Page {page_counter}: {e}")
            break

    # except Exception as e:
    #     print(f"Error processing tab {i}: {e}")

driver.quit()