import time
import json
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

driver = webdriver.Chrome()  # Optional argument, if not specified will search path.

urls = {
    'url_0': r'https://openreview.net/group?id=NeurIPS.cc/2024/Conference#tab-accept-oral',
    'url_1': r'https://openreview.net/group?id=NeurIPS.cc/2024/Conference#tab-accept-spotlight',
    'url_2': r'https://openreview.net/group?id=NeurIPS.cc/2024/Conference#tab-accept-poster'
}

xpaths = {
    'xpath_0': '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li',
    'xpath_1': '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[3]/div/div/ul/li',
    'xpath_2': '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/div/div/ul/li'
}

buttons = {
    'button_0': '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/nav/ul/li[6]/a',
    'button_1': '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[3]/div/div/nav/ul/li[13]/a',
    'button_2': '/html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/div/div/nav/ul/li[13]/a'
}
i = 2
page_counter = 1
driver.get(urls[f'url_{i}'])
time.sleep(1)

#soup = BeautifulSoup(driver.page_source, 'html.parser')
#article = soup.find_all(//*[@id="active-submissions"]/div/div/ul/li[1]/div/h4/a[1]/text())
page_counter = 1
articles_data = []

# try:
#     for i in range(1):
#         # Wait until the list of articles is present
#         WebDriverWait(driver, 10).until(
#             EC.presence_of_all_elements_located((By.XPATH, '//*[@id="active-submissions"]/div/div/ul/li'))
#         )
        
#         # Loop over each article by list item index
#         articles = driver.find_elements(By.XPATH, '//*[@id="active-submissions"]/div/div/ul/li')
#         for idx in range(1, len(articles) + 1):  # Adjust index for XPath starting at 1
#             # Title XPath
#             title_xpath = f'//*[@id="active-submissions"]/div/div/ul/li[{idx}]/div/h4/a[1]'
#             title_element = driver.find_element(By.XPATH, title_xpath)
#             title = title_element.text.strip()
#             //*[@id="active-submissions"]/div/div/ul/li[1]/div/div[2]/a
#             # Show details button XPath
#             show_details_xpath = f'//*[@id="active-submissions"]/div/div/ul/li[{idx}]/div//a[contains(text(), "Show details")]'
#             try:
#                 show_details_button = driver.find_element(By.XPATH, show_details_xpath)
#                 driver.execute_script("arguments[0].click();", show_details_button)
#                 time.sleep(1)  # Brief pause for content to load

#                 # Abstract XPath
#                 abstract_xpath = f'//*[@id="active-submissions"]/div/div/ul/li[{idx}]/div/div[2]/div/div/div[2]/div/p'
#                 abstract_element = driver.find_element(By.XPATH, abstract_xpath)
#                 abstract = abstract_element.text.strip()

#                 # Save title and abstract to list
#                 articles_data.append({"title": title, "abstract": abstract})

#             except Exception as e:
#                 print(f"Error retrieving abstract for {title}: {e}")

#         # Move to the next page
#         page_counter += 1
#         next_button = WebDriverWait(driver, 10).until(
#             EC.element_to_be_clickable((By.XPATH, '//*[@id="active-submissions"]/div/div/nav/ul/li[13]/a'))
#         )
#         next_button.click()
#         time.sleep(2)

# except Exception as e:
#     print("No more pages or an error occurred:", e)

# finally:
#     # Save all articles data to a JSON file
#     with open("articles_with_abstracts.json", "w", encoding="utf-8") as file:
#         json.dump(articles_data, file, indent=4, ensure_ascii=False)

#     driver.quit()



# /html/body/div[1]/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[1]/div/div[2]/div/div/div[2]/div/p/text()
# /html/body/div[1]/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[2]/div/div[2]/div/div/div[2]/div/p/text()
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li/div/h4/a[1]/text()
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[2]/div/h4/a[1]/text()
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[1]/div/div[2]/div/div/div[2]/div/p/text()
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[1]/div/h4/a[1]/text()
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[1]/div/div[2]/a
# NeurIPS_2024 - Oral
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[1]/div/h4/a[1]
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[2]/div/h4/a[1]
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[1]/div/div[2]/a
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[1]/div/div[2]/div/div/div[2]/div/p/text()[1]
# Button
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/nav/ul/li[6]/a
# Spotlight
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[3]/div/div/ul/li[1]/div/h4/a[1]
# /html/body/div/div[3]/div/div/main/div/div[3]/div/div[2]/div[4]/div/div/ul/li[1]/div/h4/a[1]
try:
    while True:
        # Wait until the list of articles is present
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, xpaths[f'xpath_{i}']))
        )
        articles = driver.find_elements(By.XPATH, xpaths[f'xpath_{i}'])

        for idx, article in enumerate(articles, start=1):
            # Full XPath for the title
            title_xpath = xpaths[f'xpath_{i}'] + f'[{idx}]/div/h4/a[1]'
            title_element = driver.find_element(By.XPATH, title_xpath)
            soup = BeautifulSoup(title_element.get_attribute("innerHTML"), "html.parser")
            title = soup.get_text(separator=" ").strip()

            # Try to click "Show details" to reveal the abstract
            try:
                # Full XPath for "Show details" button
                show_details_xpath = xpaths[f'xpath_{i}'] + f'[{idx}]/div//a[contains(text(), "Show details")]'
                show_details_button = driver.find_element(By.XPATH, show_details_xpath)
                driver.execute_script("arguments[0].click();", show_details_button)
                time.sleep(0.5)  # Adding a brief pause to allow the content to load

                # Full XPath for the abstract section following "Abstract:"
                try:
                    abstract_xpath = xpaths[f'xpath_{i}'] + f'[{idx}]/div/div[2]/div/div/div[2]/div/p'
                    abstract_section = driver.find_element(By.XPATH, abstract_xpath)
                    abstract_soup = BeautifulSoup(abstract_section.get_attribute("innerHTML"), "html.parser")
                    abstract = abstract_soup.get_text(separator=" ").strip()
                except NoSuchElementException:
                    abstract_xpath = xpaths[f'xpath_{i}'] + f'[{idx}]/div/div[2]/div/div/div[3]/div/p'
                    abstract_section = driver.find_element(By.XPATH, abstract_xpath)
                    abstract_soup = BeautifulSoup(abstract_section.get_attribute("innerHTML"), "html.parser")
                    abstract = abstract_soup.get_text(separator=" ").strip()


                # Save title and abstract to the list
                articles_data.append({"title": title, "abstract": abstract})

            except Exception as e:
                print(f"Error retrieving abstract for {title}: {e}")

            # Optionally close the "Show details" section if needed
            # try:
            #     driver.execute_script("arguments[0].click();", show_details_button)  # Collapse if it affects the next element
            # except:
            #     pass

        # Move to the next page

        page_counter += 1
        next_button_xpath = buttons[f'button_{i}']
        next_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, next_button_xpath))
        )
        next_button.click()
        time.sleep(2)
        


except Exception as e:
    print("No more pages or an error occurred:", e)

finally:
    # Save all articles data to a JSON file
    with open(f"articles_with_abstracts_{i}.json", "w", encoding="utf-8") as file:
        json.dump(articles_data, file, indent=4, ensure_ascii=False)

    driver.quit()


# /html/body/div[1]/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[25]/div/div[2]/div/div/div[3]/div/p/text()
# /html/body/div[1]/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[25]/div/div[2]/div/div/div[2]/div/p
# /html/body/div[1]/div[3]/div/div/main/div/div[3]/div/div[2]/div[2]/div/div/ul/li[25]/div/div[2]/div/div/div[3]/div/p/text()