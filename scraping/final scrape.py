import time
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

# Initialize the webdriver
driver = webdriver.Chrome()

# Maximize the browser window
driver.maximize_window()

# Navigate to the website
url = "https://www.galottery.com/en-us/winning-numbers.html#tab-advSearch"
driver.get(url)

# Select the Cash 3 game
driver.find_element(By.CSS_SELECTOR, "#advSearchGameSelect > option:nth-child(8)").click()

# Enter the "from" date
driver.find_element(By.CSS_SELECTOR, "#advSearchFromMonth > option:nth-child(1)").click()
driver.find_element(By.CSS_SELECTOR, "#advSearchFromDay > option:nth-child(1)").click()
driver.find_element(By.CSS_SELECTOR, "#advSearchFromYear > option:nth-child(18)").click()

# Enter the "to" date
driver.find_element(By.CSS_SELECTOR, "#advSearchToMonth > option:nth-child(1)").click()
driver.find_element(By.CSS_SELECTOR, "#advSearchToDay > option:nth-child(1)").click()
driver.find_element(By.CSS_SELECTOR, "#advSearchToYear > option:nth-child(31)").click()

# Click the "Search" button
driver.find_element(By.CSS_SELECTOR, "#btnSearchByRange").click()

# Wait for the search results to be displayed
wait = WebDriverWait(driver, 30)
wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#advSearch > div > div.panel.panel-info.search-results-header > div > h5")))

while True:
    # Get the container with all the page buttons
    page_buttons_container = driver.find_element(By.CSS_SELECTOR, "div.pagination-container")

    # Get the current page by finding the element with the "active" class
    current_page = int(page_buttons_container.find_element(By.CSS_SELECTOR, "li.active > a").text)
    print("current page:", current_page)

    # Find all the page buttons
    page_buttons = page_buttons_container.find_elements(By.CSS_SELECTOR, "li > a")

    # Initialize next page button as None
    next_page_button = None

    # Check the number of child elements in the container
    if len(page_buttons) == 7:
        # Get the 5th child element as the next page button
        next_page_button = page_buttons[4]
    else:
        # Iterate through the page buttons
        for page_button in page_buttons:
            page_number = int(page_button.text.strip())
            # Check if the page number is the next page
            if page_number == current_page + 1:
                next_page_button = page_button
                break

    # Wait for the search results to be displayed
    wait = WebDriverWait(driver, 30)
    wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#advSearch > div > div.panel.panel-info.search-results-header > div > h5")))

    # Get the header of the table
    header_element = driver.find_element(By.CSS_SELECTOR, "#cash3 > table > thead")
    header = [th.text for th in header_element.find_elements(By.CSS_SELECTOR, "th")]

        # Get the body of the table
    body_element = driver.find_element(By.CSS_SELECTOR, "#cash3 > table > tbody")
    rows = body_element.find_elements(By.CSS_SELECTOR, "tr")
    data = []
    for row in rows:
        data.append([td.text for td in row.find_elements(By.CSS_SELECTOR, "td")])
        for i in range(len(data)):
            data[i][2] = re.sub(r"(?<!\d)\s(?!\d)", "", data[i][2])

    # Save the data to a CSV file with page number as the file name
    with open(f"K:\Creative Cloud Files\python projs\galottery\csv test\page{current_page}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"data saved to page {current_page}.csv")
        
    # Check if the current page is 575
    if int(current_page) == 575:
        print("Reached last page. Scraping complete.")
        break

    # If there is a next page button, click it
    if next_page_button:
        print("Clicking next page button...")
        next_page_button.click()
        time.sleep(10)
    else:
        print("No more pages to scrape.")
        break

# Close the browser
print("Closing browser...")
driver.close()
