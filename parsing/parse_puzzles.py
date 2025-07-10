from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import json
from collections import defaultdict

def accept_cookies():
    driver.find_element(By.CSS_SELECTOR, "div[format=primary]").click()
    driver.find_element(By.CSS_SELECTOR, "button.cmplz-accept").click()

def extract_links():
    elems = driver.find_elements(By.CSS_SELECTOR, "a.index_lp_link")
    ans = {}
    for el in elems:
        print(el.text)
        ans[el.text] = el.get_attribute("href")
    return ans

def extract_permissible_feats():
    result = defaultdict(list)

    # Get column group headers from <thead> (e.g., "Age", "Toy")
    column_header_cells = driver.find_elements(By.XPATH, "//thead/tr/td[contains(@class, 'factor')]")
    col_labels = []
    col_counts = []

    for cell in column_header_cells:
        label = cell.text.strip()
        colspan = int(cell.get_attribute("colspan") or 1)
        col_labels.extend([label] * colspan)
        col_counts.append(colspan)

    # Get specific column values from the first <tbody> row
    first_value_row = driver.find_elements(By.XPATH, "//tr[@id='tr_first_values_row']/td/span")
    for idx, span in enumerate(first_value_row):
        group_label = col_labels[idx]  # dynamically mapped from headers
        result[group_label].append(span.text.strip())

    # Get row labels (e.g., "Name", "Toy") from first column of body rows
    factor_rows = driver.find_elements(By.XPATH, "//tbody/tr")
    for row in factor_rows:
        first_td = row.find_elements(By.XPATH, "./td[@class='rowValues rowValuesLeft']")
        if first_td:
            # Look for closest <td class="td-vertical factor"> up the row stack
            preceding_label_td = row.find_elements(By.XPATH, "./preceding-sibling::tr/td[@class='td-vertical factor']/span")
            if preceding_label_td:
                label = preceding_label_td[-1].text.strip()
                result[label].append(first_td[0].text.strip())
    for k in result:
        result[k] = list(set(result[k]))
    
    return result


def extract_puzzles(links):
    for name, url in links.items():
        driver.get(url)

        permissible_items = extract_permissible_feats()
        permissible_items_text = "\nThe permissible items are: "

        for item in permissible_items:
            permissible_items_text += f"\n{item}: {permissible_items[item]}"

        puzzle = {"description": driver.find_element(By.ID, "puzzle_desc").text + permissible_items_text,
                  "clues": []}


        clue_id = 1

        while True:
            clue_name = f"clue{clue_id}"
            try:
                el = driver.find_element(By.ID, clue_name)
                clue = el.text.replace(f"{clue_id}) ", "")
                puzzle["clues"].append({"id": clue_id, "text": clue})
            except NoSuchElementException:
                break
            clue_id += 1

        with open("puzzles/" + name.lower().replace(" ", "_") + ".json", "w") as f:
            json.dump(puzzle, f, indent=4)
        print(puzzle, "\n","="*8)


        
if __name__ == "__main__":
    driver = webdriver.Firefox()
    driver.get("https://daydreampuzzles.com/logic-grid-puzzles/")

    accept_cookies()
    links = extract_links()
    extract_puzzles(links)
    driver.close()