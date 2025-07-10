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

    column_header_cells = driver.find_elements(By.XPATH, "//thead/tr/td[contains(@class, 'factor')]")
    col_labels = []
    for cell in column_header_cells:
        label = cell.text.strip()
        colspan = int(cell.get_attribute("colspan") or 1)
        col_labels.extend([label] * colspan)

    first_value_row = driver.find_elements(By.XPATH, "//tr[@id='tr_first_values_row']/td/span")
    for idx, span in enumerate(first_value_row):
        result[col_labels[idx]].append(span.text.strip())

    # ===== ROW HEADERS (fixing rowspan) =====
    tbody_rows = driver.find_elements(By.XPATH, "//tbody/tr")
    row_labels_by_index = {}
    current_label = None
    remaining_span = 0

    for i, row in enumerate(tbody_rows):
        label_td = row.find_elements(By.XPATH, "./td[@class='td-vertical factor']/span")
        if label_td:
            current_label = label_td[0].text.strip()
            rowspan = int(row.find_element(By.XPATH, "./td[@class='td-vertical factor']").get_attribute("rowspan") or 1)
            remaining_span = rowspan

        if current_label and remaining_span > 0:
            row_labels_by_index[i] = current_label
            remaining_span -= 1
            if remaining_span == 0:
                current_label = None

    # Now map actual values under those row labels
    for i, row in enumerate(tbody_rows):
        label = row_labels_by_index.get(i)
        value_cells = row.find_elements(By.XPATH, "./td[@class='rowValues rowValuesLeft']")
        if label:
            if value_cells:
                result[label].append(value_cells[0].text.strip())
            elif row.text:
                result[label].append(row.text.strip())
        
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