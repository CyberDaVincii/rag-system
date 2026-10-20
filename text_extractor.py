"""text_extractor.py: scrape one Wikipedia article into Selected_Document.txt.

Fetches the hardcoded URL with requests, parses it with BeautifulSoup, keeps
the text of every <p> tag inside Wikipedia's main content container
(<div class="mw-parser-output">), joins the paragraphs with blank lines, and
writes the result to Selected_Document.txt in UTF-8.

Two cleaning steps beyond the course example:
  1. Footnote markers such as [1] and [citation needed] are removed before the
     text is read, so they never reach the embeddings.
  2. Paragraph text is read with get_text() and its whitespace collapsed,
     instead of get_text(strip=True). strip=True trims every piece of text
     inside a paragraph and glues the pieces together, so linked words fuse
     with their neighbors: "Alarge language model(LLM) is anAI model".
"""
import re

import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/Large_language_model"
OUTPUT_FILE = "Selected_Document.txt"

# Wikipedia asks scripts to identify themselves; the default python-requests
# user agent can be refused with HTTP 403.
HEADERS = {"User-Agent": "AIE-RAG-student-project/1.0 (Full Sail University coursework)"}


def fetch_and_extract(url):
    """Fetch the page, extract its paragraph text, save it, and return it."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as e:
        print(f"An error occurred while fetching the URL: {e}")
        return ""

    if response.status_code != 200:
        print(f"Failed to retrieve the page. HTTP Status Code: {response.status_code}")
        return ""

    soup = BeautifulSoup(response.text, "html.parser")
    content_div = soup.find("div", class_="mw-parser-output")
    if not content_div:
        print("The expected content container was not found.")
        return ""

    # Remove footnote markers ([1], [2], [citation needed]) and inline styles.
    for tag in content_div.select("sup.reference, sup.noprint, style"):
        tag.decompose()

    paragraphs = []
    for p in content_div.find_all("p"):
        paragraph = " ".join(p.get_text().split())          # collapse every run of whitespace to one space
        paragraph = re.sub(r"\s+([.,;:!?)])", r"\1", paragraph)   # no space before punctuation
        if paragraph:
            paragraphs.append(paragraph)

    extracted_text = "\n\n".join(paragraphs)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(extracted_text)

    print(f"Page successfully retrieved and content saved to '{OUTPUT_FILE}' "
          f"({len(paragraphs)} paragraphs, {len(extracted_text):,} characters).")
    return extracted_text


def main():
    fetch_and_extract(URL)


if __name__ == "__main__":
    main()
