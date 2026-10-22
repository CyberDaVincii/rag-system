"""text_extractor_web.py: extract a web article's readable text into Selected_Document.txt.

Generated from the assignment's web-extractor prompt, then cleaned up after
checking the output:

- Uses the page's <article> element when there is one; otherwise falls back to
  Wikipedia's main content container (#mw-content-text .mw-parser-output).
- Keeps headings, paragraphs, and list items, in page order, joined with blank lines.
- Removes things that are not article text before extracting: footnote markers
  like [1], the "[edit]" links beside headings, the reference list, navigation
  boxes, sidebars, and inline styles. Without this, the reference list alone
  adds hundreds of citation lines to the document.
- Reads each element with get_text(" ") and collapses whitespace. The common
  get_text(strip=True) glues linked words to their neighbors
  ("Alarge language model(LLM) is amachine learning model").
- Sends a descriptive User-Agent; Wikipedia refuses the default Python client
  with HTTP 403.
"""
import re
import sys

import requests
from bs4 import BeautifulSoup

URL = "https://en.wikipedia.org/wiki/Large_language_model"
OUTPUT_FILE = "Selected_Document.txt"
HEADERS = {"User-Agent": "AIE-RAG-student-project/1.0 (Full Sail University coursework)"}

# Page furniture that is not article text.
NOISE_SELECTORS = [
    "sup.reference", "sup.noprint", ".mw-editsection", "style", "script",
    ".reflist", "ol.references", ".navbox", ".vertical-navbox", ".sidebar",
    ".metadata", ".mw-empty-elt", ".shortdescription", ".hatnote",
]
# Sections that are lists of links or citations rather than content.
STOP_HEADINGS = {"see also", "references", "notes", "further reading", "external links", "bibliography"}


def clean(text):
    """Collapse whitespace and remove stray spaces around brackets and punctuation."""
    text = " ".join(text.split())
    text = re.sub(r"([(\[])\s+", r"\1", text)          # no space after an opening bracket
    return re.sub(r"\s+([.,;:!?)\]])", r"\1", text)    # no space before closing punctuation


def extract_text(url=URL, output_file=OUTPUT_FILE):
    """Fetch the page, extract headings, paragraphs and list items, save them, and return the text."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    if response.status_code != 200:
        print(f"Request failed with HTTP status {response.status_code}.")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    container = soup.find("article") or soup.select_one("#mw-content-text .mw-parser-output")
    if container is None:
        print("No article content was found on the page.")
        return None

    for tag in container.select(", ".join(NOISE_SELECTORS)):
        tag.decompose()

    blocks = []
    for element in container.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]):
        if element.name == "li" and element.find_parent(["li"]):
            continue                                     # nested list items are already inside their parent
        text = clean(element.get_text(" "))
        if not text:
            continue
        if element.name.startswith("h") and text.lower() in STOP_HEADINGS:
            break                                        # everything after this point is citations and links
        blocks.append(text)

    document = "\n\n".join(blocks)
    if not document.strip():
        print("The article content was empty after cleaning.")
        return None

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(document)
    print(f"Saved {len(blocks)} blocks ({len(document):,} characters) to {output_file}.")
    return document


def main():
    if extract_text() is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
