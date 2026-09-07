import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)

# ============================================================
# WEBSITE SETTINGS
# ============================================================

toi_class = "Kt6Pm style_change T5Q6J"


# ============================================================
# GLOBAL LISTS
# database.py will import these
# ============================================================

severity = []
data = []
responses = []


# ============================================================
# SEVERITY MODEL
# ============================================================

model_path = "haggue23/severity_detector_directory"

print("Loading severity model...")

severity_model = BertForSequenceClassification.from_pretrained(model_path)
severity_tokenizer = BertTokenizer.from_pretrained(model_path)

severity_device = torch.device("cpu")

severity_model.to(severity_device)
severity_model.eval()

print("Severity model loaded")


def severe(data):
    tok = severity_tokenizer(
        data,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    tok["input_ids"] = tok["input_ids"].to(severity_device)
    tok["attention_mask"] = tok["attention_mask"].to(severity_device)

    with torch.no_grad():
        output = severity_model(**tok)
        logits = output.logits
        probabilities = F.softmax(logits, dim=1)[0] * 100

    predicted_class = torch.argmax(probabilities).item() + 1

    severity_map = {1: "IMPORTANT", 2: "CRITICAL", 3: "LOW"}

    return severity_map[predicted_class]


# ============================================================
# KEYBART MODEL
# ============================================================

keybart_model_name = "bloomberg/KeyBART"

print("Loading KeyBART model...")

keybart_tokenizer = BartTokenizer.from_pretrained(keybart_model_name)
keybart_model = BartForConditionalGeneration.from_pretrained(
    keybart_model_name
)

keybart_model.to(torch.device("cpu"))
keybart_model.eval()

print("KeyBART model loaded")


def short(text):
    inputs = keybart_tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True
    )

    with torch.no_grad():
        summary_ids = keybart_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=10,
            min_length=2,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2,
        )

    raw_output = keybart_tokenizer.decode(
        summary_ids[0], skip_special_tokens=True
    ).strip()

    # Take only the first phrase
    first_phrase = raw_output.split(";")[0].strip()

    return first_phrase


# ============================================================
# SCRAPING FUNCTION
# ============================================================


def scrape(url):
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
                "Gecko/20100101 Firefox/125.0"
            ),
            viewport={"width": 1920, "height": 1080},
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.5",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "DNT": "1",
                "Upgrade-Insecure-Requests": "1",
            },
        )

        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        headline = None

        # ====================================================
        # INDIAN EXPRESS
        # ====================================================
        if "indianexpress.com" in url:
            page.wait_for_selector("h1", timeout=15000)

            soup = BeautifulSoup(page.content(), "html.parser")

            # Try finding by specific class first, fallback to general h1
            heading_tags = soup.find_all(
                "h1", class_="topblockNews__featuredTitle"
            ) or soup.find_all("h1")

            for tag in heading_tags:
                text = tag.get_text(separator=" ", strip=True)
                if text and len(text) > 15:
                    headline = text
                    break

            if not headline:
                headline = "Indian Express headline not found"

        # ====================================================
        # NDTV
        # ====================================================
        elif "ndtv.com" in url:
            page.wait_for_selector("h1, h3", timeout=15000)

            soup = BeautifulSoup(page.content(), "html.parser")

            headlines = []
            for tag in soup.find_all(["h1", "h3"]):
                text = tag.get_text(separator=" ", strip=True)
                if text and len(text) > 15:
                    headlines.append(text)

            headline = (
                headlines[0] if headlines else "NDTV headlines not found"
            )

        # ====================================================
        # THE HINDU
        # ====================================================
        elif "thehindu.com" in url:
            page.wait_for_selector("h1", timeout=15000)

            soup = BeautifulSoup(page.content(), "html.parser")

            head = soup.find("h1", class_="title")

            if head:
                headline = head.get_text(separator=" ", strip=True)
            else:
                head_any = soup.find("h1")
                headline = (
                    head_any.get_text(separator=" ", strip=True)
                    if head_any
                    else "The Hindu headline not found"
                )

        # ====================================================
        # TIMES OF INDIA
        # ====================================================
        else:
            page.wait_for_selector(
                f"div.{toi_class.replace(' ', '.')}", timeout=15000
            )

            soup = BeautifulSoup(page.content(), "html.parser")

            headline_div = soup.find("div", class_=toi_class)

            if headline_div:
                headline = headline_div.get_text(separator=" ", strip=True)
            else:
                headline = "TOI headline not found"

        browser.close()

        return headline


# ============================================================
# URLS
# ============================================================

Urls = [
    "https://indianexpress.com/",
    "https://www.ndtv.com",
    "https://www.thehindu.com/",
    "https://timesofindia.indiatimes.com/",
]


# ============================================================
# SCRAPING
# ============================================================

for i in Urls:
    try:
        result = scrape(i)

        if result:
            headline_clean = result.replace("'", "''")
            data.append(headline_clean)
            print(f"Scraped ({i}): {headline_clean}")

    except Exception as e:
        print(f"Error scraping {i}: {e}")

print("scraping done")


# ============================================================
# SEVERITY
# ============================================================

for i in data:
    try:
        result = severe(i)
        severity.append(result)
        print(f"Severity: {result}")
    except Exception as e:
        print(f"Severity error: {e}")
        severity.append("UNKNOWN")

print("severity done")


# ============================================================
# KEYBART SHORT HEADLINES
# ============================================================

for i in data:
    try:
        result = short(i)
        responses.append(result)
        print(f"Short: {result}")
    except Exception as e:
        print(f"KeyBART error: {e}")
        responses.append("")

print("response done")


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("================================")
print("FINAL RESULTS")
print("================================")

print("Headlines:", len(data))
print("Severity:", len(severity))
print("Short headlines:", len(responses))

print("================================")
