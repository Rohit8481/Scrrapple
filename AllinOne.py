import re
from pathlib import Path
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)

# Global Data
data = []
severity = []
responses = []

# Model Setup
device = torch.device("cpu")

severity_model_path = "haggue23/severity_detector_directory"
severity_model = BertForSequenceClassification.from_pretrained(severity_model_path).to(device)
severity_model.eval()
severity_tokenizer = BertTokenizer.from_pretrained(severity_model_path)

keybart_model_name = "bloomberg/KeyBART"
keybart_tokenizer = AutoTokenizer.from_pretrained(keybart_model_name)
keybart_model = AutoModelForSeq2SeqLM.from_pretrained(keybart_model_name)

toi_class = "Kt6Pm style_change T5Q6J"
COPYRIGHT_BOILERPLATE_WORDS = ["live updates", "breaking news", "copyright"]

def severe(text_input: str) -> str:
    tok = severity_tokenizer(text_input, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        output = severity_model(**tok)
        probabilities = F.softmax(output.logits, dim=1)[0] * 100
    predicted_class = torch.argmax(probabilities).item() + 1
    severity_map = {1: "MEDIUM", 2: "HIGH", 3: "LOW"}
    return severity_map.get(predicted_class, "UNKNOWN")

def clean_sentence(text: str, remove_words: list) -> str:
    sorted_words = sorted(remove_words, key=len, reverse=True)
    patterns = [re.escape(word) for word in sorted_words]
    regex_pattern = re.compile(r'\b(' + '|'.join(patterns) + r')\b', flags=re.IGNORECASE)
    cleaned_text = regex_pattern.sub("", text)
    return re.sub(r'\s+', ' ', cleaned_text).strip()

def short(cleaned_text: str) -> str:
    inputs = keybart_tokenizer(cleaned_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = keybart_model.generate(inputs["input_ids"], max_length=15, min_length=2, num_beams=4, early_stopping=True)
    raw_output = keybart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return re.split(r'[;,]', raw_output)[0].strip()

def scrape(page, url: str) -> str:
    page.goto(url, wait_until="domcontentloaded")
    
    if "ndtv.com" in url:
        try: page.wait_for_selector("h1, h3", timeout=15000)
        except Exception: pass
        content = page.content()
        soup = BeautifulSoup(content, "html.parser")
        headlines = [tag.get_text(separator=" ", strip=True) for tag in soup.find_all(["h1", "h3"]) if len(tag.get_text(strip=True)) > 15]
        return headlines[0] if headlines else "NDTV headline not found"

    elif "thehindu.com" in url:
        try: page.wait_for_selector("h1", timeout=15000)
        except Exception: pass
        content = page.content()
        soup = BeautifulSoup(content, "html.parser")
        head = soup.find("h1", class_="title") or soup.find("h1")
        return head.get_text(separator=" ", strip=True) if head else "The Hindu headline not found"

    else:
        try: page.wait_for_selector(f"div.{toi_class.replace(' ', '.')}", timeout=15000)
        except Exception: pass
        content = page.content()
        soup = BeautifulSoup(content, "html.parser")
        headline_div = soup.find("div", class_="Kt6Pm style_change T5Q6J")
        return headline_div.get_text(separator=" ", strip=True) if headline_div else "TOI headline not found"

if __name__ == "__main__":
    urls = [
        "https://www.ndtv.com",
        "https://www.thehindu.com/",
        "https://timesofindia.indiatimes.com/",
    ]

    # Run Playwright Sync in CLI / GitHub Actions
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()

        for url in urls:
            result = scrape(page, url)
            data.append(result.replace("'", "''"))

        browser.close()

    for text in data:
        severity.append(severe(text))
        cleaned_text = clean_sentence(text, COPYRIGHT_BOILERPLATE_WORDS)
        responses.append(short(cleaned_text))

    print("Data:", data)
    print("Severity:", severity)
    print("Responses:", responses)
