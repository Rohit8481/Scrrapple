import asyncio
import re
from pathlib import Path
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertForSequenceClassification,
    BertTokenizer,
)

# --- 1. Global Module Data (Lists accessible from external scripts) ---

data: list = []
severity: list = []
responses: list = []


# --- 2. Global Setup & Model Loading ---

severity_model_path = "haggue23/severity_detector_directory"
severity_model = BertForSequenceClassification.from_pretrained(severity_model_path)
severity_tokenizer = BertTokenizer.from_pretrained(severity_model_path)

device = torch.device("cpu")
severity_model.to(device)
severity_model.eval()

keybart_model_name = "bloomberg/KeyBART"
keybart_tokenizer = AutoTokenizer.from_pretrained(keybart_model_name)
keybart_model = AutoModelForSeq2SeqLM.from_pretrained(keybart_model_name)

toi_class = "Kt6Pm style_change T5Q6J"
COPYRIGHT_BOILERPLATE_WORDS = [
    "live updates", "live update", "breaking news", "live news", "live blog", "live coverage",
    "just in", "developing story", "watch live", "watch video", "exclusive", "special report",
    "top stories", "latest news", "flash news", "news alert", "trending now",
    "all rights reserved", "rights reserved", "copyright", "copyrighted", "courtesy", 
    "courtesy of", "source", "photo credit", "image source", "disclaimer", "terms of use", 
    "privacy policy", "reproduction prohibited", "published by", "reported by", "file photo",
    "ndtv", "ndtv live", "reuters", "associated press", "ap news", "afp", "bloomberg", 
    "bbc", "bbc news", "cnn", "fox news", "al jazeera", "times of india", "toi", 
    "hindustan times", "the hindu", "indian express", "aaj tak", "zee news", "indiatv", 
    "ani news", "pti", "press trust of india", "financial times", "wall street journal", 
    "wsj", "the guardian", "new york times", "nyt", "washington post", "forbes",
    "com", "org", "net", "in", "co", "co.in", "gov", "edu", "info", "io", "news", 
    "www", "http", "https", "dot com",
    "click here", "read more", "subscribe", "follow us", "share", "tweet", "retweet", 
    "facebook", "twitter", "x.com", "instagram", "youtube", "telegram", "whatsapp", 
    "podcast", "newsletter", "advertisement", "sponsored", "editorial", "opinion",
    "view original", "full story", "full report"
]


# --- 3. Processing Functions ---

def severe(text_input: str) -> str:
    """Classifies severity level of text using BERT."""
    tok = severity_tokenizer(
        text_input,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = severity_model(**tok)
        logits = output.logits
        probabilities = F.softmax(logits, dim=1)[0] * 100

    predicted_class = torch.argmax(probabilities).item() + 1

    severity_map = {
        1: "MEDIUM",
        2: "HIGH",
        3: "LOW"
    }
    return severity_map.get(predicted_class, "UNKNOWN")


def clean_sentence(text: str, remove_words: list) -> str:
    """Removes media boilerplate terms and extra formatting."""
    sorted_words = sorted(remove_words, key=len, reverse=True)
    patterns = [re.escape(word) for word in sorted_words]
    
    regex_pattern = re.compile(r'\b(' + '|'.join(patterns) + r')\b', flags=re.IGNORECASE)
    cleaned_text = regex_pattern.sub("", text)
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = re.sub(r'^\s*[\:\-\|\,\.\?]+\s*', '', cleaned_text)
    cleaned_text = re.sub(r'\s*[\:\-\|\,\.\?]+\s*$', '', cleaned_text)
    cleaned_text = re.sub(r'\s+([\:\-\|\,])', r'\1', cleaned_text)
    
    return cleaned_text.strip()


def short(cleaned_text: str) -> str:
    """Extracts key phrase using KeyBART."""
    inputs = keybart_tokenizer(cleaned_text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = keybart_model.generate(
        inputs["input_ids"], 
        max_length=15,
        min_length=2,
        num_beams=4, 
        length_penalty=0.6,
        early_stopping=True
    )

    raw_output = keybart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    single_keyphrase = re.split(r'[;,]', raw_output)[0].strip()
    return single_keyphrase


async def scrape(page, url: str) -> str:
    """Scrapes headline given an active Playwright page instance."""
    await page.goto(url, wait_until="domcontentloaded")
    headline = None

    if "ndtv.com" in url:
        try:
            await page.wait_for_selector("h1, h3", timeout=15000)
        except Exception:
            pass

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        headlines = [
            tag.get_text(separator=" ", strip=True) 
            for tag in soup.find_all(["h1", "h3"]) 
            if len(tag.get_text(strip=True)) > 15
        ]
        headline = headlines[0] if headlines else "NDTV headline not found"

    elif "thehindu.com" in url:
        try:
            await page.wait_for_selector("h1", timeout=15000)
        except Exception:
            pass

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        head = soup.find("h1", class_="title") or soup.find("h1")
        headline = head.get_text(separator=" ", strip=True) if head else "The Hindu headline not found"

    else:
        try:
            await page.wait_for_selector(f"div.{toi_class.replace(' ', '.')}", timeout=15000)
        except Exception:
            pass

        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        headline_div = soup.find("div", class_="Kt6Pm style_change T5Q6J")
        headline = headline_div.get_text(separator=" ", strip=True) if headline_div else "TOI headline not found"

    return headline


# --- 4. Pipeline Execution Function ---

async def run_pipeline() -> tuple[list, list, list]:
    """Runs pipeline and populates global lists in-place."""
    global data, severity, responses

    urls = [
        "https://www.ndtv.com",
        "https://www.thehindu.com/",
        "https://timesofindia.indiatimes.com/",
    ]

    temp_data = []
    temp_severity = []
    temp_responses = []

    # Playwright Scraping
    async with async_playwright() as p:
        browser = await p.firefox.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        for url in urls:
            result = await scrape(page, url)
            headline_clean = result.replace("'", "''")
            temp_data.append(headline_clean)

        await browser.close()

    # Classification & Keyphrase extraction
    for text in temp_data:
        temp_severity.append(severe(text))
        cleaned_text = clean_sentence(text, COPYRIGHT_BOILERPLATE_WORDS)
        temp_responses.append(short(cleaned_text))

    # Clear and update global lists in-place to preserve imports in other modules
    data.clear()
    data.extend(temp_data)

    severity.clear()
    severity.extend(temp_severity)

    responses.clear()
    responses.extend(temp_responses)

    return data, severity, responses


if __name__ == "__main__":
    asyncio.run(run_pipeline())
    print("\n--- Pipeline Execution Output ---")
    print("data =", data)
    print("severity =", severity)
    print("responses =", responses)
