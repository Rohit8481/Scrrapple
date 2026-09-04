import torch
import torch.nn.functional as F
from transformers import BertForSequenceClassification, BertTokenizer
from bs4 import BeautifulSoup
from google import genai
from playwright.sync_api import sync_playwright

toi_class = "Kt6Pm style_change T5Q6J"
ndtv_class = "crd_lnk"

severity =[]
data =[]
responses =[]

def severe(data) : 
# 1. Load Model & Tokenizer
    model_path = r"newz_app\severity_detector"
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    tok = tokenizer(
        data,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    tok["input_ids"] = tok["input_ids"].to(device)
    tok["attention_mask"] = tok["attention_mask"].to(device)

    # 4. Inference & Softmax
    with torch.no_grad():
        output = model(**tok)
        logits = output.logits  # Shape: [1, 3]

        # Convert raw logits to percentage probabilities (0% - 100%)
        probabilities = F.softmax(logits, dim=1)[0] * 100
    
    predicted_class = torch.argmax(probabilities).item() + 1 # 1-indexed (1, 2, or 3)

    severity_map = {1: "MEDIUM", 2: "HIGH", 3: "LOW"}

    return severity_map[predicted_class]


def scrape(url):
    with sync_playwright() as p:
        browser = p.firefox.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            viewport={"width": 1920, "height": 1080},
        )
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")

        headline = None

        if url == "https://www.ndtv.com":
            page.wait_for_selector("h1, h3", timeout=15000)

            # Re-fetch HTML after waiting
            soup = BeautifulSoup(page.content(), "html.parser")
            headlines = []

            for tag in soup.find_all(["h1", "h3"]):
                text = tag.get_text(separator=" ", strip=True)
                if text and len(text) > 15:
                    headlines.append(text)

            # Combine or take the top headline
            headline = headlines[0] if headlines else "NDTV headlines not found"

        elif url == "https://www.thehindu.com/":
            page.wait_for_selector("h1", timeout=15000)

            # Re-fetch HTML after waiting
            soup = BeautifulSoup(page.content(), "html.parser")
            head = soup.find("h1", class_="title")

            if head:
                headline = head.get_text(separator=" ", strip=True)
            else:
                # Fallback to any h1 if class='title' isn't used
                head_any = soup.find("h1")
                headline = (
                    head_any.get_text(separator=" ", strip=True)
                    if head_any
                    else "The Hindu headline not found"
                )

        else:
            page.wait_for_selector(
                f"div.{toi_class.replace(' ', '.')}", timeout=15000
            )

            # Re-fetch HTML after waiting
            soup = BeautifulSoup(page.content(), "html.parser")
            headline_div = soup.find("div", class_="Kt6Pm style_change T5Q6J")

            if headline_div:
                headline = headline_div.get_text(separator=" ", strip=True)
            else:
                headline = "TOI headline not found"

        browser.close()
        return headline
        


Urls = [
    "https://www.ndtv.com",
    "https://www.thehindu.com/",
    "https://timesofindia.indiatimes.com/",
]

for i in Urls:
    result = scrape(i)
    headline_clean = result.replace("'", "''")
    data.append(headline_clean)
print("scraping done")    

for i in data:
    severity.append(severe(i))
print("severity done")

# Pass the API key using the keyword argument `api_key=`
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found. Check your .env file."
    )

client = genai.Client(api_key=api_key)

responses = []

for i in data:
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=f"""
Rewrite the following headline into EXACTLY 3 words.
Respond with ONLY those 3 words—no quotes, no punctuation,
and no additional text.

Original Headline: {i}
"""
    )

    responses.append(response.text.strip())

print("response done")
print(responses)