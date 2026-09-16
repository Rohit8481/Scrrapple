import torch
import os
import torch.nn.functional as F
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BartForConditionalGeneration,
    BartTokenizer
)
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


# ============================================================
# ARRAYS
# ============================================================

severity = []
data = []
responses = []
links = []


# ============================================================
# SEVERITY MODEL
# ============================================================

model_path = os.getenv("MODEL_PATH")

severity_tokenizer = BertTokenizer.from_pretrained(model_path)

severity_model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cpu")
severity_model.to(device)
severity_model.eval()


def severe(data):

    tok = severity_tokenizer(
        data,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    tok = {k: v.to(device) for k, v in tok.items()}

    with torch.no_grad():

        output = severity_model(**tok)

        probs = F.softmax(output.logits, dim=1)

        prediction = torch.argmax(probs, dim=1).item()

    label_map = {
        0: "CRITICAL",
        1: "IMPORTANT",
        2: "AVERAGE",
        3: "LOW"
    }

    return label_map.get(prediction, "UNKNOWN")


# ============================================================
# KEYBART MODEL
# ============================================================

keybart_model_name = "bloomberg/KeyBART"

keybart_tokenizer = BartTokenizer.from_pretrained(keybart_model_name)

keybart_model = BartForConditionalGeneration.from_pretrained(
    keybart_model_name
)

keybart_model.to(device)
keybart_model.eval()


def short(text):

    inputs = keybart_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        output = keybart_model.generate(
            **inputs,
            max_length=10,
            min_length=2,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    result = keybart_tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    # Only take text before ;
    result = result.split(";")[0].strip()

    return result


# ============================================================
# SCRAPER
# ============================================================

async def scrape(url):

    async with async_playwright() as p:

        browser = await p.firefox.launch(
            headless=True
        )

        page = await browser.new_page(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            viewport={
                "width": 1280,
                "height": 720
            }
        )

        try:

            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=60000
            )

            # ==================================================
            # NDTV
            # ==================================================

            if "ndtv.com" in url:

                await page.wait_for_selector(
                    "h1, h3",
                    timeout=15000
                )

                html = await page.content()

                soup = BeautifulSoup(
                    html,
                    "html.parser"
                )

                headline = None
                news_link = None

                elements = soup.find_all(
                    ["h1", "h3"]
                )

                for element in elements:

                    text = element.get_text(
                        " ",
                        strip=True
                    )

                    if len(text) > 15:

                        headline = text

                        # Check if headline itself is inside <a>
                        parent_a = element.find_parent("a")

                        if parent_a and parent_a.get("href"):
                            news_link = parent_a.get("href")

                        # Check for <a> inside headline
                        if not news_link:

                            a_tag = element.find("a")

                            if a_tag and a_tag.get("href"):
                                news_link = a_tag.get("href")

                        break


            # ==================================================
            # INDIAN EXPRESS
            # ==================================================

            elif "indianexpress.com" in url:

                await page.wait_for_selector(
                    "h1",
                    timeout=15000
                )

                html = await page.content()

                soup = BeautifulSoup(
                    html,
                    "html.parser"
                )

                element = soup.select_one(
                    "h1.topblockNews__featuredTitle"
                )

                if not element:
                    element = soup.find("h1")

                headline = None
                news_link = None

                if element:

                    headline = element.get_text(
                        " ",
                        strip=True
                    )

                    parent_a = element.find_parent("a")

                    if parent_a and parent_a.get("href"):
                        news_link = parent_a.get("href")

                    if not news_link:

                        a_tag = element.find("a")

                        if a_tag and a_tag.get("href"):
                            news_link = a_tag.get("href")


            # ==================================================
            # THE HINDU
            # ==================================================

            elif "thehindu.com" in url:

                await page.wait_for_selector(
                    "h1",
                    timeout=15000
                )

                html = await page.content()

                soup = BeautifulSoup(
                    html,
                    "html.parser"
                )

                element = soup.select_one(
                    "h1.title"
                )

                if not element:
                    element = soup.find("h1")

                headline = None
                news_link = None

                if element:

                    headline = element.get_text(
                        " ",
                        strip=True
                    )

                    parent_a = element.find_parent("a")

                    if parent_a and parent_a.get("href"):
                        news_link = parent_a.get("href")

                    if not news_link:

                        a_tag = element.find("a")

                        if a_tag and a_tag.get("href"):
                            news_link = a_tag.get("href")


            # ==================================================
            # HINDUSTAN TIMES
            # ==================================================

            elif "hindustantimes.com" in url:

                await page.wait_for_selector(
                    "h2",
                    timeout=15000
                )

                html = await page.content()

                soup = BeautifulSoup(
                    html,
                    "html.parser"
                )

                element = soup.select_one(
                    "h2.hdg3"
                )

                if not element:
                    element = soup.find("h2")

                headline = None
                news_link = None

                if element:

                    headline = element.get_text(
                        " ",
                        strip=True
                    )

                    parent_a = element.find_parent("a")

                    if parent_a and parent_a.get("href"):
                        news_link = parent_a.get("href")

                    if not news_link:

                        a_tag = element.find("a")

                        if a_tag and a_tag.get("href"):
                            news_link = a_tag.get("href")


            # ==================================================
            # TIMES OF INDIA
            # ==================================================

            else:

                await page.wait_for_selector(
                    "div.Kt6Pm.style_change.T5Q6J",
                    timeout=15000
                )

                html = await page.content()

                soup = BeautifulSoup(
                    html,
                    "html.parser"
                )

                element = soup.select_one(
                    "div.Kt6Pm.style_change.T5Q6J"
                )

                headline = None
                news_link = None

                if element:

                    headline = element.get_text(
                        " ",
                        strip=True
                    )

                    parent_a = element.find_parent("a")

                    if parent_a and parent_a.get("href"):
                        news_link = parent_a.get("href")

                    if not news_link:

                        a_tag = element.find("a")

                        if a_tag and a_tag.get("href"):
                            news_link = a_tag.get("href")


            # ==================================================
            # MAKE LINK ABSOLUTE
            # ==================================================

            if news_link:

                if news_link.startswith("//"):
                    news_link = "https:" + news_link

                elif news_link.startswith("/"):
                    from urllib.parse import urljoin
                    news_link = urljoin(url, news_link)

            return headline, news_link

        except Exception as e:

            print(
                f"Error scraping {url}: {e}"
            )

            return None, None

        finally:

            await browser.close()


# ============================================================
# WEBSITE URLS
# ============================================================

urls = [

    "https://www.ndtv.com/",
    "https://www.thehindu.com/",
    "https://timesofindia.indiatimes.com/",
    "https://www.hindustantimes.com/",
    "https://indianexpress.com/"

]


# ============================================================
# SCRAPE NEWS
# ============================================================

import asyncio

for i in urls:

    result = asyncio.run(
        scrape(i)
    )

    if result:

        headline, news_link = result

        if headline:

            headline = headline.replace(
                "'",
                "''"
            )

            data.append(
                headline
            )

            links.append(
                news_link
            )

            print(
                "Headline:",
                headline
            )

            print(
                "Link:",
                news_link
            )

            print(
                "--------------------------------"
            )


# ============================================================
# SEVERITY
# ============================================================

for i in data:

    try:

        result = severe(i)

        severity.append(
            result
        )

        print(
            "Severity:",
            result
        )

    except Exception as e:

        severity.append(
            "UNKNOWN"
        )

        print(
            "Severity Error:",
            e
        )


# ============================================================
# SHORT HEADLINE
# ============================================================

for i in data:

    try:

        result = short(i)

        responses.append(
            result
        )

        print(
            "Short:",
            result
        )

    except Exception as e:

        responses.append(
            "UNKNOWN"
        )

        print(
            "Short Headline Error:",
            e
        )


# ============================================================
# FINAL CHECK
# ============================================================

print("\n==============================")

print(
    "Data:",
    len(data)
)

print(
    "Severity:",
    len(severity)
)

print(
    "Responses:",
    len(responses)
)

print(
    "Links:",
    len(links)
)

print("==============================\n")

print("DATA:")
print(data)

print("\nSEVERITY:")
print(severity)

print("\nRESPONSES:")
print(responses)

print("\nLINKS:")
print(links)
