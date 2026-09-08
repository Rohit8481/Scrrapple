import torch
import torch.nn.functional as F
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BartForConditionalGeneration,
    BartTokenizer
)
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


# ============================================================
# WEBSITE SETTINGS
# ============================================================

toi_class = "Kt6Pm style_change T5Q6J"
ndtv_class = "crd_lnk"


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

model_path = "haggue23/severity_detector_ai"

print("Loading severity model...")

severity_model = BertForSequenceClassification.from_pretrained(
    model_path
)

severity_tokenizer = BertTokenizer.from_pretrained(
    model_path
)

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
        return_tensors="pt"
    )

    tok = {k: v.to(severity_device) for k, v in tok.items()}

    with torch.no_grad():
        output = severity_model(**tok)

    predicted_class = torch.argmax(output.logits, dim=1).item()

    severity_map = {
        0: "CRITICAL",
        1: "IMPORTANT",
        2: "AVERAGE",
        3: "LOW"
    }

    return severity_map[predicted_class]


# ============================================================
# KEYBART MODEL
# ============================================================

keybart_model_name = "bloomberg/KeyBART"

print("Loading KeyBART model...")

keybart_tokenizer = BartTokenizer.from_pretrained(
    keybart_model_name
)

keybart_model = BartForConditionalGeneration.from_pretrained(
    keybart_model_name
)

keybart_model.to(torch.device("cpu"))
keybart_model.eval()

print("KeyBART model loaded")


def short(text):

    inputs = keybart_tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    with torch.no_grad():

        summary_ids = keybart_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=10,
            min_length=2,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )

    raw_output = keybart_tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    ).strip()

    # Take only the first phrase
    first_phrase = raw_output.split(";")[0].strip()

    # Add semicolon at the end
    return first_phrase 


# ============================================================
# SCRAPING FUNCTION
# ============================================================

def scrape(url):

    with sync_playwright() as p:

        browser = p.firefox.launch(
            headless=True
        )

        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
                "Gecko/20100101 Firefox/125.0"
            ),
            viewport={
                "width": 1920,
                "height": 1080
            },
        )

        page = context.new_page()

        page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=60000
        )

        headline = None

        # ====================================================
        # NDTV
        # ====================================================

        if url == "https://www.ndtv.com":

            page.wait_for_selector(
                "h1, h3",
                timeout=15000
            )

            soup = BeautifulSoup(
                page.content(),
                "html.parser"
            )

            headlines = []

            for tag in soup.find_all(
                ["h1", "h3"]
            ):

                text = tag.get_text(
                    separator=" ",
                    strip=True
                )

                if text and len(text) > 15:
                    headlines.append(text)

            headline = (
                headlines[0]
                if headlines
                else "NDTV headlines not found"
            )

        
        elif url == "https://indianexpress.com/":
            page.wait_for_selector(
                            "h1",
                            timeout=40000
                        )
            soup = BeautifulSoup(
                            page.content(),
                            "html.parser"
                        )
            head =  soup.find("h1", class_="topblockNews__featuredTitle") or soup.find("h1")
            text = head.get_text(separator=" ", strip=True)
            
            if text and len(text) > 15:
                headline = text
        # ====================================================
        # THE HINDU
        # ====================================================

        elif url == "https://www.thehindu.com/":

            page.wait_for_selector(
                "h1",
                timeout=15000
            )

            soup = BeautifulSoup(
                page.content(),
                "html.parser"
            )

            head = soup.find(
                "h1",
                class_="title"
            )

            if head:

                headline = head.get_text(
                    separator=" ",
                    strip=True
                )

            else:

                head_any = soup.find("h1")

                headline = (
                    head_any.get_text(
                        separator=" ",
                        strip=True
                    )
                    if head_any
                    else "The Hindu headline not found"
                )
            

        elif url == "https://www.hindustantimes.com/india-news" : 
            
            page.wait_for_selector(
                            "h2",
                            timeout=15000
                        )
            
            soup = BeautifulSoup(
                page.content(),
                "html.parser"
            )

            head = soup.find(
                "h2",
                class_="hdg3"
            )

            if head:

                headline = head.get_text(
                    separator=" ",
                    strip=True
                )

            else:

                head_any = soup.find("h2")

                headline = (
                    head_any.get_text(
                        separator=" ",
                        strip=True
                    )
                    if head_any
                    else "The HindustanTimes headline not found"
                )

        # ====================================================
        # TIMES OF INDIA
        # ====================================================

        else:

            page.wait_for_selector(
                f"div.{toi_class.replace(' ', '.')}",
                timeout=15000
            )

            soup = BeautifulSoup(
                page.content(),
                "html.parser"
            )

            headline_div = soup.find(
                "div",
                class_="Kt6Pm style_change T5Q6J"
            )

            if headline_div:

                headline = headline_div.get_text(
                    separator=" ",
                    strip=True
                )

            else:

                headline = "TOI headline not found"

        browser.close()

        return headline


# ============================================================
# URLS
# ============================================================

Urls = [
    "https://www.ndtv.com",
    "https://www.thehindu.com/",
    "https://timesofindia.indiatimes.com/",
    "https://www.hindustantimes.com/india-news",
    "https://indianexpress.com/"

]


# ============================================================
# SCRAPING
# ============================================================

for i in Urls:

    try:

        result = scrape(i)

        if result:

            headline_clean = result.replace(
                "'",
                "''"
            )

            data.append(headline_clean)

            print(
                f"Scraped: {headline_clean}"
            )

    except Exception as e:

        print(
            f"Error scraping {i}: {e}"
        )


print("scraping done")
print(data)

# ============================================================
# SEVERITY
# ============================================================

for i in data:

    try:

        result = severe(i)

        severity.append(result)

        print(
            f"Severity: {result}"
        )

    except Exception as e:

        print(
            f"Severity error: {e}"
        )

        severity.append("UNKNOWN")


print("severity done")


# ============================================================
# KEYBART SHORT HEADLINES
# ============================================================

for i in data:

    try:

        result = short(i)

        responses.append(result)

        print(
            f"Short: {result}"
        )

    except Exception as e:

        print(
            f"KeyBART error: {e}"
        )

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



