import asyncio
import os
from urllib.parse import urljoin

import torch
import torch.nn.functional as F

from bs4 import BeautifulSoup

from google import genai

from playwright.async_api import async_playwright

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)


# ============================================================
# ARRAYS
# ============================================================

severity = []
data = []
responses = []
links = []
articles = []
summary = []


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# SEVERITY MODEL
# ============================================================

model_path = os.getenv("MODEL_PATH")

if not model_path:
    raise ValueError(
        "MODEL_PATH environment variable is not set."
    )


severity_tokenizer = BertTokenizer.from_pretrained(
    model_path
)

severity_model = BertForSequenceClassification.from_pretrained(
    model_path
)

severity_model.to(device)
severity_model.eval()


# ============================================================
# KEYBART MODEL
# ============================================================

keybart_model_name = "bloomberg/KeyBART"

keybart_tokenizer = BartTokenizer.from_pretrained(
    keybart_model_name
)

keybart_model = BartForConditionalGeneration.from_pretrained(
    keybart_model_name
)

keybart_model.to(device)
keybart_model.eval()


# ============================================================
# SEVERITY FUNCTION
# ============================================================

def severe(data_text):

    tok = severity_tokenizer(
        data_text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    tok = {
        k: v.to(device)
        for k, v in tok.items()
    }

    with torch.no_grad():

        output = severity_model(
            **tok
        )

        probs = F.softmax(
            output.logits,
            dim=1
        )

        prediction = torch.argmax(
            probs,
            dim=1
        ).item()

    label_map = {
        0: "CRITICAL",
        1: "IMPORTANT",
        2: "AVERAGE",
        3: "LOW",
    }

    return label_map.get(
        prediction,
        "UNKNOWN"
    )


# ============================================================
# KEYBART SHORT HEADLINE
# ============================================================

def short(text):

    inputs = keybart_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
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
            no_repeat_ngram_size=2,
        )

    result = keybart_tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    # Only take text before ;
    result = result.split(";")[0].strip()

    return result


# ============================================================
# GEMINI SUMMARY WITH MODEL FALLBACK
# ============================================================

def gemini_summarize(article_text):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not set."
        )

    client = genai.Client(
        api_key=api_key
    )

    # Models will be tried in this exact order
    models = [
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-2.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite",
    ]

    prompt = f"""
Rewrite the given news article in simple, clear,
and easy-to-understand English.

Follow these rules strictly:
Summarize the provided news article into exactly 5 points. Follow these strict guidelines:

### Rules:
1. Tone & Vocabulary: Use simple, clear, and easy-to-understand everyday language.
2. Accuracy: Keep the meaning strictly accurate to the text. Do not invent, assume, or add outside details.
3. Structure: Provide a single sentence per point. Do NOT include section titles or subheaders in the final output—only clean bullet points.

---

### Framework & Point Breakdown:

Point 1: The Core Event & Primary Subject
• What to cover: State the main headline event—what happened, who was involved, and where/when it occurred.
• Purpose: Gives the reader instant context without needing any prior knowledge.

Point 2: The Direct Cause or Key Trigger
• What to cover: Explain why or how this happened—the underlying catalyst, incident, or motive.
• Purpose: Provides essential backstory without clogging the summary with minor fluff.

Point 3: Crucial Facts, Background, or Data
• What to cover: Include key numbers, figures, key profiles, or hard evidence supporting the event.
• Purpose: Adds objective facts and substance to reinforce Point 1.

Point 4: Immediate Impact & Key Stakeholders
• What to cover: Explain who or what was directly affected—aftermath, reactions, emergency actions, or immediate fallout.
• Purpose: Highlights real-world outcomes and human consequences.

Point 5: Future Outlook & Next Steps
• What to cover: Explain what happens next—ongoing investigations, future steps, or upcoming developments.
• Purpose: Leaves the reader with a complete picture of the situation going forward.

Main Point
- Give the main point of the news in around 10 words.
- Keep the meaning exactly the same.
- Do not use complex or difficult words.
- Do not add any information.
- Do not change the meaning.

5 Key Points
- Give exactly 5 key points from the article.
- Each point should be around 6-7 words only.
- Use simple and common words.
- Do not add information that is not in the article.

Output format:

Main Point:
[Simple rewritten main point]

Key Points:
- [4–5 words]
- [4–5 words]
- [4–5 words]
- [4–5 words]
- [4–5 words]

Article content:

{article_text}
"""

    last_error = None

    # ========================================================
    # TRY EACH MODEL
    # ========================================================

    for model_name in models:

        try:

            print(
                f"      Trying Gemini model: {model_name}"
            )

            response = client.models.generate_content(

                model=model_name,

                contents=prompt,
            )

            result = response.text.strip()

            if result:

                print(
                    f"      Gemini success: {model_name}"
                )

                return result

            else:

                print(
                    f"      Empty response from {model_name}"
                )

        except Exception as e:

            last_error = e

            print(
                f"      Gemini error on {model_name}: {e}"
            )

            print(
                f"      Trying next Gemini model..."
            )

    # ========================================================
    # ALL MODELS FAILED
    # ========================================================

    raise RuntimeError(
        "All Gemini models failed. "
        f"Last error: {last_error}"
    )
# ============================================================
# ARTICLE CONTENT SCRAPER
# ============================================================

async def scrape_article_content(page, link):

    """
    Fetch the article page and extract paragraph text.
    """

    try:

        print(
            f"      Fetching article: {link}"
        )

        await page.goto(
            link,
            wait_until="domcontentloaded",
            timeout=20000,
        )

        await page.wait_for_selector(
            "p",
            timeout=10000,
        )

        art_soup = BeautifulSoup(
            await page.content(),
            "html.parser",
        )

        paragraphs = art_soup.find_all(
            "p"
        )

        article_text = "\n".join(
            p.get_text(
                " ",
                strip=True,
            )
            for p in paragraphs
            if p.get_text(
                strip=True
            )
        )

        return article_text

    except Exception as e:

        print(
            f"      Error fetching article body "
            f"from {link}: {e}"
        )

        return ""


# ============================================================
# SITE SCRAPER
# ============================================================

async def scrape_site(browser, url):

    page = await browser.new_page(

        user_agent=(
            "Mozilla/5.0 "
            "(Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "Chrome/120 Safari/537.36"
        ),

        viewport={
            "width": 1280,
            "height": 720,
        },
    )

    headline = None
    news_link = None
    article = None

    try:

        print("\n================================")
        print(f"Scraping: {url}")
        print("================================")

        await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=60000,
        )


        # ====================================================
        # NDTV
        # ====================================================

        if "ndtv.com" in url:

            await page.wait_for_selector(
                "h1, h3",
                timeout=15000,
            )

            soup = BeautifulSoup(
                await page.content(),
                "html.parser",
            )

            for element in soup.find_all(
                ["h1", "h3"]
            ):

                text = element.get_text(
                    " ",
                    strip=True,
                )

                if len(text) > 15:

                    headline = text

                    parent_a = (
                        element.find_parent("a")
                        or element.find("a")
                    )

                    if (
                        parent_a
                        and parent_a.get("href")
                    ):
                        news_link = parent_a.get(
                            "href"
                        )

                    break


        # ====================================================
        # INDIAN EXPRESS
        # ====================================================

        elif "indianexpress.com" in url:

            await page.wait_for_selector(
                "h1",
                timeout=15000,
            )

            soup = BeautifulSoup(
                await page.content(),
                "html.parser",
            )

            element = soup.select_one(
                "h1.topblockNews__featuredTitle"
            )

            if not element:
                element = soup.find("h1")

            if element:

                headline = element.get_text(
                    " ",
                    strip=True,
                )

                parent_a = (
                    element.find_parent("a")
                    or element.find("a")
                )

                if (
                    parent_a
                    and parent_a.get("href")
                ):
                    news_link = parent_a.get(
                        "href"
                    )


        # ====================================================
        # THE HINDU
        # ====================================================

        elif "thehindu.com" in url:

            await page.wait_for_selector(
                "h1",
                timeout=15000,
            )

            soup = BeautifulSoup(
                await page.content(),
                "html.parser",
            )

            element = soup.select_one(
                "h1.title"
            )

            if not element:
                element = soup.find("h1")

            if element:

                headline = element.get_text(
                    " ",
                    strip=True,
                )

                parent_a = (
                    element.find_parent("a")
                    or element.find("a")
                )

                if (
                    parent_a
                    and parent_a.get("href")
                ):
                    news_link = parent_a.get(
                        "href"
                    )


        # ====================================================
        # HINDUSTAN TIMES
        # ====================================================

        elif "hindustantimes.com" in url:

            await page.wait_for_selector(
                "h2",
                timeout=15000,
            )

            soup = BeautifulSoup(
                await page.content(),
                "html.parser",
            )

            element = soup.select_one(
                "h2.hdg3"
            )

            if not element:
                element = soup.find("h2")

            if element:

                headline = element.get_text(
                    " ",
                    strip=True,
                )

                parent_a = (
                    element.find_parent("a")
                    or element.find("a")
                )

                if (
                    parent_a
                    and parent_a.get("href")
                ):
                    news_link = parent_a.get(
                        "href"
                    )


        # ====================================================
        # TIMES OF INDIA
        # ====================================================

        else:

            await page.wait_for_selector(
                "div.Kt6Pm.style_change.T5Q6J",
                timeout=15000,
            )

            soup = BeautifulSoup(
                await page.content(),
                "html.parser",
            )

            element = soup.select_one(
                "div.Kt6Pm.style_change.T5Q6J"
            )

            if element:

                headline = element.get_text(
                    " ",
                    strip=True,
                )

                parent_a = (
                    element.find_parent("a")
                    or element.find("a")
                )

                if (
                    parent_a
                    and parent_a.get("href")
                ):
                    news_link = parent_a.get(
                        "href"
                    )


        # ====================================================
        # MAKE LINK ABSOLUTE
        # ====================================================

        if news_link:

            if news_link.startswith("//"):

                news_link = (
                    "https:" + news_link
                )

            elif news_link.startswith("/"):

                news_link = urljoin(
                    url,
                    news_link
                )


        # ====================================================
        # FETCH ARTICLE
        # ====================================================

        if news_link:

            article = await scrape_article_content(
                page,
                news_link
            )

        else:

            print(
                "      No article link found."
            )


        return (
            headline,
            news_link,
            article,
        )


    except Exception as e:

        print(
            f"Error scraping {url}: {e}"
        )

        return (
            None,
            None,
            None,
        )


    finally:

        await page.close()


# ============================================================
# WEBSITE URLS
# ============================================================

urls = [

    "https://www.ndtv.com/",

    "https://www.thehindu.com/",

    "https://timesofindia.indiatimes.com/",

    "https://www.hindustantimes.com/",

    "https://indianexpress.com/",
]


# ============================================================
# MAIN
# ============================================================

async def main():

    global severity
    global data
    global responses
    global links
    global articles
    global summary


    # ========================================================
    # RESET ARRAYS
    # ========================================================

    severity = []
    data = []
    responses = []
    links = []
    articles = []
    summary = []


    # ========================================================
    # START PLAYWRIGHT
    # ========================================================

    async with async_playwright() as p:

        browser = await p.firefox.launch(
            headless=True
        )

        try:

            # =================================================
            # SCRAPE ALL WEBSITES
            # =================================================

            for url in urls:

                (
                    headline,
                    news_link,
                    article,
                ) = await scrape_site(
                    browser,
                    url,
                )

                if headline:

                    # Keep your previous SQL escaping
                    headline = headline.replace(
                        "'",
                        "''",
                    )

                    data.append(
                        headline
                    )

                    links.append(
                        news_link or "N/A"
                    )

                    articles.append(
                        article or ""
                    )

                    print(
                        "\nHeadline:",
                        headline,
                    )

                    print(
                        "Link:",
                        news_link,
                    )

                    print(
                        "Article length:",
                        len(article or ""),
                    )

                    print(
                        "--------------------------------"
                    )

        finally:

            await browser.close()


    # ========================================================
    # SEVERITY
    # ========================================================

    print("\n==============================")
    print("RUNNING SEVERITY MODEL")
    print("==============================")

    for i in data:

        try:

            result = severe(i)

            severity.append(
                result
            )

            print(
                "Headline:",
                i
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


    # ========================================================
    # SHORT HEADLINE / KEYBART
    # ========================================================

    print("\n==============================")
    print("RUNNING KEYBART")
    print("==============================")

    for i in data:

        try:

            result = short(i)

            responses.append(
                result
            )

            print(
                "Original:",
                i
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


    # ========================================================
    # GEMINI SUMMARY
    # ========================================================

    print("\n==============================")
    print("RUNNING GEMINI")
    print("==============================")

    for i in range(len(data)):

        current_severity = severity[i]
        article_text = articles[i]

        if (
            current_severity
            in [
                "CRITICAL",
                "IMPORTANT",
            ]
            and article_text
        ):

            try:

                result = gemini_summarize(
                    article_text
                )

                summary.append(
                    result
                )

                print(
                    f"[{i + 1}] Gemini summary generated."
                )

            except Exception as e:

                summary.append(
                    f"Gemini Error: {e}"
                )

                print(
                    f"[{i + 1}] Gemini Error:",
                    e,
                )

        else:

            result = (
                f"Skipped "
                f"(Severity: {current_severity})"
            )

            summary.append(
                result
            )

            print(
                f"[{i + 1}] Gemini skipped."
            )


    # ========================================================
    # FINAL CHECK
    # ========================================================

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

    print(
        "Articles:",
        len(articles)
    )

    print(
        "Summary:",
        len(summary)
    )

    print(
        "==============================\n"
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    for i in range(len(data)):

        print(
            f"[{i + 1}]"
        )

        print(
            "Headline:",
            data[i]
        )

        print(
            "Severity:",
            severity[i]
        )

        print(
            "Response:",
            responses[i]
        )

        print(
            "Link:",
            links[i]
        )

        print(
            "Summary:"
        )

        print(
            summary[i]
        )

        print(
            "--------------------------------"
        )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )
