from pathlib import Path
from datetime import datetime, UTC
import os
import json
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "medical_docs"

print("Saving dataset to:", BASE_DIR)

SOURCES = {
    "diabetes": [
        "https://www.cdc.gov/diabetes/about/index.html",
        "https://www.cdc.gov/diabetes/signs-symptoms/index.html",
        "https://www.who.int/news-room/fact-sheets/detail/diabetes",
        "https://medlineplus.gov/diabetes.html",
        "https://www.cdc.gov/diabetes/diabetes-testing/prediabetes-a1c-test.html",
        "https://medlineplus.gov/lab-tests/hemoglobin-a1c-hba1c-test/",
        "https://www.niddk.nih.gov/health-information/diagnostic-tests/a1c-test",
        "https://diabetes.org/about-diabetes/a1c",
        "https://my.clevelandclinic.org/health/diagnostics/9731-a1c",
    ],
    "high_blood_pressure": [
        "https://www.cdc.gov/high-blood-pressure/about/index.html",
        "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "https://medlineplus.gov/highbloodpressure.html",
    ],
    "cholesterol": [
        "https://www.cdc.gov/cholesterol/about/index.html",
        "https://medlineplus.gov/cholesterol.html",
    ],
    "asthma": [
        "https://www.cdc.gov/asthma/about/index.html",
        "https://www.who.int/news-room/fact-sheets/detail/asthma",
        "https://medlineplus.gov/asthma.html",
    ],
    "flu": [
        "https://www.cdc.gov/flu/about/index.html", 
        "https://www.who.int/news-room/fact-sheets/detail/influenza-%28seasonal%29",
        "https://medlineplus.gov/flu.html",
    ],
    "covid19": [
        "https://www.cdc.gov/covid/about/index.html",
        "https://www.who.int/news-room/fact-sheets/detail/coronavirus-disease-(covid-19)",
        "https://medlineplus.gov/covid19coronavirusdisease2019.html",
    ],
    "dengue": [
        "https://www.cdc.gov/dengue/about/index.html",
        "https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
        "https://medlineplus.gov/dengue.html",
    ],
    "migraine": [
    "https://medlineplus.gov/migraine.html",
    "https://www.nccih.nih.gov/health/headaches-what-you-need-to-know",
],
    "food_poisoning": [
        "https://www.cdc.gov/food-safety/about/index.html",
        "https://medlineplus.gov/foodborneillness.html",
    ],
    "eczema": [
    "https://medlineplus.gov/eczema.html",
    "https://www.niams.nih.gov/health-topics/atopic-dermatitis",
    "https://www.niaid.nih.gov/diseases-conditions/eczema-atopic-dermatitis",
    ],
}


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_url(url):
    headers = {
        "User-Agent": "MedicalRAGDatasetBuilder/1.0"
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else "Untitled"

    main = soup.find("main")
    content_area = main if main else soup.body

    text = content_area.get_text(separator=" ", strip=True)
    return title, clean_text(text)


def safe_filename(url):
    parsed = urlparse(url)
    name = parsed.netloc.replace("www.", "") + parsed.path
    name = name.strip("/").replace("/", "_")
    name = re.sub(r"[^a-zA-Z0-9_\\-]", "_", name)
    return name[:120] + ".txt"


def build_dataset():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    metadata = []

    for topic, urls in SOURCES.items():
        topic_dir = BASE_DIR / topic
        topic_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing topic: {topic}")

        for url in urls:
            try:
                title, text = extract_text_from_url(url)

                if len(text) < 500:
                    print(f"Skipped short content: {url}")
                    continue

                filename = safe_filename(url)
                file_path = topic_dir / filename

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\n")
                    f.write(f"Source URL: {url}\n")
                    f.write(f"Topic: {topic}\n")
                    f.write("\n--- CONTENT ---\n\n")
                    f.write(text)

                metadata.append({
                    "topic": topic,
                    "title": title,
                    "source_url": url,
                    "file_path": str(file_path),
                    "source_domain": urlparse(url).netloc,
                    "downloaded_at": datetime.now(UTC).isoformat()
                })

                print(f"Saved: {file_path}")

            except Exception as e:
                print(f"Failed: {url}")
                print(f"Reason: {e}")

    metadata_path = BASE_DIR / "metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nDataset build completed.")
    print(f"Metadata saved at: {metadata_path}")


if __name__ == "__main__":
    build_dataset()