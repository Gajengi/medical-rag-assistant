from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse
import json
import re

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError
from bs4 import BeautifulSoup


# =========================================================
# Project and AWS configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Existing local dataset is used only when a website blocks scraping.
LOCAL_FALLBACK_DIR = PROJECT_ROOT / "data" / "medical_docs"

AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "medical-rag-manideep-dev"

# S3 object prefixes
DATASET_PREFIX = "datasets"
METADATA_KEY = "metadata/metadata.json"


# Boto3 automatically uses the credentials configured through:
# aws configure
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
)


# =========================================================
# Medical source URLs
# =========================================================

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


# =========================================================
# Text-processing functions
# =========================================================

def clean_text(text: str) -> str:
    """
    Remove repeated spaces, line breaks, and tabs.
    """
    return re.sub(r"\s+", " ", text).strip()


def safe_filename(url: str) -> str:
    """
    Convert a URL into a safe .txt filename.

    Example:
    https://www.cdc.gov/flu/about/index.html

    becomes:
    cdc_gov_flu_about_index_html.txt
    """

    parsed = urlparse(url)

    name = parsed.netloc.replace("www.", "") + parsed.path
    name = name.strip("/")
    name = name.replace("/", "_")

    name = re.sub(
        r"[^a-zA-Z0-9_\-]",
        "_",
        name,
    )

    if not name:
        name = "document"

    return name[:120] + ".txt"


def extract_text_from_url(url: str) -> tuple[str, str]:
    """
    Download a webpage and extract its title and readable text.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,image/avif,"
            "image/webp,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "lxml",
    )

    # Remove page sections that usually contain navigation,
    # scripts, styling, advertisements, and unrelated content.
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "noscript",
        ]
    ):
        tag.decompose()

    title = (
        soup.title.get_text(strip=True)
        if soup.title
        else "Untitled"
    )

    main = soup.find("main")
    content_area = main if main else soup.body

    if content_area is None:
        raise ValueError(
            "No readable page content was found."
        )

    text = content_area.get_text(
        separator=" ",
        strip=True,
    )

    return title, clean_text(text)


def read_local_fallback(
    topic: str,
    url: str,
) -> tuple[str, str]:
    """
    Read a previously saved local document when the
    source website blocks an automated request.

    The local files are expected under:

    data/medical_docs/<topic>/<filename>.txt
    """

    filename = safe_filename(url)

    local_path = (
        LOCAL_FALLBACK_DIR
        / topic
        / filename
    )

    if not local_path.exists():
        raise FileNotFoundError(
            f"No local fallback file found: {local_path}"
        )

    full_content = local_path.read_text(
        encoding="utf-8",
    )

    title = "Untitled"
    content = full_content

    for line in full_content.splitlines():
        if line.startswith("Title:"):
            title = line.replace(
                "Title:",
                "",
                1,
            ).strip()
            break

    content_marker = "--- CONTENT ---"

    if content_marker in full_content:
        content = full_content.split(
            content_marker,
            1,
        )[1].strip()

    return title, clean_text(content)


def create_document_content(
    title: str,
    source_url: str,
    topic: str,
    text: str,
) -> str:
    """
    Create the final text document uploaded to S3.
    """

    return (
        f"Title: {title}\n"
        f"Source URL: {source_url}\n"
        f"Topic: {topic}\n"
        "\n--- CONTENT ---\n\n"
        f"{text}"
    )


# =========================================================
# S3 functions
# =========================================================

def verify_bucket_access() -> None:
    """
    Verify that the configured AWS credentials can access
    the S3 bucket before starting the full dataset build.
    """

    try:
        s3_client.head_bucket(
            Bucket=S3_BUCKET_NAME
        )

        print(
            "Connected to S3 bucket:",
            S3_BUCKET_NAME,
        )

    except ClientError as error:
        raise RuntimeError(
            "Unable to access the S3 bucket. "
            "Verify the bucket name, region, "
            "AWS credentials, and IAM permissions."
        ) from error


def upload_text_to_s3(
    key: str,
    content: str,
) -> None:
    """
    Upload text content directly from memory to S3.
    """

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=content.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
        ServerSideEncryption="AES256",
    )


def upload_json_to_s3(
    key: str,
    data: list[dict],
) -> None:
    """
    Upload metadata as a JSON object to S3.
    """

    json_content = json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
    )

    s3_client.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=json_content.encode("utf-8"),
        ContentType="application/json",
        ServerSideEncryption="AES256",
    )


# =========================================================
# Dataset-builder process
# =========================================================

def build_dataset() -> None:
    """
    Download medical content and upload it directly to S3.
    """

    verify_bucket_access()

    metadata: list[dict] = []

    successful_uploads = 0
    failed_documents = 0
    fallback_documents = 0
    skipped_documents = 0

    print(
        "Saving dataset to:",
        f"s3://{S3_BUCKET_NAME}/{DATASET_PREFIX}/",
    )

    for topic, urls in SOURCES.items():
        print(f"\nProcessing topic: {topic}")

        for url in urls:
            used_fallback = False

            try:
                try:
                    # First attempt: download the live webpage.
                    title, text = extract_text_from_url(url)

                except requests.HTTPError as error:
                    status_code = (
                        error.response.status_code
                        if error.response is not None
                        else None
                    )

                    # Use a previously downloaded local file
                    # when the website returns 403 Forbidden.
                    if status_code == 403:
                        print(
                            "Website blocked automated request."
                        )
                        print(
                            f"Using local fallback: {url}"
                        )

                        title, text = read_local_fallback(
                            topic=topic,
                            url=url,
                        )

                        used_fallback = True
                        fallback_documents += 1

                    else:
                        raise

                if len(text) < 500:
                    print(
                        f"Skipped short content: {url}"
                    )

                    skipped_documents += 1
                    continue

                filename = safe_filename(url)

                s3_key = (
                    f"{DATASET_PREFIX}/"
                    f"{topic}/"
                    f"{filename}"
                )

                document_content = create_document_content(
                    title=title,
                    source_url=url,
                    topic=topic,
                    text=text,
                )

                upload_text_to_s3(
                    key=s3_key,
                    content=document_content,
                )

                metadata.append(
                    {
                        "topic": topic,
                        "title": title,
                        "source_url": url,
                        "source_domain": urlparse(
                            url
                        ).netloc,
                        "s3_bucket": S3_BUCKET_NAME,
                        "s3_key": s3_key,
                        "download_method": (
                            "local_fallback"
                            if used_fallback
                            else "live_web_download"
                        ),
                        "downloaded_at": datetime.now(
                            UTC
                        ).isoformat(),
                    }
                )

                successful_uploads += 1

                print(
                    "Uploaded:",
                    f"s3://{S3_BUCKET_NAME}/{s3_key}",
                )

            except requests.RequestException as error:
                failed_documents += 1

                print(
                    f"Failed to download: {url}"
                )
                print(f"Reason: {error}")

            except FileNotFoundError as error:
                failed_documents += 1

                print(
                    f"No fallback available for: {url}"
                )
                print(f"Reason: {error}")

            except (BotoCoreError, ClientError) as error:
                failed_documents += 1

                print(
                    f"Failed to upload: {url}"
                )
                print(f"Reason: {error}")

            except Exception as error:
                failed_documents += 1

                print(f"Failed: {url}")
                print(f"Reason: {error}")

    try:
        upload_json_to_s3(
            key=METADATA_KEY,
            data=metadata,
        )

        print("\nDataset build completed.")

        print(
            "Metadata uploaded to:",
            f"s3://{S3_BUCKET_NAME}/{METADATA_KEY}",
        )

    except (BotoCoreError, ClientError) as error:
        print(
            "\nFailed to upload metadata."
        )
        print(f"Reason: {error}")

    print("\nDataset summary")
    print("-------------------------------")
    print(
        f"Successful uploads: {successful_uploads}"
    )
    print(
        f"Local fallback documents: "
        f"{fallback_documents}"
    )
    print(
        f"Skipped documents: {skipped_documents}"
    )
    print(
        f"Failed documents: {failed_documents}"
    )


if __name__ == "__main__":
    build_dataset()