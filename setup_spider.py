# setup_spider.py
import json
import os
import urllib.request
import zipfile
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

# 1. Download dev questions/queries via Hugging Face API
print("1/3 Downloading dev dataset split...")
ds = load_dataset("xlangai/spider", split="validation")
dev_records = []
for row in ds:
    dev_records.append({
        "db_id": row["db_id"],
        "question": row["question"],
        "query": row["query"]
    })

with open("data/dev.json", "w", encoding="utf-8") as f:
    json.dump(dev_records, f, indent=2, ensure_ascii=False)
print("Saved data/dev.json successfully.")

# 2. Download official SQLite databases and table schemas (~100 MB)
# Hosted on the canonical Yale LILY lab mirror
URL = "https://raw.githubusercontent.com/Yale-LILY/Spider/master"
ZIP_URL = "https://spider-eval.s3.amazonaws.com/spider.zip"  # Reliable S3 mirror

print("2/3 Downloading spider database archive...")
try:
    urllib.request.urlretrieve(ZIP_URL, "spider.zip")
    print("3/3 Extracting archive...")
    with zipfile.ZipFile("spider.zip", "r") as z:
        z.extractall("temp_extract")
    
    # Move database directory and cleanup
    if os.path.exists("temp_extract/spider/database"):
        os.replace("temp_extract/spider/database", "data/database")
        if os.path.exists("temp_extract/spider/tables.json"):
            os.replace("temp_extract/spider/tables.json", "data/tables.json")
    import shutil
    shutil.rmtree("temp_extract", ignore_errors=True)
    os.remove("spider.zip")
    print("Done! Evaluation assets are ready in data/")
except Exception as e:
    print(f"Direct S3 download failed ({e}), falling back to HF zip...")