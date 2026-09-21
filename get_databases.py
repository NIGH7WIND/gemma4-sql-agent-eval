# get_databases.py
import os
import shutil
import zipfile
import gdown

os.makedirs("data", exist_ok=True)

# Canonical Yale LILY Spider Archive ID
GDRIVE_FILE_ID = "11icoH_EA-NYb0OrPTdehRWm_d7-DIzWX"
ZIP_PATH = "spider_full.zip"

print("Downloading Spider databases (~100 MB)...")
gdown.download(id=GDRIVE_FILE_ID, output=ZIP_PATH, quiet=False)

print("Extracting...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall("temp_spider")

# Move database directory and tables.json into ./data/
if os.path.exists("temp_spider/spider/database"):
    if os.path.exists("data/database"):
        shutil.rmtree("data/database")
    shutil.move("temp_spider/spider/database", "data/database")

if os.path.exists("temp_spider/spider/tables.json"):
    shutil.move("temp_spider/spider/tables.json", "data/tables.json")

# Clean up temporary archive
shutil.rmtree("temp_spider", ignore_errors=True)
if os.path.exists(ZIP_PATH):
    os.remove(ZIP_PATH)

print("Setup complete. Check 'data/database' and 'data/tables.json'.")