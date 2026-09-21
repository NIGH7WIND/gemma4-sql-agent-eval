# extract_spider.py
import os
import re
import zipfile

zip_path = r"C:\Users\Sahal\Downloads\AutoDownload\spider_data.zip"
dest_dir = r"data"

os.makedirs(dest_dir, exist_ok=True)

print(f"Extracting relevant files from {zip_path}...")

with zipfile.ZipFile(zip_path, "r") as archive:
    for member in archive.infolist():
        # 1. Skip useless macOS metadata files
        if "__MACOSX" in member.filename:
            continue

        # 2. Extract only sqlite databases and tables.json/dev.json
        # Normalize slashes
        clean_name = member.filename.replace("\\", "/")

        # Match tables.json or files under database/
        if "tables.json" in clean_name or "/database/" in clean_name:
            # Re-root the path so it goes into data/database/... or data/tables.json
            if "tables.json" in clean_name:
                target_path = os.path.join(dest_dir, "tables.json")
            else:
                rel_path = clean_name.split("/database/", 1)[-1]
                # Replace invalid Windows filename characters (: * ? " < > |) just in case
                sanitized_rel_path = re.sub(r'[:*?"<>|]', "_", rel_path)
                target_path = os.path.join(dest_dir, "database", sanitized_rel_path)

            if member.is_dir():
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with archive.open(member) as source, open(target_path, "wb") as target:
                    target.write(source.read())

print("Extraction complete!")