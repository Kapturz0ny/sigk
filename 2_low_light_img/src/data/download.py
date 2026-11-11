import csv
import os

import requests


def download_raise_nef(csv_path, output_dir, limit=None):
    """
    Downloads NEF files from the RAISE dataset based on a CSV file.
    The 'limit' parameter specifies the maximum number of files to process.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(csv_reader):
            if limit is not None and i >= limit:
                break

            nef_url = row["NEF"]
            filename = row["File"]
            if not filename.upper().endswith(".NEF"):
                filename += ".NEF"

            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                print(f"Skipping {filename} - already exists.")
                continue
            print(f"Downloading {filename}...")
            response = requests.get(nef_url, stream=True)

            with open(filepath, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
