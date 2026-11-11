import csv
import os

import requests
from tqdm import tqdm


def download_raise_nef(csv_path, output_dir, limit=None):
    """
    Downloads NEF files from the RAISE dataset based on a CSV file.
    The 'limit' parameter specifies the maximum number of files to process.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        if limit is not None:
            rows = rows[:limit]

        for row in tqdm(rows, desc="Downloading NEF files"):
            nef_url = row["NEF"]
            filename = row["File"]
            if not filename.upper().endswith(".NEF"):
                filename += ".NEF"

            filepath = os.path.join(output_dir, filename)

            if os.path.exists(filepath):
                continue

            response = requests.get(nef_url, stream=True)

            with open(filepath, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)


if __name__ == "__main__":
    CSV_PATH = "data/raw/RAISE_258.csv"
    OUTPUT_DIR = "data/raw/nef"
    LIMIT = 100

    download_raise_nef(csv_path=CSV_PATH, output_dir=OUTPUT_DIR, limit=LIMIT)
