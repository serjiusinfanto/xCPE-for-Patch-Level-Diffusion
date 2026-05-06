"""
Download the Weather dataset from HuggingFace (thuml/Time-Series-Library, CC BY 4.0).

Usage:
    python scripts/download_weather.py

Downloads to:
    data/raw/data/raw/weather/weather.csv
(Matches the nested path structure created by earlier ETT downloads.)
"""

from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil

LOCAL_DIR = Path("data/raw")
REMOTE_FILE = "weather/weather.csv"
EXPECTED_PATH = LOCAL_DIR / REMOTE_FILE  # data/raw/weather/weather.csv

print(f"Downloading {REMOTE_FILE} from thuml/Time-Series-Library ...")
downloaded = hf_hub_download(
    repo_id="thuml/Time-Series-Library",
    filename=REMOTE_FILE,
    repo_type="dataset",
    local_dir=str(LOCAL_DIR),
)
print(f"Saved to: {downloaded}")
print("Done.")
