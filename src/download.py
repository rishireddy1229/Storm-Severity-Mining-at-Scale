"""
Download NOAA Storm Events detail CSVs (1996-2024).

Scrapes the NOAA NCEI bulk-download directory listing, identifies the most
recent detail file for each year, downloads and decompresses them.
"""

import gzip
import re
import shutil
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import DATA_RAW, NOAA_BASE_URL, YEAR_RANGE
from src.utils import logger


def list_remote_files():
    """Scrape the NOAA directory listing and return all filenames."""
    logger.info(f"Fetching directory listing from {NOAA_BASE_URL}")
    resp = requests.get(NOAA_BASE_URL, timeout=60)
    resp.raise_for_status()
    # Extract filenames from HTML href attributes
    filenames = re.findall(r'href="(StormEvents_details[^"]+\.csv\.gz)"', resp.text)
    logger.info(f"Found {len(filenames)} detail files on server")
    return filenames


def pick_latest_per_year(filenames):
    """
    For each year in YEAR_RANGE, pick the file with the latest creation date.
    Filename pattern: StormEvents_details-ftp_v1.0_d{YEAR}_c{CREATED}.csv.gz
    """
    year_files = {}
    pattern = re.compile(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d{8})\.csv\.gz")
    for fn in filenames:
        m = pattern.match(fn)
        if m:
            year = int(m.group(1))
            created = m.group(2)
            if year in YEAR_RANGE:
                if year not in year_files or created > year_files[year][1]:
                    year_files[year] = (fn, created)
    selected = {y: info[0] for y, info in sorted(year_files.items())}
    logger.info(f"Selected {len(selected)} files for years {min(selected)}-{max(selected)}")
    return selected


def download_file(filename, dest_dir):
    """Download a single .csv.gz file and decompress it."""
    url = NOAA_BASE_URL + filename
    gz_path = dest_dir / filename
    csv_path = dest_dir / filename.replace(".gz", "")

    if csv_path.exists():
        logger.info(f"Already exists: {csv_path.name}")
        return csv_path

    # Download
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with open(gz_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc=filename, leave=False) as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    # Decompress
    with gzip.open(gz_path, "rb") as f_in:
        with open(csv_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()
    return csv_path


def download_all():
    """Download all detail CSVs for YEAR_RANGE."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)

    filenames = list_remote_files()
    year_files = pick_latest_per_year(filenames)

    missing_years = set(YEAR_RANGE) - set(year_files.keys())
    if missing_years:
        logger.warning(f"No files found for years: {sorted(missing_years)}")

    downloaded = []
    for year in sorted(year_files):
        fn = year_files[year]
        path = download_file(fn, DATA_RAW)
        downloaded.append(path)
        logger.info(f"[{year}] {path.name}")

    logger.info(f"Download complete: {len(downloaded)} files in {DATA_RAW}")
    return downloaded


if __name__ == "__main__":
    download_all()
