"""
Optional helper script for full-address geocoding using the Google Maps Geocoding API.

WARNING: This script requires a billable Google Maps Platform project.
Running it at scale may incur charges; review current pricing:
https://developers.google.com/maps/documentation/geocoding/usage-and-billing
"""

import os
import time
import logging
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from a .env file in the project root
load_dotenv("GeocodeAPI.env")

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not GOOGLE_MAPS_API_KEY:
    raise ValueError(
        "Google Maps API key not found. "
        "Set GOOGLE_MAPS_API_KEY in GeocodeAPI.env."
    )

# Simple rate limiting so you do not accidentally hammer the API
API_CALLS = 0
API_CALL_LIMIT = 50      # calls per interval
API_CALL_INTERVAL = 1.0  # seconds to sleep when limit reached


def _throttle():
    """Basic client‑side rate limiting."""
    global API_CALLS
    if API_CALLS >= API_CALL_LIMIT:
        time.sleep(API_CALL_INTERVAL)
        API_CALLS = 0
    API_CALLS += 1


def _request_geocode(query: str):
    """Call Google Geocoding API and return parsed JSON or None."""
    _throttle()
    url = (
        "https://maps.googleapis.com/maps/api/geocode/json"
        f"?address={query}&key={GOOGLE_MAPS_API_KEY}"
    )
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        logging.error(f"Error calling Google Geocoding API for '{query}': {exc}")
        return None

    if data.get("status") != "OK":
        logging.warning(
            "Google Geocoding API status %s for query '%s'",
            data.get("status"),
            query,
        )
        return None

    return data


def get_lat_lon_from_address(address: str, state: str):
    """Geocode a full street address."""
    if pd.isna(address) or not str(address).strip():
        return None, None

    query = f"{str(address).strip()}, {state}"
    data = _request_geocode(query)
    if not data:
        return None, None

    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def get_lat_lon_from_county(county: str, state: str):
    """Geocode a county when no street address is available."""
    if pd.isna(county) or not str(county).strip():
        return None, None

    query = f"{str(county).strip()} County, {state}"
    data = _request_geocode(query)
    if not data:
        return None, None

    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def get_lat_lon_from_zip(zip_code: str, state: str):
    """Geocode a ZIP code (used for STZP files or as a last resort)."""
    if pd.isna(zip_code) or not str(zip_code).strip():
        return None, None

    query = f"{str(zip_code).strip()}, {state}"
    data = _request_geocode(query)
    if not data:
        return None, None

    loc = data["results"][0]["geometry"]["location"]
    return loc["lat"], loc["lng"]


def process_csv(input_path: str, output_path: str):
    """
    Read a CSV with practice location fields, append Latitude/Longitude,
    and write an Excel file for downstream use.
    """
    try:
        df_full = pd.read_csv(input_path, encoding="utf-8")
    except UnicodeDecodeError:
        df_full = pd.read_csv(input_path, encoding="latin-1")

    required_columns = [
        "PrimaryPracticeAddress",
        "PrimaryPracticeCounty",
        "PrimaryPracticeState",
        "PrimaryPracticeZIP",
    ]
    missing = [c for c in required_columns if c not in df_full.columns]
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")

    df_geo = df_full[required_columns].copy()

    results = []
    for _, row in tqdm(df_geo.iterrows(), total=len(df_geo), desc="Geocoding"):
        if row["PrimaryPracticeAddress"]:
            lat, lon = get_lat_lon_from_address(
                row["PrimaryPracticeAddress"],
                row["PrimaryPracticeState"],
            )
        elif row["PrimaryPracticeCounty"]:
            lat, lon = get_lat_lon_from_county(
                row["PrimaryPracticeCounty"],
                row["PrimaryPracticeState"],
            )
        elif row["PrimaryPracticeZIP"]:
            lat, lon = get_lat_lon_from_zip(
                row["PrimaryPracticeZIP"],
                row["PrimaryPracticeState"],
            )
        else:
            lat, lon = None, None

        results.append((lat, lon))

    df_geo[["Latitude", "Longitude"]] = pd.DataFrame(results, index=df_geo.index)

    # Merge geocoding results back into the full dataset
    df_full[["Latitude", "Longitude"]] = df_geo[["Latitude", "Longitude"]]

    # Write Excel output
    try:
        with pd.ExcelWriter(output_path) as writer:
            df_full.to_excel(writer, index=False)
        print(f"Output written to {output_path}")
    except Exception as exc:
        logging.error(f"Error writing output file '{output_path}': {exc}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    # Folder containing MatchedCombinedData_AddressLookup_* CSVs
    INPUT_FOLDER = "data/address_lookup"          # <- adjust for your project
    OUTPUT_FOLDER = "data/address_processed"      # <- adjust for your project

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    input_files = [
        # Add the profession-specific files you want to geocode, for example:
        # "addressLookup_APRN_001.csv",
        # "address_AddressLookup_RN_001.csv",
    ]

    for name in input_files:
        input_path = os.path.join(INPUT_FOLDER, name)
        base = os.path.splitext(name)[0]
        output_name = base.replace("AddressLookup", "AddressProcessed") + ".xlsx"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        try:
            process_csv(input_path, output_path)
        except Exception as exc:
            logging.error("Error processing %s: %s", name, exc)