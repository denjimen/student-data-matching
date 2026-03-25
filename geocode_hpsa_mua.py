import os
import math
import logging
import time

import requests
import pandas as pd
from tqdm import tqdm  # pip install tqdm

# Configure logging for production use
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

tqdm.pandas()

# Indiana bounding box endpoints (west, south, east, north) [web:653][web:656]
IN_WEST  = -88.09776
IN_SOUTH = 37.771742
IN_EAST  = -84.784579
IN_NORTH = 41.760592


def safe_bool_or_nan(lat, lon, raw_bool):
    """
    For a single point:
    - If lat/lon invalid or out of Indiana → return NaN
    - Else return the original boolean (True/False)
    """
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (TypeError, ValueError):
        return math.nan

    if not (IN_SOUTH <= lat_f <= IN_NORTH and IN_WEST <= lon_f <= IN_EAST):
        return math.nan

    # raw_bool is whatever your HRSA query produced: True/False
    return raw_bool


def query_mua(latitudes, longitudes):
    """Query HRSA Medically Underserved Areas (MUA) API for spatial intersection."""
    base_url = "https://gisportal.hrsa.gov/server/rest/services/Shortage/MedicallyUnderservedAreas_FS/MapServer/0/query"
    results = []

    for lat, lon in tqdm(list(zip(latitudes, longitudes)), desc="Processing MUAs"):
        params = {
            "f": "json",
            "outFields": "MUA_DESIGNATION_TYP_CD,MUA_DESIGNATION_TYP_DESC,MUA_SERVICE_AREA_NM,MUA_DESIGNATION_DT_TXT,MUA_UPDATE_DT_TXT,CMN_STATE_ABBR",
            "geometryType": "esriGeometryPoint",
            "geometry": f"{lon},{lat}",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelWithin",
            "returnGeometry": "false",
        }

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                results.append(bool(data.get("features")))  # True if point is within MUA
            else:
                results.append(False)  # API error → treat as not MUA (will be NaN if coords bad)
        except Exception as e:
            logging.warning(f"MUA request failed for ({lat}, {lon}): {e}")
            results.append(False)

        time.sleep(0.1)  # Rate limiting - HRSA APIs throttle heavy usage

    return results


def query_mup(latitudes, longitudes):
    """Query HRSA Medically Underserved Populations (MUP) API for spatial intersection."""
    base_url = "https://gisportal.hrsa.gov/server/rest/services/Shortage/MedicallyUnderservedPopulations_FS/MapServer/0/query"
    results = []

    for lat, lon in tqdm(list(zip(latitudes, longitudes)), desc="Processing MUPs"):
        params = {
            "f": "json",
            "outFields": "MUP_DESIGNATION_TYP_CD,MUP_DESIGNATION_TYP_DESC,MUP_SERVICE_AREA_NM,MUP_DESIGNATION_DT_TXT,MUP_UPDATE_DT_TXT,CMN_STATE_ABBR",
            "geometryType": "esriGeometryPoint",
            "geometry": f"{lon},{lat}",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelWithin",
            "returnGeometry": "false",
        }

        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                results.append(bool(data.get("features")))  # True if point is within MUP
            else:
                results.append(False)  # API error
        except Exception as e:
            logging.warning(f"MUP request failed for ({lat}, {lon}): {e}")
            results.append(False)

        time.sleep(0.1)  # Rate limiting

    return results


def update_hpsa_with_mua_mup(input_file):
    """
    Main ETL function:
    - Reads HPSA CSV with Lat/Lon columns
    - Queries HRSA MUA/MUP APIs for each geocoded location
    - Adds MUA/MUP columns with Yes/No, BUT
      invalid / out-of-IN coords become NaN instead of No
    - Overwrites original file in-place
    """
    df = pd.read_csv(input_file)

    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        raise ValueError("Input CSV must contain 'Latitude' and 'Longitude' columns")

    latitudes = df["Latitude"].tolist()
    longitudes = df["Longitude"].tolist()

    # Query MUAs and MUPs (booleans) with progress bars
    mua_raw = query_mua(latitudes, longitudes)
    mup_raw = query_mup(latitudes, longitudes)

    # Apply bounding box + NaN logic per row
    mua_validated = [
        safe_bool_or_nan(lat, lon, result)
        for lat, lon, result in zip(latitudes, longitudes, mua_raw)
    ]
    mup_validated = [
        safe_bool_or_nan(lat, lon, result)
        for lat, lon, result in zip(latitudes, longitudes, mup_raw)
    ]

    # Turn booleans into Yes/No, keep NaN as NaN
    df["MUA"] = [
        ("Yes" if val is True else "No") if isinstance(val, bool) else math.nan
        for val in mua_validated
    ]
    df["MUP"] = [
        ("Yes" if val is True else "No") if isinstance(val, bool) else math.nan
        for val in mup_validated
    ]

    # Overwrite original file with enriched data
    df.to_csv(input_file, index=False)
    logging.info(
        f"Enriched {input_file}: {len(df)} rows, "
        f"{sum(v is True for v in mua_validated)} MUA sites, "
        f"{sum(v is True for v in mup_validated)} MUP sites"
    )


def main():
    """Process multiple CSV files (one per medical profession) using .env config."""
    input_folder = "/Users/Denis/Desktop/GitHub"
    # For now, just use the fake test file
    input_files = ["indiana_test_geocode.csv"]
    # Or, if you want to go back to env-based:
    # input_folder = os.getenv('INPUT_FOLDER', 'InputFolder')
    # input_files = os.getenv('INPUT_FILES', 'InputFile.csv').split(',')

    for input_file in input_files:
        input_path = os.path.join(input_folder, input_file.strip())
        if not os.path.exists(input_path):
            logging.warning(f"File not found, skipping: {input_path}")
            continue

        try:
            file_size_mb = os.path.getsize(input_path) / 1e6
            print(f"Processing {input_file} ({file_size_mb:.1f} MB)")
            update_hpsa_with_mua_mup(input_path)
            print(f"✓ {input_file} enriched in-place")
        except Exception as e:
            logging.error(f"✗ Failed {input_file}: {e}")


if __name__ == "__main__":
    main()