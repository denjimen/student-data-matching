"""
geocode_in_county_centers.py: County Seat Fallback Geocoding (Code 1c)
======================================================================
Purpose:
    Final geocoding pass for records that still have null lat/lon after
    ZIP-based geocoding. Uses the geographic center (centroid) of each
    county as a conservative fallback so remaining records can still be
    evaluated for HPSA/Rural status.

Inputs:
    - Profession CSVs in ./AddressData with columns:
        County, Latitude, Longitude
      (these are post-zip_lookup_generator + update_coords_from_zip_lookup)
    - CountyCenters.csv in ./AddressData with:
        County, Latitude, Longitude
      (county geographic centroids, not necessarily county seats)

Outputs:
    - In-place updates to profession CSVs:
        Remaining null coordinates are filled with county centroid lat/lon.

Pipeline position:
    zip_lookup_generator.py  →  update_coords_from_zip_lookup.py  →  geocode_in_county_centers.py
    (ZIP table)                 (ZIP gap filler)                     (county fallback, near 100% coverage)
"""

import os
import pandas as pd
from tqdm import tqdm

def geocode_county_centers(input_dir: str = "./AddressData",
                           centers_file: str = "CountyCenters.csv") -> None:
    """Batch county-center fallback geocoding for all profession CSVs."""

    base_dir = os.path.abspath(input_dir)
    centers_path = os.path.join(base_dir, centers_file)

    if not os.path.exists(centers_path):
        print(f"County centers file not found: {centers_path}")
        return

    print("Loading county center lookup table...")
    centers_df = pd.read_csv(centers_path)

    # Expect columns: County, Latitude, Longitude
    required_cols = {"County", "Latitude", "Longitude"}
    if not required_cols.issubset(set(centers_df.columns)):
        print(f"CountyCenters.csv must contain columns: {required_cols}")
        return

    centers_df["County"] = centers_df["County"].astype(str).str.strip()
    county_centers = {
        c: (lat, lon)
        for c, lat, lon in zip(
            centers_df["County"],
            centers_df["Latitude"],
            centers_df["Longitude"],
        )
    }

    # All CSVs in input_dir except lookup tables
    all_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
    data_files = [
        f for f in all_files
        if f not in {centers_file, "CountyZipCoordinates.csv"}
    ]

    print(f"Found {len(data_files)} CSV files for county-center fallback")

    for filename in tqdm(data_files, desc="County-center geocoding"):
        path = os.path.join(base_dir, filename)
        df = pd.read_csv(path)

        if "County" not in df.columns or "Latitude" not in df.columns or "Longitude" not in df.columns:
            print(f"  → Skipping {filename} (missing County/Latitude/Longitude columns)")
            continue

        # Rows still missing coordinates after ZIP geocoding
        missing_mask = (
            df["Latitude"].isna() | (df["Latitude"] == "") |
            df["Longitude"].isna() | (df["Longitude"] == "")
        )
        to_fill = df[missing_mask].copy()

        if to_fill.empty:
            print(f"  → {filename}: No remaining coordinate gaps; skipping.")
            continue

        filled = 0
        for idx in to_fill.index:
            county_name = str(to_fill.loc[idx, "County"]).strip()
            center = county_centers.get(county_name)

            if center is not None:
                lat, lon = center
                df.at[idx, "Latitude"] = lat
                df.at[idx, "Longitude"] = lon
                filled += 1

        df.to_csv(path, index=False)
        print(f"  → {filename}: Filled {filled}/{to_fill.shape[0]} remaining gaps using county centers.")

    print("\nCounty-center fallback complete. All files are ready for HPSA geocoding (Codes 3 and 4).")

if __name__ == "__main__":
    geocode_county_centers()