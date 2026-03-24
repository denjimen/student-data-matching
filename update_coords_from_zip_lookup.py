"""
update_coords_from_zip_lookup.py: ZIP Coordinate Gap Filler (Code 0.5)
=====================================================================
**Purpose**: Fills missing lat/lon in fuzzy-matched provider data using ZIP lookup table
**Input**: Post-fuzzy CSVs (*.csv) with missing coordinates + CountyZipCoordinates.csv
**Output**: In-place coordinate updates for HPSA geocoding (Codes 3+4)
**Method**: Fast County/ZIP table lookup (10x faster than API calls)

**Pipeline Position**: Fuzzy Providers → ZIP Fill → HPSA Analysis
**Note**: Preserves ID columns end-to-end for production integration
"""

import pandas as pd
import os
from tqdm import tqdm

def fill_coords_from_zip_lookup(input_dir="./AddressData", lookup_file="CountyZipCoordinates.csv"):
    """Batch ZIP coordinate gap filling across all profession CSVs"""
    
    base_dir = os.path.abspath(input_dir)
    lookup_path = os.path.join(base_dir, lookup_file)
    
    # Load ZIP lookup table (from zip_lookup_generator.py)
    print("Loading ZIP coordinate lookup table...")
    zip_lookup = pd.read_csv(lookup_path)
    zip_lookup['ZIP Code'] = zip_lookup['ZIP Code'].astype(str)
    
    # Find all CSV files (post-fuzzy matched provider data)
    all_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    data_files = [f for f in all_files if 'SourceFile_' not in f and 'CountyZipCoordinates' not in f]
    
    print(f"Found {len(data_files)} profession datasets for coordinate gap filling")
    
    for filename in tqdm(data_files, desc="Filling ZIP coordinates"):
        filepath = os.path.join(base_dir, filename)
        
        # Load provider data
        df = pd.read_csv(filepath)
        
        # Check for coordinate columns
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            print(f"  → Skipping {filename} (no lat/lon columns)")
            continue
            
        # Find rows needing ZIP lookup (missing coordinates)
        missing_coords = df['Latitude'].isna() | (df['Latitude'] == '') | df['Longitude'].isna() | (df['Longitude'] == '')
        rows_to_fill = df[missing_coords].copy()
        
        if 'County' in rows_to_fill.columns and 'ZIP Code' in rows_to_fill.columns:
            # ZIP lookup for missing coordinates
            filled_count = 0
            for idx in rows_to_fill.index:
                county = str(rows_to_fill.loc[idx, 'County'])
                zip_code = str(rows_to_fill.loc[idx, 'ZIP Code'])
                
                # County + ZIP lookup (most precise)
                match = zip_lookup[
                    (zip_lookup['County'] == county) & 
                    (zip_lookup['ZIP Code'] == zip_code)
                ]
                
                if not match.empty:
                    lat = match.iloc[0]['Latitude']
                    lon = match.iloc[0]['Longitude']
                    df.loc[idx, 'Latitude'] = lat
                    df.loc[idx, 'Longitude'] = lon
                    filled_count += 1
                elif not pd.isna(zip_code) and zip_code != 'nan':
                    # ZIP-only fallback
                    zip_match = zip_lookup[zip_lookup['ZIP Code'] == zip_code]
                    if not zip_match.empty:
                        lat = zip_match.iloc[0]['Latitude']
                        lon = zip_match.iloc[0]['Longitude']
                        df.loc[idx, 'Latitude'] = lat
                        df.loc[idx, 'Longitude'] = lon
                        filled_count += 1
            
            print(f"  → {filename}: {filled_count}/{len(rows_to_fill)} ZIP gaps filled")
        
        # Save updated file
        df.to_csv(filepath, index=False)
        print(f"  → {filename}: Saved with ZIP-filled coordinates")
    
    print("\n✅ ZIP gap filling complete. Files ready for HPSA geocoding (Codes 3+4)")

if __name__ == "__main__":
    fill_coords_from_zip_lookup()