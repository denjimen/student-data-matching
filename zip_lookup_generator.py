"""
zip_lookup_generator.py: Statewide ZIP Coordinate Reference Table
============================================================================
**Purpose**: Creates cached County/ZIP → Lat/Lon lookup table (run ONCE)
**Input**: CountiesWithZipCodes.csv (ALL ZIP codes in target state)
**Output**: CountyZipCoordinates.csv (reference table for HPSA geocoding)
**API**: Nominatim OSM (FREE, no API key)
**Note**: Input contains complete ZIP coverage for single U.S. state

**Production pipeline accelerator** - enhances geocode_hpsa_mua.py + geocode_hpsa_pc_mh_dt.py
"""

import requests
import pandas as pd
import time
import os
from tqdm import tqdm

def get_coordinates(zip_codes):
    """Compliant Nominatim geocoding (1 req/sec max)"""
    coordinates = []
    
    for zip_code in tqdm(zip_codes, desc="Geocoding ZIPs"):
        if pd.isna(zip_code):
            coordinates.append((None, None))
            continue
        
        zip_code_str = str(int(zip_code))
        url = f'https://nominatim.openstreetmap.org/search?postalcode={zip_code_str}&country=United%20States&format=json'
        
        headers = {
            'User-Agent': 'HealthcareResearchBot/1.0 (contact@example.com)'
        }
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            if data:
                coordinates.append((data[0].get('lat'), data[0].get('lon')))
            else:
                coordinates.append((None, None))
        except Exception:
            coordinates.append((None, None))
        
        time.sleep(2)  # OSM policy compliance

    return coordinates

if __name__ == "__main__":
    base_dir = "./AddressData"
    input_file = os.path.join(base_dir, "CountiesWithZipCodes.csv")
    output_file = os.path.join(base_dir, "CountyZipCoordinates.csv")
    
    print("Loading statewide ZIP coverage...")
    df = pd.read_csv(input_file)
    df = df[df['ZIP Code'].notna()]
    print(f"Processing {len(df)} ZIP codes across {df['County'].nunique()} counties")
    
    # Process in 4 chunks (OSM best practice)
    chunk_size = len(df) // 4 + 1
    results = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        print(f"\nChunk {i//chunk_size + 1}/4: {len(chunk)} ZIPs")
        
        coords = get_coordinates(chunk['ZIP Code'].tolist())
        chunk_results = []
        
        for j in range(len(chunk)):
            chunk_results.append({
                'County': chunk.iloc[j]['County'],
                'ZIP Code': chunk.iloc[j]['ZIP Code'],
                'Latitude': coords[j][0],
                'Longitude': coords[j][1]
            })
        results.extend(chunk_results)
    
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
    print(f"\nCreated: {output_file} ({len(result_df)} records)")
    print("Reference table ready for HPSA/MUA/PC geocoding acceleration!")