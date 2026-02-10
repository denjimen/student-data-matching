import requests
import pandas as pd
from tqdm import tqdm
import time
import os
import logging

# Configure logging for production use
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

tqdm.pandas()

def query_mua(latitudes, longitudes):
    """Query HRSA Medically Underserved Areas (MUA) API for spatial intersection."""
    base_url = "https://gisportal.hrsa.gov/server/rest/services/Shortage/MedicallyUnderservedAreas_FS/MapServer/0/query"
    results = []
    
    for lat, lon in zip(latitudes, longitudes):
        params = {
            'f': 'json',
            'outFields': 'MUA_DESIGNATION_TYP_CD,MUA_DESIGNATION_TYP_DESC,MUA_SERVICE_AREA_NM,MUA_DESIGNATION_DT_TXT,MUA_UPDATE_DT_TXT,CMN_STATE_ABBR',
            'geometryType': 'esriGeometryPoint',
            'geometry': f"{lon},{lat}",
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelWithin',
            'returnGeometry': 'false'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results.append(bool(data.get('features')))  # True if point is within MUA
        else:
            results.append(False)  # API error
        
        time.sleep(0.1)  # Rate limiting - HRSA APIs throttle heavy usage
    
    return results

def query_mup(latitudes, longitudes):
    """Query HRSA Medically Underserved Populations (MUP) API for spatial intersection."""
    base_url = "https://gisportal.hrsa.gov/server/rest/services/Shortage/MedicallyUnderservedPopulations_FS/MapServer/0/query"
    results = []
    
    for lat, lon in zip(latitudes, longitudes):
        params = {
            'f': 'json',
            'outFields': 'MUP_DESIGNATION_TYP_CD,MUP_DESIGNATION_TYP_DESC,MUP_SERVICE_AREA_NM,MUP_DESIGNATION_DT_TXT,MUP_UPDATE_DT_TXT,CMN_STATE_ABBR',
            'geometryType': 'esriGeometryPoint',
            'geometry': f"{lon},{lat}",
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelWithin',
            'returnGeometry': 'false'
        }
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            results.append(bool(data.get('features')))  # True if point is within MUP
        else:
            results.append(False)  # API error
        
        time.sleep(0.1)  # Rate limiting
    
    return results

def update_hpsa_with_mua_mup(input_file):
    """
    Main ETL function: 
    - Reads HPSA CSV with Lat/Lon columns
    - Queries HRSA MUA/MUP APIs for each geocoded location
    - Adds MUA/MUP columns (Yes/No) 
    - Overwrites original file in-place
    """
    df = pd.read_csv(input_file)
    
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        raise ValueError("Input CSV must contain 'Latitude' and 'Longitude' columns")
    
    latitudes = df['Latitude'].tolist()
    longitudes = df['Longitude'].tolist()
    
    # Query MUAs and MUPs with progress bars
    tqdm.pandas(desc="Processing MUAs")
    mua_results = query_mua(latitudes, longitudes)
    
    tqdm.pandas(desc="Processing MUPs")
    mup_results = query_mup(latitudes, longitudes)

    # Add MUA/MUP columns (simplified names)
    df['MUA'] = ['Yes' if result else 'No' for result in mua_results]
    df['MUP'] = ['Yes' if result else 'No' for result in mup_results]

    # Overwrite original file with enriched data
    df.to_csv(input_file, index=False)
    print(f"Updated output successfully written to {input_file}")
    logging.info(f"Enriched {input_file}: {len(df)} rows, {df['MUA'].sum()} MUA sites, {df['MUP'].sum()} MUP sites")

def main():
    """Process multiple CSV files (one per medical profession) using .env config."""
    input_folder = os.getenv('INPUT_FOLDER', 'InputFolder')
    input_files = os.getenv('INPUT_FILES', 'InputFile.csv').split(',')
    
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
