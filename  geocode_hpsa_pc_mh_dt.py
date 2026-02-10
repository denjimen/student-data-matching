import requests
import pandas as pd
from tqdm import tqdm
import time
import os

def query_hrsa_hpsa(lat, lon):
    """
    Query HRSA HPSA GIS API for Primary Care, Dental, and Mental Health shortage areas.
    Returns dict with True/False for each HPSA type based on point-in-polygon match.
    
    Layer IDs sourced from ArcGIS REST service documentation:
    https://gisportal.hrsa.gov/server/rest/services/Shortage/HealthProfessionalShortageAreas_FS/MapServer
    """
    hpsa_layers = {
        'Primary Care': 10,    # Layer 10: Primary Medical Care HPSAs (ArcGIS REST docs)
        'Dental': 2,           # Layer 2: Dental HPSAs (ArcGIS REST docs)  
        'Mental': 6            # Layer 6: Mental Health HPSAs (ArcGIS REST docs)
    }
    
    results = {}
    
    for hpsa_type, layer in hpsa_layers.items():
        base_url = "https://gisportal.hrsa.gov/server/rest/services/Shortage/HealthProfessionalShortageAreas_FS/MapServer"
        point = f"{lon},{lat}"
        
        params = {
            'f': 'json',
            'outFields': 'DISCIPLINE_CLASS_DESC,HPSA_NM,HPSA_DESIGNATION_DT_TXT,HPSA_DESIG_LAST_UPD_DT_TXT',
            'geometryType': 'esriGeometryPoint',
            'geometry': point,
            'inSR': '4326',
            'spatialRel': 'esriSpatialRelWithin',
            'returnGeometry': 'false'
        }
        
        url = f"{base_url}/{layer}/query"
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('features'):
                results[hpsa_type] = True
            else:
                results[hpsa_type] = False
        else:
            results[hpsa_type] = False  # In case of an error, assume no HPSA
    
    return results


def process_csv_for_hpsa(input_file, output_file):
    """
    Process CSV with geocoded addresses to determine HPSA designations.
    Makes API calls to HRSA GIS service to check if each lat/lon falls within HPSA boundaries.
    """
    df = pd.read_csv(input_file)
    
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        raise ValueError("Input CSV must contain 'Latitude' and 'Longitude' columns")
    
    # Initialize HPSA designation columns
    df['HPSA'] = 'No'              # Any HPSA designation (Primary Care OR Mental Health OR Dental)
    df['MH HPSA'] = 'No'           # Mental Health HPSA designation
    df['PC HPSA'] = 'No'           # Primary Care HPSA designation
    df['Dental HPSA'] = 'No'       # Dental HPSA designation
    
    # Progress bar shows processing status for large files
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking HPSA"):
        lat, lon = row['Latitude'], row['Longitude']
        results = query_hrsa_hpsa(lat, lon)
        
        # Set HPSA flags based on API response
        if results['Mental']:
            df.at[index, 'MH HPSA'] = 'Yes'
            df.at[index, 'HPSA'] = 'Yes'
            
        if results['Primary Care']:
            df.at[index, 'PC HPSA'] = 'Yes'
            df.at[index, 'HPSA'] = 'Yes'
            
        if results['Dental']:
            df.at[index, 'Dental HPSA'] = 'Yes'
            df.at[index, 'HPSA'] = 'Yes'

        time.sleep(3)  # Respect HRSA API rate limits (20 req/min)

    # Save output with original columns + HPSA designation columns
    df.to_csv(output_file, index=False)
    print(f"Output successfully written to {output_file}")


# Main execution - processes ONE geocoded CSV file at a time
# NOTE: Do NOT process multiple files in sequence - will get rate limited/blocked (tested)
# File size doesn't matter unless extremely large (>100k rows) due to 3-sec rate limiting
if __name__ == "__main__":
    # Generic relative folder structure (update paths as needed for your environment)
    base_dir = "./AddressData"  # Relative to script location
    input_folder = os.path.join(base_dir, "ProcessedAddressCSVs")
    output_folder = os.path.join(base_dir, "HSPACSVs")
    
    # Create output folder if it doesn't exist (error prevention)
    os.makedirs(output_folder, exist_ok=True)
    
    # PROCESS ONE FILE AT A TIME ONLY - HRSA blocks multi-file runs
    # Update this filename manually for each run
    input_file = "MatchedCombinedData_AddressProcessed_Pharmacists_STCT_001.csv"
    
    try:
        # Full path construction prevents file not found errors
        full_input_path = os.path.join(input_folder, input_file)
        output_file_name = input_file.replace("AddressProcessed", "HPSA")
        full_output_path = os.path.join(output_folder, output_file_name)
        
        print(f"Processing: {input_file}")
        process_csv_for_hpsa(full_input_path, full_output_path)
        print(f"Completed: {input_file}")
        
    except FileNotFoundError:
        print(f"File not found: {input_file}")
    except pd.errors.EmptyDataError:
        print(f"Empty CSV: {input_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")