import requests
import pandas as pd
import time
from tqdm import tqdm
import os  # Import os for path handling

def query_am_i_rural(lat, lon):
    url = f"https://www.ruralhealthinfo.org/contact=json"  # API ACCESS REQUIRES PERMISSION FROM RHIHUB
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def is_rural(json_data):
    if not json_data:
        return None
    
    # Extract RUCA and RUCC information
    ruca_code = float(json_data.get('ruca', {}).get('code', 0))
    rucc_number = json_data.get('rucc', {}).get('number', 0)

    # Rural designation: Custom middle-ground criteria (RUCA >=4 OR RUCA 2-3 + RUCC 5)
    # Balances tract-level granularity with county context; no universal rural definition exists
    if ruca_code >= 4 or (ruca_code >= 2 and rucc_number == 5):
        return True
    return False

def update_hpsa_with_rural_status(input_file):
    df = pd.read_csv(input_file)
    
    if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
        raise ValueError("Input CSV must contain 'Latitude' and 'Longitude' columns")
    
    df['Employed in Rural Area'] = 'No'  # Default value

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Coordinates"):
        json_data = query_am_i_rural(row['Latitude'], row['Longitude'])
        time.sleep(3)  # 3-second delay between API calls
        if is_rural(json_data):
            df.at[index, 'Employed in Rural Area'] = 'Yes'

    # Save the updated DataFrame back to the same file
    df.to_csv(input_file, index=False)
    print(f"Updated output successfully written to {input_file}")

# Main execution - SINGLE FILE ONLY
if __name__ == "__main__":
    """
    PROCESS ONE CSV AT A TIME TO AVOID RHIHUB API OVERLOAD
    
    Bulk loops crash their servers (tested). Run this script separately for each file:
    1. Update the filename below  
    2. Run script
    3. Check results, then repeat for next file
    """
    input_folder = "data/raw"  # ← EDIT with your actual CSV filename
    
    input_files = [
        "your_hpsa_file.csv"  # CHANGE THIS FILENAME
    ]
    
    if len(input_files) > 1:
        print("ERROR: Set input_files to exactly 1 filename to respect RHI API limits")
        exit(1)
    
    try:
        full_input_path = os.path.join(input_folder, input_files[0])
        update_hpsa_with_rural_status(full_input_path)
    except Exception as e:
        print(f"Error processing {input_files[0]}: {str(e)}")