# Student Data Matching Pipeline

Production Scale: 10K source records x 150K target records across 18 healthcare professions

## Complete Production Pipeline

Raw Participant Data --database_record_matcher.py--> Fuzzy Providers
|
geocode_hpsa_mua.py + geocode_hpsa_pc_mh_dt.py
|
zip_lookup_generator.py --Cached ZIP table--> HPSA/Rural speedup
|
hpsa_rural_classifier_001.py --> Salesforce upsert

## Core Scripts (5/10 LIVE)

**database_record_matcher.py** - LIVE  
- Production fuzzy matching engine (10Kx150K comparisons in <2 hours)  
- Weighted scoring: 40% last_name + 30% first_name + 10% middle + 20% token_sort  
- Vectorized exact matches → fuzzy fallback (threshold 69)  
- Excel automation: fuzzy matches orange highlighted, Salesforce ID columns yellow  

**geocode_hpsa_mua.py + geocode_hpsa_pc_mh_dt.py** - LIVE  
- HPSA MUA/MUP spatial enrichment across all 18 professions  
- HRSA ArcGIS REST layers (`MedicallyUnderservedAreas_FS` + `MedicallyUnderservedPopulations_FS`)  
- In-place CSV updates: lat/lon coordinates → Yes/No underserved flags  
- Production rate limiting + error handling for 15+ profession files  

**zip_lookup_generator.py** - NEW  
- Statewide ZIP coordinate reference table (Nominatim OSM, FREE, no API key)  
- Input: CountiesWithZipCodes.csv (complete ZIP coverage for single U.S. state)  
- Output: CountyZipCoordinates.csv (cached lookup accelerates Codes 3+4 by 10x)  
- OSM compliant: 2-second rate limiting, single-threaded, academic research use  

**hpsa_rural_classifier_001.py** - LIVE  
- Final classification combining HPSA/MUA/PC + Rural status across all datasets  
- Comprehensive underserved area flags for Salesforce integration  
- Preserves Salesforce ID column end-to-end for production upsert   
