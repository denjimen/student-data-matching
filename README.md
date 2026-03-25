# Student Data Matching Pipeline

Production scale: 14.3K source records × 157.3K target records across 18 healthcare professions

Scalable record linkage matching education program data against state license data, with rural/HPSA spatial enrichment and outputs ready for Salesforce (or other) upsert.

## Upstream Data Preparation

This pipeline is designed so that each step builds on the previous one: files accumulate more fields as they move through matching, address enrichment, geocoding, and finally rural/HPSA tagging.

- First, `database_record_matcher.py` links raw program (or Salesforce) participant data to PLA license records, producing MatchedData_*.csv files with IDs and match scores but no practice addresses yet.  
- A separate address-join step (script or manual) then merges practice address columns from a unified address table into each profession’s matched file, keyed by shared IDs such as Salesforce ID and PLA license number.  
- The geocoding and rural/HPSA scripts in this repo assume you start from those address-enriched MatchedCombinedData_*.csv files and then add coordinates and shortage-area flags in successive stages.  
- The address-join step can also be done manually in Excel by matching the unified address table to the matched data on Salesforce ID (and/or PLA license number) and saving the result as MatchedCombinedData_*.csv.

## Complete Production Pipeline

```text
Raw Participant Data --database_record_matcher.py--> Fuzzy Providers
                      |
zip_lookup_generator.py --CountyZipCoordinates.csv--> update_coords_from_zip_lookup.py
                      |
   geocode_in_county_centers.py (county centroid fallback)
                      |
      hpsa_rural_classifier_001.py (RHIhub “Am I Rural?”)
                      |
   geocode_hpsa_mua.py + geocode_hpsa_pc_mh_dt.py (HPSA/MUA tags)
                      |
                 Salesforce upsert
```

> Note: An optional Google-based full-address geocoder is included for experimentation,  
> but the production pipeline relies on free/open geocoding (Nominatim + ZIP table).

## Core Scripts (7 production + 1 optional)

**database_record_matcher.py**  
- Production fuzzy matching engine (10K×150K comparisons in under two hours)  
- Weighted scoring: 40% last_name, 30% first_name, 10% middle, 20% token_sort  
- Vectorized exact matches with fuzzy fallback (threshold 69)  
- Excel automation for manual review (fuzzy rows and ID columns highlighted)

**zip_lookup_generator.py**  
- Builds a statewide ZIP→Lat/Lon reference table using Nominatim / OpenStreetMap (no API key, free)  
- Input: CountiesWithZipCodes.csv (complete ZIP coverage for a single U.S. state)  
- Output: CountyZipCoordinates.csv used to pre-populate coordinates before external classification steps  
- OSM-compliant: conservative rate limiting, single-threaded, academic research use

**update_coords_from_zip_lookup.py**  
- Reads CountyZipCoordinates.csv and merges ZIP-level Latitude/Longitude into existing profession CSVs  
- Overwrites missing or less precise coordinates with ZIP-based values when available  
- Writes out updated CSVs so downstream scripts see improved coordinates by default

**geocode_in_county_centers.py**  
- Final geocoding fallback for records still missing coordinates after ZIP lookup  
- Uses geographic centroids of counties (county centers), not county seats, as a conservative approximation  
- Ensures every record can still be evaluated for rural and HPSA/MUA status at least at the county level

**hpsa_rural_classifier_001.py**  
- Calls the Rural Health Information Hub “Am I Rural?” service to classify locations as rural / non-rural under federal definitions  
- Writes rural status back into each profession CSV so it is available for later HPSA/MUA tagging  
- Preserves ID columns end-to-end to support reliable upsert into Salesforce or other systems

**geocode_hpsa_mua.py + geocode_hpsa_pc_mh_dt.py**  
- Adds HPSA and MUA/MUP attributes to geocoded, rural-classified records across all 18 professions  
- Uses HRSA ArcGIS REST services such as `MedicallyUnderservedAreas_FS` and related HPSA layers to attach shortage-area fields to each row  
- Designed for production: handles 15+ profession files with rate limiting and error handling
- Uses HRSA ArcGIS REST services (MedicallyUnderservedAreas_FS and MedicallyUnderservedPopulations_FS) to tag each geocoded record with MUA/MUP status, with built-in rate limiting and error handling.
- Validates coordinates against the Indiana bounding box before classification, writing NaN for invalid or out-of-state points so bad geocodes do not get mis-labeled as non–MUA/MUP.

**google_geocode_full_addresses.py**  *(optional, paid API)*  
- One-pass helper that uses the Google Maps Geocoding API to resolve full practice addresses, falling back to county or ZIP as needed  
- Produces AddressProcessed files with Latitude/Longitude for high-precision use cases  
- Requires a billable Google Maps project and an API key stored in `GeocodeAPI.env`; not used in the default free/open pipeline