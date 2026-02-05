# Student Data Matching Pipeline

**Production Scale: 10K source records × 150K target records across 18 healthcare professions**

**Pipeline Scale:**
- 10K × 150K record comparisons  
- 18 healthcare professions
- Python, pandas, numpy, fuzzywuzzy, openpyxl, requests, tqdm, logging

Scalable record linkage matching healthcare education programs against license data with Excel automation.

## Core Pipeline Scripts

**Customer_Record_LinkageScript001.py**  
- Production fuzzy matching engine (10K×150K comparisons)  
- Weighted scoring: 40% last_name + 30% first_name + 10% middle + 20% token_sort  
- Vectorized exact matches → fuzzy fallback (threshold 69)  
- Excel output: fuzzy matches orange highlighted, ID columns yellow  

**Geocode_HPSA_MUAScript-001.py**  
- HPSA MUA/MUP spatial enrichment across all 18 professions  
- HRSA GIS API queries for every geocode (lat/lon → Yes/No)  
- In-place CSV updates for 15+ profession files  

_*Note: Several additional scripts comprise the complete pipeline. All scripts will be published soon.*_
