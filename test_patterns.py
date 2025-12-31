import re
import sys
sys.path.insert(0, 'C:/One Intelligcnc agents/agent')
sys.path.insert(0, 'C:/One Intelligcnc agents/ocr_training')

# Test the actual extraction functions
from extractor import extract_fields

# Real text samples from claims
test_texts = [
    # Sample 1: From 1398305 claim - Sandhya
    """Name : Sandhya Raghunath Kate
DOB/Gender : 20-06-2001 (Female)
Member ID : VOLO02333361
Policy No : PRABOS2MOOD
Date of Admission: 06/02/2024
Date of Discharge: 10/02/2024
Hospital Name: ACCIDENT HOSPITAL, VADUJ
Diagnosis: Trimalleolar Fracture
Total Amount: Rs. 33,200/-""",

    # Sample 2: From 1417523 claim - Zoya
    """Patient Name : Miss. ZOYA SAMIR SHAIKH
UHID No.: 2024-2025/3108
IPD No.: IP/1144
Age/Sex: 13/Female
D.O.A.: 12-02-2025
D.O.D.: 16-02-2025
Hospital: BASIL HOSPITAL
Diagnosis: Viral Fever
Total Amount: Rs. 23,805.00""",

    # Sample 3: Mixed format
    """Name of Patient:- Mr. ANIL KUMAR SHARMA
Policy Number: PLY123456789
Claim No: CLM-2024-98765
Date of Birth: 15/06/1985
Admission: 20/01/2024
Discharge: 25/01/2024
Hospital Name: Apollo Hospital Mumbai
Provisional Diagnosis: Acute Appendicitis
Grand Total: Rs. 1,50,000/-"""
]

print("="*70)
print("TESTING IMPROVED EXTRACTION PATTERNS")
print("="*70)

for i, text in enumerate(test_texts):
    print(f"\n{'='*70}")
    print(f"TEST {i+1}")
    print(f"{'='*70}")
    
    result = extract_fields(text)
    
    # Print key fields
    print(f"Patient Name:    {result.get('claimant_name', 'NOT FOUND')}")
    print(f"Policy Number:   {result.get('policy_number', 'NOT FOUND')}")
    print(f"Claim Number:    {result.get('claim_number', 'NOT FOUND')}")
    print(f"DOB:             {result.get('dob', 'NOT FOUND')}")
    print(f"Date of Service: {result.get('date_of_service', 'NOT FOUND')}")
    print(f"Provider:        {result.get('provider_name', 'NOT FOUND')}")
    print(f"Diagnosis:       {result.get('diagnosis', 'NOT FOUND')}")
    print(f"Total Amount:    {result.get('total_amount', 'NOT FOUND')}")

