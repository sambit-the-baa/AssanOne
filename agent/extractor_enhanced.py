# agent/extractor_enhanced.py
"""
Enhanced Claim Extractor with NLP, Pattern Matching, and Validation
Production-grade document field extraction for insurance claims
"""

import os
import re
import io
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, date
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from PIL import Image

# Optional imports with graceful fallback
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    from dateutil import parser as date_parser
    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False
    date_parser = None

# Smart Bill Extractor
try:
    from agent.extractors.bill_extractor import BillExtractor, extract_bills_from_text
    BILL_EXTRACTOR_AVAILABLE = True
except ImportError:
    BillExtractor = None
    extract_bills_from_text = None
    BILL_EXTRACTOR_AVAILABLE = False

# Form Section Parser
try:
    from agent.extractors.form_section_parser import FormSectionParser, parse_claim_form, FormSection as FSFormSection
    FORM_PARSER_AVAILABLE = True
except ImportError:
    FormSectionParser = None
    parse_claim_form = None
    FSFormSection = None
    FORM_PARSER_AVAILABLE = False

# Medical Code Extractor
try:
    from agent.extractors.medical_code_extractor import MedicalCodeExtractor, extract_medical_codes, MedicalCodeSummary
    MEDICAL_CODE_EXTRACTOR_AVAILABLE = True
except ImportError:
    MedicalCodeExtractor = None
    extract_medical_codes = None
    MedicalCodeSummary = None
    MEDICAL_CODE_EXTRACTOR_AVAILABLE = False

# Date Parser
try:
    from agent.extractors.date_parser import DateParser, parse_dates, DateExtractionResult
    DATE_PARSER_AVAILABLE = True
except ImportError:
    DateParser = None
    parse_dates = None
    DateExtractionResult = None
    DATE_PARSER_AVAILABLE = False

# Data Validator
try:
    from agent.extractors.data_validator import DataValidator, validate_claim, is_valid_amount, ValidationLevel
    DATA_VALIDATOR_AVAILABLE = True
except ImportError:
    DataValidator = None
    validate_claim = None
    is_valid_amount = None
    ValidationLevel = None
    DATA_VALIDATOR_AVAILABLE = False

# OCR imports
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception:
    convert_from_path = None
    PDF2IMAGE_AVAILABLE = False

try:
    from google.cloud import vision
    VISION_AVAILABLE = True
except Exception:
    vision = None
    VISION_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    pytesseract = None
    TESSERACT_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except Exception:
    fitz = None
    FITZ_AVAILABLE = False


# ============================================================================
# CONFIDENCE LEVELS AND DATA STRUCTURES
# ============================================================================

class ConfidenceLevel(Enum):
    """Extraction confidence levels"""
    HIGH = 0.85
    MEDIUM = 0.65
    LOW = 0.40
    FAILED = 0.0


@dataclass
class ExtractionResult:
    """Result with metadata for each extracted field"""
    value: Any
    confidence: float
    confidence_level: ConfidenceLevel
    sources: List[str]
    raw_values: List[Any]
    validation_status: str
    notes: str = ""


@dataclass
class BillingItem:
    """Individual billing line item"""
    description: str
    amount: float
    quantity: int = 1
    unit_price: float = 0.0
    code: str = ""
    category: str = ""


@dataclass
class FormSection:
    """Represents a section of the claim form"""
    name: str
    content: str
    start_line: int
    end_line: int


# ============================================================================
# ENHANCED CLAIM EXTRACTOR
# ============================================================================

class EnhancedClaimExtractor:
    """
    Production-grade claim document extractor using:
    - Multiple OCR backends with quality scoring
    - Named Entity Recognition (NER) with medical domain
    - Pattern matching with fuzzy logic
    - Context-aware field validation
    - Data correction and normalization
    - Multi-section form parsing
    - Smart billing extraction
    """

    def __init__(self, use_ml_models: bool = True):
        """Initialize extractor with optional ML models"""
        self.use_ml_models = use_ml_models
        self.nlp_general = None
        self.nlp_medical = None

        if use_ml_models and SPACY_AVAILABLE:
            try:
                self.nlp_general = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"Warning: Could not load spaCy model: {e}")

        self.patterns = self._build_patterns()
        self.known_providers = self._load_provider_database()
        self.icd_codes = self._load_icd_codes()
        self.cpt_codes = self._load_cpt_codes()
        self.insurance_companies = self._load_insurance_companies()
        
        # Non-billing number patterns to exclude
        self.non_billing_patterns = self._build_non_billing_patterns()
        
        # PERFORMANCE: Precompile all regex patterns for reuse
        self._compiled_patterns = {}
        for field_name, pattern_list in self.patterns.items():
            self._compiled_patterns[field_name] = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                for pattern in pattern_list
            ]

    def _build_non_billing_patterns(self) -> List[re.Pattern]:
        """Build patterns to identify numbers that are NOT billing amounts"""
        return [
            # Bank account numbers (10-18 digits)
            re.compile(r'(?:A/?C|Account|Acct)[\s:]*\d{10,18}', re.IGNORECASE),
            # IFSC codes
            re.compile(r'[A-Z]{4}0[A-Z0-9]{6}', re.IGNORECASE),
            # PIN codes (6 digits after pincode/pin)
            re.compile(r'(?:pin\s*code|pincode|pin)[\s:]*(\d{6})', re.IGNORECASE),
            # Phone numbers (10 digits)
            re.compile(r'(?:phone|mobile|tel|contact)[\s:]*\+?\d{10,12}', re.IGNORECASE),
            # Policy/Member ID numbers
            re.compile(r'(?:policy|member|cert|certificate)[\s]*(?:no|number|id)?[\s:]*[A-Z0-9\-]{8,20}', re.IGNORECASE),
            # Address numbers (plot no, sector, shop no)
            re.compile(r'(?:plot|sector|shop|flat|house|building)[\s]*(?:no)?[\s:]*\d+', re.IGNORECASE),
            # Dates in numeric format (DDMMYYYY, DD-MM-YYYY)
            re.compile(r'\d{2}[-/]?\d{2}[-/]?\d{4}'),
            # Branch codes
            re.compile(r'(?:branch|brn)[\s:]*\d{4,6}', re.IGNORECASE),
            # RTGS/NEFT codes
            re.compile(r'(?:rtgs|neft|ifsc)[\s:]*[A-Z0-9]+', re.IGNORECASE),
        ]

    def _build_patterns(self) -> Dict[str, List[str]]:
        """Build comprehensive regex patterns for all claim fields"""
        return {
            'policy_number': [
                r'(?:Policy\s*(?:Number|No\.?|#)?\s*[:=]?\s*)([A-Z0-9\-]{6,25})',
                r'(?:POL(?:ICY)?\s*(?:Number|No\.?|#)?\s*[:=]?\s*)([A-Z0-9\-]{6,25})',
                r'(?:Cert\.?(?:ificate)?\s*(?:No\.?|Number|#)?\s*[:=]?\s*)([A-Z0-9\-]{6,25})',
            ],
            'claim_number': [
                r'(?:Claim\s*(?:Number|No\.?|#|ID)?\s*[:=]?\s*)([A-Z0-9\-]{5,25})',
                r'(?:CLM(?:\s*NO\.?)?\s*[:=]?\s*)([A-Z0-9\-]{5,25})',
                r'(?:Reference\s*(?:Number|No\.?)?\s*[:=]?\s*)([A-Z0-9\-]{5,25})',
                r'(?:Case\s*(?:Number|No\.?)?\s*[:=]?\s*)([A-Z0-9\-]{5,25})',
            ],
            'claimant_name': [
                r"(?:Name\s+of\s+(?:the\s+)?(?:Patient|Insured|Claimant))[\s:]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
                r"(?:Patient\s*Name|Claimant\s*Name|Insured\s*Name)[\s:]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
                r'(?:(?:Claimant|Insured|Patient|Member|Policyholder)\s*(?:Name|:)?\s*)\n?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
                r"Name\s*[:=]?\s*([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+)",
            ],
            'date_of_birth': [
                r'(?:(?:DOB|Date\s+of\s+Birth|Birth\s+Date|D\.O\.B)\s*[:=]?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                r'(?:Born\s*(?:on)?\s*[:=]?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                r'(?:Age\s*[:=]?\s*)(\d{1,3})\s*(?:years?|yrs?)',
            ],
            'date_of_admission': [
                r'(?:(?:Date\s+of\s+Admission|Admission\s+Date|Admitted\s+on|DOA)\s*[:=]?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                r'(?:Admit\s+Date\s*[:=]?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            ],
            'date_of_discharge': [
                r'(?:(?:Date\s+of\s+Discharge|Discharge\s+Date|Discharged\s+on|DOD)\s*[:=]?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            ],
            'date_of_service': [
                r'(?:(?:DOS|Date\s+of\s+Service|Service\s+Date)\s*[:=]?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
                r'(?:From\s*[:=]?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            ],
            'hospital_name': [
                r"(?:Hospital\s*Name|Name\s+of\s+Hospital|Hospital)[\s:]+([A-Z][A-Za-z\s&]+(?:Hospital|Medical|Centre|Center|Clinic|Institute))",
                r"(?:Facility\s*Name|Treatment\s+at)[\s:]+([A-Z][A-Za-z\s&]+(?:Hospital|Medical|Centre|Center|Clinic))",
            ],
            'provider_name': [
                r'(?:(?:Treating\s+)?(?:Doctor|Physician|Surgeon|Consultant)\s*(?:Name|:)?\s*)\n?(?:Dr\.?\s*)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:Dr\.?\s+)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:Performed\s+By\s*[:=]?\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r"(?:Attending\s+Physician)[\s:]+(?:Dr\.?\s*)?([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)",
            ],
            'npi_number': [
                r'(?:(?:NPI|Provider\s+ID|National\s+Provider\s+ID)\s*[:=]?\s*)(\d{10})',
                r'(?:NPI#\s*)(\d{10})',
            ],
            # ICD-10 codes (e.g., S82.891A, K35.80, J18.9)
            'diagnosis_icd': [
                r'\b([A-TV-Z]\d{2}(?:\.\d{1,4})?[A-Z]?)\b',  # Standard ICD-10 format
                r'(?:ICD[-\s]?10?\s*[:=]?\s*)([A-TV-Z]\d{2}(?:\.\d{1,4})?[A-Z]?)',
                r'(?:Diagnosis\s*Code\s*[:=]?\s*)([A-TV-Z]\d{2}(?:\.\d{1,4})?[A-Z]?)',
                r'(?:DX\s*[:=]?\s*)([A-TV-Z]\d{2}(?:\.\d{1,4})?[A-Z]?)',
            ],
            'diagnosis_text': [
                r"(?:Diagnosis|Chief\s+Complaint|Presenting\s+Complaint|Nature\s+of\s+Illness)[\s:]+([A-Za-z][A-Za-z\s,]+(?:fracture|injury|disease|syndrome|infection|disorder)?)",
                r"(?:Provisional\s+Diagnosis|Final\s+Diagnosis)[\s:]+([A-Za-z][A-Za-z\s,]+)",
            ],
            'procedure_code': [
                r'(?:(?:Procedure|CPT|HCPCS)\s*(?:Code|#)?\s*[:=]?\s*)(\d{5})',
                r'\b(\d{5})\b(?=\s*[-\s]?\$)',
            ],
            'procedure_name': [
                r"(?:Procedure|Surgery|Operation|Treatment)[\s:]+([A-Za-z][A-Za-z\s]+(?:surgery|procedure|repair|replacement|removal)?)",
                r"(?:Type\s+of\s+Surgery)[\s:]+([A-Za-z][A-Za-z\s]+)",
            ],
            'billed_amount': [
                r'(?:(?:Total\s+)?(?:Bill|Billed|Amount|Charges?|Claim)\s*(?:Amount)?\s*[:=]?\s*)(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d{0,2})',
                r'(?:Grand\s+Total|Net\s+Amount|Total\s+Due|Payable)[\s:]*(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d{0,2})',
                r'(?:Final\s+Bill)[\s:]*(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d{0,2})',
            ],
            'insurance_company': [
                r'(?:(?:Insurance|Carrier|Payer|Plan|TPA)\s*(?:Company|Name)?\s*[:=]?\s*)([A-Z][a-zA-Z\s&]+(?:Insurance|Health|Medical|Blue\s*Cross|Aetna|Cigna|UnitedHealth)?)',
            ],
            'member_id': [
                r'(?:(?:Member|Subscriber|ID|Health\s*ID)\s*(?:Number|No\.?|#|ID)?\s*[:=]?\s*)([A-Z0-9\-]{6,20})',
                r'(?:UHID|MRN|Patient\s*ID)[\s:]*([A-Z0-9\-]{6,20})',
            ],
            'group_number': [
                r'(?:(?:Group|GRP)\s*(?:Number|No\.?|#)?\s*[:=]?\s*)([A-Z0-9\-]{4,20})',
            ],
            'room_type': [
                r"(?:Room\s*(?:Type|Category)|Type\s+of\s+Room|Accommodation)[\s:]+([A-Za-z]+(?:\s+[A-Za-z]+)?)",
                r"(?:Single|Double|General|ICU|ICCU|Private|Semi-Private|Ward)",
            ],
        }

    def _load_provider_database(self) -> Dict[str, Any]:
        """Load known provider names for validation"""
        return {
            'hospitals': ['General Hospital', 'Medical Center', 'Regional Medical'],
            'prefixes': ['Dr.', 'Dr', 'Doctor'],
            'suffixes': ['MD', 'DO', 'DDS', 'DMD', 'PhD', 'RN', 'NP', 'PA'],
        }

    def _load_icd_codes(self) -> set:
        """Load valid ICD-10 code prefixes"""
        return {
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        }

    def _load_cpt_codes(self) -> Dict[str, str]:
        """Load common CPT code ranges"""
        return {
            '99201-99215': 'Office Visit',
            '99281-99285': 'Emergency Department',
            '10000-19999': 'Integumentary',
            '20000-29999': 'Musculoskeletal',
            '30000-39999': 'Respiratory',
            '40000-49999': 'Digestive',
            '50000-59999': 'Urinary',
            '60000-69999': 'Endocrine',
            '70000-79999': 'Radiology',
            '80000-89999': 'Pathology/Lab',
            '90000-99999': 'Medicine',
        }

    def _load_insurance_companies(self) -> List[str]:
        """Load known insurance company names"""
        return [
            'Aetna', 'Cigna', 'UnitedHealthcare', 'Blue Cross Blue Shield',
            'Humana', 'Kaiser Permanente', 'Anthem', 'Centene', 'Molina',
            'WellCare', 'Highmark', 'Independence Blue Cross', 'Carefirst',
            'Health Net', 'Amerigroup', 'Oscar Health', 'Clover Health',
            # Indian Insurance Companies
            'HDFC ERGO', 'ICICI Lombard', 'Star Health', 'Max Bupa',
            'Bajaj Allianz', 'New India Assurance', 'Oriental Insurance',
            'United India', 'National Insurance', 'Reliance General',
            'TATA AIG', 'SBI General', 'Religare', 'Apollo Munich',
            'Care Health', 'Niva Bupa', 'Aditya Birla Health', 'Future Generali',
            'Kotak Mahindra General', 'Cholamandalam MS', 'Iffco Tokio',
            'Liberty General', 'Magma HDI', 'Royal Sundaram',
            'Shriram General', 'Universal Sompo', 'Raheja QBE',
        ]

    def _load_hospital_database(self) -> List[str]:
        """Load known hospital names for fuzzy matching"""
        return [
            # Major Indian Hospital Chains
            'Apollo Hospital', 'Apollo Hospitals', 'Apollo Spectra',
            'Fortis Hospital', 'Fortis Healthcare', 'Fortis Memorial',
            'Max Hospital', 'Max Super Speciality', 'Max Healthcare',
            'Medanta', 'Medanta The Medicity', 'Medanta Hospital',
            'AIIMS', 'All India Institute of Medical Sciences',
            'Manipal Hospital', 'Manipal Hospitals',
            'Columbia Asia', 'Columbia Asia Hospital',
            'Narayana Health', 'Narayana Hrudayalaya',
            'Global Hospital', 'Global Hospitals',
            'Kokilaben Hospital', 'Kokilaben Dhirubhai Ambani',
            'Lilavati Hospital', 'Breach Candy Hospital',
            'Hinduja Hospital', 'Jaslok Hospital',
            'Wockhardt Hospital', 'Wockhardt Hospitals',
            'Rainbow Hospital', 'Rainbow Children Hospital',
            'Yashoda Hospital', 'Yashoda Hospitals',
            'KIMS Hospital', 'KIMS Healthcare',
            'Aster Hospital', 'Aster DM Healthcare',
            'Artemis Hospital', 'Artemis Health Institute',
            'BLK Hospital', 'BLK Super Speciality',
            'Sir Ganga Ram Hospital', 'Ganga Ram Hospital',
            'Safdarjung Hospital', 'RML Hospital', 'GTB Hospital',
            'Basil Hospital', 'Basil Medical Centre',
            # Generic patterns
            'General Hospital', 'District Hospital', 'Civil Hospital',
            'Medical College', 'Medical Centre', 'Medical Center',
            'Nursing Home', 'Healthcare', 'Clinic', 'Polyclinic',
        ]

    def _load_provider_database(self) -> List[str]:
        """Load common provider name patterns for validation"""
        # Common Indian doctor name patterns (titles and suffixes)
        return [
            'Dr.', 'Doctor', 'Dr ', 
            'MD', 'MBBS', 'MS', 'MCh', 'DM', 'DNB',
            'FRCS', 'MRCP', 'FRCOG', 'FICS',
        ]

    # ========================================================================
    # FUZZY MATCHING ENGINE
    # ========================================================================

    def fuzzy_match_hospital(self, extracted_name: str, threshold: int = 70) -> Tuple[str, int]:
        """
        Match extracted hospital name against master database
        Returns: (matched_name, confidence_score)
        """
        if not FUZZY_AVAILABLE or not extracted_name:
            return extracted_name, 0
        
        extracted_clean = self._clean_entity_name(extracted_name)
        hospitals = self._load_hospital_database()
        
        # Try exact match first
        for hospital in hospitals:
            if hospital.lower() in extracted_clean.lower() or extracted_clean.lower() in hospital.lower():
                return hospital, 100
        
        # Fuzzy match
        best_match = process.extractOne(extracted_clean, hospitals, scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= threshold:
            return best_match[0], best_match[1]
        
        return extracted_name, 0

    def fuzzy_match_insurance(self, extracted_name: str, threshold: int = 70) -> Tuple[str, int]:
        """
        Match extracted insurance company against master database
        Returns: (matched_name, confidence_score)
        """
        if not FUZZY_AVAILABLE or not extracted_name:
            return extracted_name, 0
        
        extracted_clean = self._clean_entity_name(extracted_name)
        
        # Try exact match first
        for company in self.insurance_companies:
            if company.lower() in extracted_clean.lower() or extracted_clean.lower() in company.lower():
                return company, 100
        
        # Fuzzy match
        best_match = process.extractOne(extracted_clean, self.insurance_companies, scorer=fuzz.token_sort_ratio)
        if best_match and best_match[1] >= threshold:
            return best_match[0], best_match[1]
        
        return extracted_name, 0

    def fuzzy_match_provider(self, extracted_name: str) -> Tuple[str, bool]:
        """
        Validate and clean provider/doctor name
        Returns: (cleaned_name, is_valid_doctor_name)
        """
        if not extracted_name:
            return extracted_name, False
        
        cleaned = self._clean_entity_name(extracted_name)
        
        # Check for doctor title patterns
        provider_patterns = self._load_provider_database()
        is_doctor = any(pattern.lower() in extracted_name.lower() for pattern in provider_patterns[:3])
        
        # Clean up common OCR errors in names
        cleaned = re.sub(r'\b(Dr\.?|Doctor)\s*', 'Dr. ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Remove trailing qualifications for cleaner name
        cleaned = re.sub(r'\s*,?\s*(MBBS|MD|MS|MCh|DM|DNB|FRCS|MRCP).*$', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip(), is_doctor

    def _clean_entity_name(self, name: str) -> str:
        """Clean and normalize entity names for matching"""
        if not name:
            return ""
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[|\\/_\[\]{}()<>]', ' ', name)
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Remove leading/trailing junk
        cleaned = re.sub(r'^[\W\d]+|[\W\d]+$', '', cleaned)
        
        return cleaned.strip()

    def fuzzy_match_all(self, claim_data: Dict) -> Dict:
        """
        Apply fuzzy matching to all relevant fields in claim data
        Returns enhanced claim_data with matched values and confidence scores
        """
        enhanced = claim_data.copy()
        match_info = {}
        
        # Match hospital name
        if claim_data.get('hospital_name'):
            matched, conf = self.fuzzy_match_hospital(claim_data['hospital_name'])
            if conf >= 70:
                enhanced['hospital_name'] = matched
                match_info['hospital_match'] = {'original': claim_data['hospital_name'], 'matched': matched, 'confidence': conf}
        
        # Match provider/hospital from provider field
        if claim_data.get('provider'):
            # Check if it's actually a hospital name
            matched_hosp, conf_hosp = self.fuzzy_match_hospital(claim_data['provider'])
            if conf_hosp >= 70:
                enhanced['hospital_name'] = enhanced.get('hospital_name') or matched_hosp
                match_info['provider_hospital_match'] = {'matched': matched_hosp, 'confidence': conf_hosp}
            else:
                # Clean as provider name
                cleaned, is_doc = self.fuzzy_match_provider(claim_data['provider'])
                if cleaned:
                    enhanced['provider'] = cleaned
                    match_info['provider_cleaned'] = {'original': claim_data['provider'], 'cleaned': cleaned, 'is_doctor': is_doc}
        
        # Match insurance company
        if claim_data.get('insurance_company'):
            matched, conf = self.fuzzy_match_insurance(claim_data['insurance_company'])
            if conf >= 70:
                enhanced['insurance_company'] = matched
                match_info['insurance_match'] = {'original': claim_data['insurance_company'], 'matched': matched, 'confidence': conf}
        
        # Add match metadata
        enhanced['fuzzy_match_info'] = match_info
        
        return enhanced

    # ========================================================================
    # FORM SECTION PARSING (Part A, B, C, D, E)
    # ========================================================================

    def parse_form_sections(self, text: str) -> Dict[str, FormSection]:
        """
        Parse multi-section forms (Part A, Part B, etc.)
        Returns dictionary of section name -> FormSection
        """
        sections = {}
        lines = text.split('\n')
        
        # Section patterns
        section_patterns = [
            r'(?:PART|SECTION)\s*[-–—]?\s*([A-E])\s*[-–—:]?\s*(.*)',
            r'([A-E])\s*[-–—.]\s*(CLAIM\s+FORM|DETAILS|INFORMATION|PARTICULARS)',
            r'FORM\s*[-–—]?\s*([A-E])',
        ]
        
        current_section = None
        current_start = 0
        
        for i, line in enumerate(lines):
            for pattern in section_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous section
                    if current_section:
                        content = '\n'.join(lines[current_start:i])
                        sections[current_section] = FormSection(
                            name=current_section,
                            content=content,
                            start_line=current_start,
                            end_line=i
                        )
                    
                    # Start new section
                    current_section = f"PART_{match.group(1).upper()}"
                    current_start = i
                    break
        
        # Save last section
        if current_section:
            content = '\n'.join(lines[current_start:])
            sections[current_section] = FormSection(
                name=current_section,
                content=content,
                start_line=current_start,
                end_line=len(lines)
            )
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections['MAIN'] = FormSection(
                name='MAIN',
                content=text,
                start_line=0,
                end_line=len(lines)
            )
        
        return sections

    def parse_form_sections_advanced(self, text: str) -> Dict[str, Any]:
        """
        Parse form sections using the advanced FormSectionParser
        Returns structured data for each section (Part A, B, C, D, E)
        """
        if not FORM_PARSER_AVAILABLE or not FormSectionParser:
            # Fallback to basic parsing
            return {'sections': {}, 'form_type': 'Unknown', 'parse_confidence': 0.0}
        
        parser = FormSectionParser()
        parsed = parser.parse(text)
        
        # Extract data from each section
        result = {
            'form_type': parsed.form_type,
            'parse_confidence': parsed.parse_confidence,
            'section_count': len(parsed.sections),
            'sections': {}
        }
        
        for section_key, section_info in parsed.sections.items():
            section_data = parser.extract_section_data(section_info)
            result['sections'][section_key] = {
                'type': section_info.section_type.value,
                'confidence': section_info.confidence,
                'data': section_data,
                'content_preview': section_info.content[:300] if section_info.content else ''
            }
        
        return result

    # ========================================================================
    # SMART BILLING EXTRACTION (Filters non-billing numbers)
    # ========================================================================

    def _is_non_billing_context(self, text: str, amount_pos: int, window: int = 100) -> bool:
        """Check if amount is in a non-billing context (bank, address, phone, etc.)"""
        start = max(0, amount_pos - window)
        end = min(len(text), amount_pos + window)
        context = text[start:end].lower()
        
        # Non-billing context keywords
        non_billing_keywords = [
            'account', 'a/c', 'bank', 'ifsc', 'rtgs', 'neft', 'branch',
            'pin code', 'pincode', 'postal', 'zip',
            'phone', 'mobile', 'tel', 'contact', 'fax',
            'plot', 'sector', 'shop', 'flat', 'house', 'building', 'floor',
            'policy no', 'member id', 'certificate', 'uhid', 'mrn',
            'date of birth', 'dob', 'age',
        ]
        
        return any(kw in context for kw in non_billing_keywords)

    def extract_billing_items(self, text: str) -> List[BillingItem]:
        """
        Extract itemized billing with smart filtering
        Returns list of BillingItem objects
        """
        items = []
        
        # Billing line patterns (description + amount)
        billing_patterns = [
            # "Room Charges ... Rs. 5000" or "Room Charges: 5000"
            r'([A-Za-z][A-Za-z\s/&]+?)[\s\.]*(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d{0,2})\s*(?:/-)?',
            # Table format: "Description | Amount"
            r'([A-Za-z][A-Za-z\s/&]+?)\s*\|\s*(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d{0,2})',
        ]
        
        # Known billing categories (using set for O(1) lookup)
        billing_keywords = {
            'room', 'bed', 'nursing', 'icu', 'ot', 'operation', 'theatre',
            'surgeon', 'doctor', 'physician', 'consultation', 'visit',
            'medicine', 'drug', 'pharmacy', 'injection', 'iv',
            'pathology', 'lab', 'test', 'investigation', 'diagnostic',
            'radiology', 'x-ray', 'mri', 'ct', 'scan', 'ultrasound', 'usg',
            'implant', 'prosthesis', 'consumable', 'equipment',
            'blood', 'transfusion', 'oxygen',
            'ambulance', 'transport',
            'anesthesia', 'anaesthesia',
            'professional', 'fee', 'charge',
            'misc', 'miscellaneous', 'other', 'sundry',
        }
        
        # Precompile patterns for better performance
        compiled_patterns = [re.compile(p, re.IGNORECASE) for p in billing_patterns]
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # PERFORMANCE: Use set intersection for exact word matches, fallback for partial matches
            line_lower = line.lower()
            line_words = set(line_lower.split())
            # Check both exact word matches (fast) and partial matches for compound words
            has_billing_keyword = bool(line_words & billing_keywords)
            if not has_billing_keyword:
                # Fallback: check for partial matches (e.g., "x-ray", "iv-fluid")
                # Filter out single-char keywords to avoid false positives
                multi_char_keywords = {kw for kw in billing_keywords if len(kw) > 1}
                has_billing_keyword = any(kw in line_lower for kw in multi_char_keywords)
            
            if has_billing_keyword:
                for compiled_pattern in compiled_patterns:
                    matches = compiled_pattern.findall(line)
                    for match in matches:
                        if isinstance(match, tuple) and len(match) >= 2:
                            desc, amount_str = match[0].strip(), match[1]
                            try:
                                amount = float(amount_str.replace(',', ''))
                                # Reasonable billing amount range (10 to 10,000,000)
                                if 10 <= amount <= 10000000:
                                    items.append(BillingItem(
                                        description=desc,
                                        amount=amount,
                                        category=self._categorize_billing(desc)
                                    ))
                            except ValueError:
                                continue
        
        return items

    def _categorize_billing(self, description: str) -> str:
        """Categorize billing item by description"""
        desc_lower = description.lower()
        
        categories = {
            'Room & Board': ['room', 'bed', 'nursing', 'ward', 'icu', 'iccu'],
            'Surgery': ['operation', 'theatre', 'ot ', 'surgery', 'surgical'],
            'Professional Fees': ['doctor', 'surgeon', 'physician', 'consultation', 'fee'],
            'Medicines': ['medicine', 'drug', 'pharmacy', 'injection', 'iv fluid'],
            'Diagnostics': ['pathology', 'lab', 'test', 'investigation'],
            'Radiology': ['x-ray', 'mri', 'ct', 'scan', 'ultrasound', 'usg', 'radiology'],
            'Implants': ['implant', 'prosthesis', 'stent'],
            'Consumables': ['consumable', 'equipment', 'disposable'],
            'Other': ['misc', 'other', 'sundry'],
        }
        
        for category, keywords in categories.items():
            if any(kw in desc_lower for kw in keywords):
                return category
        return 'Other'

    def extract_total_amount(self, text: str) -> Optional[float]:
        """
        Extract the total/final billing amount with smart validation
        """
        # Total amount patterns (ordered by priority)
        total_patterns = [
            r'(?:Grand\s+Total|Net\s+(?:Payable|Amount)|Final\s+(?:Bill|Amount)|Total\s+(?:Bill|Amount|Payable|Due))[\s:]*(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d{0,2})',
            r'(?:Amount\s+Claimed|Claim\s+Amount)[\s:]*(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d{0,2})',
            r'Total[\s:]*(?:Rs\.?|INR|₹)?\s*([\d,]+\.?\d{0,2})\s*(?:/-)?$',
        ]
        
        candidates = []
        
        for pattern in total_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    # Validate: reasonable total range (100 to 10,000,000)
                    if 100 <= amount <= 10000000:
                        # Check it's not a non-billing number
                        pos = text.find(match)
                        if not self._is_non_billing_context(text, pos):
                            candidates.append(amount)
                except ValueError:
                    continue
        
        # Return largest amount (likely the total)
        return max(candidates) if candidates else None

    # ========================================================================
    # DATE STANDARDIZATION
    # ========================================================================

    def standardize_date(self, date_str: str) -> Optional[str]:
        """
        Standardize date to YYYY-MM-DD format
        Handles: DD/MM/YYYY, DD-MM-YYYY, DDMMYYYY, DD.MM.YYYY, etc.
        """
        if not date_str:
            return None
            
        date_str = str(date_str).strip()
        
        # Common date patterns
        date_patterns = [
            # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
            (r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})', '%d/%m/%Y'),
            # DDMMYYYY (no separator)
            (r'^(\d{2})(\d{2})(\d{4})$', '%d%m%Y'),
            # YYYY-MM-DD (ISO format)
            (r'(\d{4})[/\-\.](\d{1,2})[/\-\.](\d{1,2})', '%Y/%m/%d'),
            # DD/MM/YY
            (r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})$', '%d/%m/%y'),
        ]
        
        for pattern, _ in date_patterns:
            match = re.match(pattern, date_str)
            if match:
                groups = match.groups()
                try:
                    if len(groups) == 3:
                        if len(groups[0]) == 4:  # YYYY-MM-DD
                            year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                        elif len(groups[2]) == 4:  # DD-MM-YYYY
                            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                        elif len(groups[2]) == 2:  # DD-MM-YY
                            day, month, year = int(groups[0]), int(groups[1]), int(groups[2])
                            year = 2000 + year if year < 50 else 1900 + year
                        else:
                            continue
                        
                        # Validate date parts
                        if 1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100:
                            return f"{year:04d}-{month:02d}-{day:02d}"
                except (ValueError, IndexError):
                    continue
        
        # Try dateutil as fallback
        if DATEUTIL_AVAILABLE and date_parser:
            try:
                parsed = date_parser.parse(date_str, dayfirst=True)
                return parsed.strftime('%Y-%m-%d')
            except:
                pass
        
        return None

    def extract_dates_advanced(self, text: str, sections: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Advanced date extraction using DateParser
        Extracts all dates with type detection and key date identification
        
        Returns:
            {
                'admission_date': str (YYYY-MM-DD),
                'discharge_date': str (YYYY-MM-DD),
                'date_of_birth': str (YYYY-MM-DD),
                'policy_start': str,
                'policy_end': str,
                'length_of_stay': int (days),
                'all_dates': list of parsed dates,
                'extraction_confidence': float
            }
        """
        if not DATE_PARSER_AVAILABLE or not DateParser:
            # Fallback to basic extraction
            return {
                'available': False,
                'admission_date': None,
                'discharge_date': None,
                'date_of_birth': None,
                'length_of_stay': None
            }
        
        parser = DateParser()
        result = parser.extract_all(text, sections)
        
        return {
            'available': True,
            'admission_date': result.admission_date,
            'discharge_date': result.discharge_date,
            'date_of_birth': result.date_of_birth,
            'policy_start': result.policy_start,
            'policy_end': result.policy_end,
            'length_of_stay': result.length_of_stay,
            'all_dates': [d.to_dict() for d in result.all_dates],
            'key_dates': result.key_dates,
            'extraction_confidence': result.extraction_confidence
        }

    # ========================================================================
    # ICD-10 CODE EXTRACTION
    # ========================================================================

    def extract_icd_codes(self, text: str) -> List[Dict[str, str]]:
        """
        Extract ICD-10 codes with descriptions
        Returns list of {code, description, category}
        
        Uses MedicalCodeExtractor if available for improved accuracy
        """
        # Use advanced extractor if available
        if MEDICAL_CODE_EXTRACTOR_AVAILABLE and MedicalCodeExtractor:
            extractor = MedicalCodeExtractor()
            summary = extractor.extract_all_codes(text)
            
            codes = []
            for med_code in summary.diagnosis_codes:
                codes.append({
                    'code': med_code.code,
                    'description': med_code.description,
                    'category': med_code.category,
                    'confidence': med_code.confidence,
                    'is_primary': med_code.is_primary
                })
            return codes
        
        # Fallback to basic extraction
        codes = []
        seen = set()
        
        # ICD-10 pattern: Letter + 2 digits + optional decimal + up to 4 more digits + optional letter
        icd_pattern = re.compile(r'\b([A-TV-Z]\d{2}(?:\.\d{1,4})?[A-Z]?)\b', re.IGNORECASE)
        
        # Common ICD-10 category descriptions
        icd_categories = {
            'A': 'Infectious diseases', 'B': 'Infectious diseases',
            'C': 'Neoplasms', 'D': 'Blood disorders/Neoplasms',
            'E': 'Endocrine/Metabolic', 'F': 'Mental disorders',
            'G': 'Nervous system', 'H': 'Eye/Ear',
            'I': 'Circulatory system', 'J': 'Respiratory system',
            'K': 'Digestive system', 'L': 'Skin disorders',
            'M': 'Musculoskeletal', 'N': 'Genitourinary',
            'O': 'Pregnancy/Childbirth', 'P': 'Perinatal conditions',
            'Q': 'Congenital abnormalities', 'R': 'Symptoms/Signs',
            'S': 'Injury', 'T': 'Injury/Poisoning',
            'V': 'External causes', 'W': 'External causes',
            'X': 'External causes', 'Y': 'External causes',
            'Z': 'Health status factors',
        }
        
        # PERFORMANCE: Precompile non-medical context keywords check
        non_medical_keywords = ['pin', 'phone', 'mobile', 'account', 'address']
        
        matches = icd_pattern.finditer(text)
        for match in matches:
            code = match.group(1).upper()
            
            # Skip if already seen or invalid
            if code in seen or code[0] == 'U':  # U codes are special use
                continue
                
            # Validate it's a real ICD code context
            pos = match.start()
            context = text[max(0, pos-50):min(len(text), pos+50)].lower()
            
            # PERFORMANCE: Use any() with generator instead of loop
            if any(kw in context for kw in non_medical_keywords):
                continue
            
            seen.add(code)
            category = icd_categories.get(code[0], 'Unknown')
            
            # Try to find nearby diagnosis description
            desc = self._find_diagnosis_description(text, pos)
            
            codes.append({
                'code': code,
                'description': desc,
                'category': category
            })
        
        return codes
    
    def extract_medical_codes_advanced(self, text: str) -> Dict[str, Any]:
        """
        Extract all medical codes using advanced MedicalCodeExtractor
        Returns comprehensive code summary with diagnosis and procedure codes
        """
        if not MEDICAL_CODE_EXTRACTOR_AVAILABLE or not MedicalCodeExtractor:
            return {
                'available': False,
                'diagnosis_codes': self.extract_icd_codes(text),
                'procedure_codes': [],
                'primary_diagnosis': None
            }
        
        extractor = MedicalCodeExtractor()
        summary = extractor.extract_all_codes(text)
        formatted = extractor.format_codes_for_display(summary)
        
        return {
            'available': True,
            'diagnosis_codes': [c.to_dict() for c in summary.diagnosis_codes],
            'procedure_codes': [c.to_dict() for c in summary.procedure_codes],
            'primary_diagnosis': summary.primary_diagnosis.to_dict() if summary.primary_diagnosis else None,
            'icd_codes_str': formatted['icd_codes_str'],
            'summary_text': formatted['summary_text'],
            'extraction_confidence': summary.extraction_confidence
        }

    def _find_diagnosis_description(self, text: str, code_pos: int) -> str:
        """Find diagnosis description near an ICD code"""
        # Look for text before or after the code
        start = max(0, code_pos - 100)
        end = min(len(text), code_pos + 100)
        context = text[start:end]
        
        # Common diagnosis pattern
        desc_pattern = r'(?:diagnosis|dx|complaint|illness|condition)[\s:]+([A-Za-z][A-Za-z\s,]+)'
        match = re.search(desc_pattern, context, re.IGNORECASE)
        if match:
            return match.group(1).strip()[:50]  # Limit length
        
        return ""

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, float]:
        """
        Extract text from PDF using multiple OCR backends
        Returns (text, confidence_score)
        PERFORMANCE: Uses early exit strategy to avoid unnecessary OCR attempts
        """
        text = ""
        confidence = 0.0

        # Try PyMuPDF first (fastest for text-based PDFs)
        if FITZ_AVAILABLE and fitz:
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
                # PERFORMANCE: Early exit if we got sufficient text (cache strip result)
                text_stripped = text.strip()
                if text_stripped and len(text_stripped) > 100:
                    confidence = 0.95
                    return text, confidence
            except Exception as e:
                print(f"PyMuPDF extraction failed: {e}")

        # Try Tesseract OCR for image-based PDFs (only if PyMuPDF failed)
        if TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE and pytesseract and convert_from_path:
            try:
                # PERFORMANCE: Use 200 DPI for good balance between speed and quality
                images = convert_from_path(pdf_path, dpi=200)
                for img in images:
                    page_text = pytesseract.image_to_string(img)
                    text += page_text + "\n"
                    # PERFORMANCE: Early exit if we have enough text (cache strip result)
                    text_stripped = text.strip()
                    if len(text_stripped) > 500:
                        break
                text_stripped = text.strip()
                if text_stripped:
                    confidence = 0.75
                    return text, confidence
            except Exception as e:
                print(f"Tesseract OCR failed: {e}")

        # Try Google Vision API (only as last resort)
        if VISION_AVAILABLE and vision and not text:
            try:
                client = vision.ImageAnnotatorClient()
                with open(pdf_path, 'rb') as f:
                    content = f.read()
                image = vision.Image(content=content)
                response = client.document_text_detection(image=image)
                if response.full_text_annotation:
                    text = response.full_text_annotation.text
                    confidence = 0.90
                    return text, confidence
            except Exception as e:
                print(f"Google Vision failed: {e}")

        return text, confidence

    def extract_field(self, text: str, field_name: str) -> ExtractionResult:
        """Extract a single field using multiple strategies"""
        candidates = []
        sources = []

        # Strategy 1: Regex patterns (use precompiled patterns for performance)
        if field_name in self._compiled_patterns:
            for compiled_pattern in self._compiled_patterns[field_name]:
                try:
                    matches = compiled_pattern.findall(text)
                    for match in matches:
                        if isinstance(match, tuple):
                            match = match[0] if match[0] else match[-1]
                        if match and len(str(match).strip()) > 0:
                            candidates.append(str(match).strip())
                            sources.append('regex')
                except Exception:
                    continue

        # Strategy 2: NER (for names, organizations)
        if self.nlp_general and field_name in ['claimant_name', 'provider_name', 'insurance_company']:
            try:
                doc = self.nlp_general(text[:5000])  # Limit for performance
                entity_types = {
                    'claimant_name': ['PERSON'],
                    'provider_name': ['PERSON', 'ORG'],
                    'insurance_company': ['ORG'],
                }
                for ent in doc.ents:
                    if ent.label_ in entity_types.get(field_name, []):
                        candidates.append(ent.text)
                        sources.append('ner')
            except Exception:
                pass

        # Strategy 3: Fuzzy matching for known values
        if FUZZY_AVAILABLE and field_name == 'insurance_company' and candidates:
            for candidate in candidates[:]:
                best_match = process.extractOne(candidate, self.insurance_companies, scorer=fuzz.ratio)
                if best_match and best_match[1] > 80:
                    candidates.append(best_match[0])
                    sources.append('fuzzy')

        # Determine best value and confidence
        if candidates:
            # Use most common candidate or first one
            best_value = max(set(candidates), key=candidates.count)
            confidence = min(0.95, 0.5 + (len(candidates) * 0.1))
            confidence_level = (
                ConfidenceLevel.HIGH if confidence >= 0.85 else
                ConfidenceLevel.MEDIUM if confidence >= 0.65 else
                ConfidenceLevel.LOW
            )
            validation_status = "valid"
        else:
            best_value = None
            confidence = 0.0
            confidence_level = ConfidenceLevel.FAILED
            validation_status = "not_found"

        return ExtractionResult(
            value=best_value,
            confidence=confidence,
            confidence_level=confidence_level,
            sources=list(set(sources)),
            raw_values=candidates,
            validation_status=validation_status,
        )

    def extract_fields(self, text: str) -> Dict[str, Any]:
        """
        Extract all fields from pre-extracted OCR text.
        This method is called by the pipeline after OCR has already been done.
        
        Uses enhanced extraction with:
        - Multi-section form parsing
        - Smart billing extraction (filters non-billing numbers)
        - ICD-10 code extraction
        - Date standardization
        
        Args:
            text: Pre-extracted OCR text string
            
        Returns:
            Dictionary with extracted claim fields
        """
        if not text or not text.strip():
            return {
                'success': False,
                'error': 'Empty text provided',
                'raw_text_preview': '',
                'policy_number': None,
                'claim_number': None,
                'claimant_name': None,
                'dob': None,
                'date_of_loss': None,
                'amounts_found': None,
                'provider': None,
                'diagnosis': None,
            }

        # Parse form sections (Part A, B, C, D, E)
        sections = self.parse_form_sections(text)
        
        # Use advanced form parser if available
        advanced_sections = None
        if FORM_PARSER_AVAILABLE:
            advanced_sections = self.parse_form_sections_advanced(text)
        
        # Extract fields from appropriate sections
        field_names = [
            'policy_number', 'claim_number', 'claimant_name', 'date_of_birth',
            'date_of_service', 'date_of_admission', 'date_of_discharge',
            'provider_name', 'hospital_name', 'npi_number',
            'diagnosis_icd', 'diagnosis_text', 'procedure_code', 'procedure_name',
            'billed_amount', 'insurance_company', 'member_id', 'group_number',
            'room_type',
        ]

        extracted = {}
        for field_name in field_names:
            result = self.extract_field(text, field_name)
            extracted[field_name] = result.value

        # SMART BILLING: Use BillExtractor if available (filters false positives)
        billing_items = []
        total_amount = None
        billing_summary = []
        category_summary = []
        smart_billing = None
        
        if BILL_EXTRACTOR_AVAILABLE and BillExtractor:
            bill_extractor = BillExtractor()
            smart_billing = bill_extractor.extract_bills(text)
            
            # Convert to our format
            for item in smart_billing.items:
                billing_items.append(BillingItem(
                    description=item.description,
                    amount=item.amount,
                    quantity=item.quantity,
                    unit_price=item.unit_price,
                    code=item.code,
                    category=item.category.value if hasattr(item.category, 'value') else str(item.category)
                ))
            
            total_amount = smart_billing.total or smart_billing.subtotal
            
            # Format billing items for display
            if billing_items:
                by_category = {}
                for item in billing_items:
                    cat = item.category
                    if cat not in by_category:
                        by_category[cat] = 0
                    by_category[cat] += item.amount
                    billing_summary.append(f"{item.description}: ₹{item.amount:,.2f}")
                
                category_summary = [f"{cat}: ₹{amt:,.2f}" for cat, amt in by_category.items()]
        else:
            # Fallback to legacy extraction
            billing_items = self.extract_billing_items(text)
            total_amount = self.extract_total_amount(text)
            
            # Format billing items for display
            if billing_items:
                by_category = {}
                for item in billing_items:
                    if item.category not in by_category:
                        by_category[item.category] = 0
                    by_category[item.category] += item.amount
                    billing_summary.append(f"{item.description}: ₹{item.amount:,.2f}")
                
                # Create category summary
                category_summary = [f"{cat}: ₹{amt:,.2f}" for cat, amt in by_category.items()]

        # ADVANCED MEDICAL CODE EXTRACTION
        medical_codes = self.extract_medical_codes_advanced(text)
        icd_codes = medical_codes.get('diagnosis_codes', [])
        procedure_codes = medical_codes.get('procedure_codes', [])
        primary_diagnosis = medical_codes.get('primary_diagnosis')
        
        # Format for backward compatibility
        icd_codes_str = medical_codes.get('icd_codes_str') or (', '.join([c.get('code', '') for c in icd_codes[:5]]) if icd_codes else None)
        icd_with_desc = []
        for c in icd_codes:
            code = c.get('code', '')
            desc = c.get('description', '') or c.get('category', '')
            conf = c.get('confidence', 0)
            is_primary = c.get('is_primary', False)
            primary_marker = ' [PRIMARY]' if is_primary else ''
            icd_with_desc.append(f"{code} ({desc}) [{conf:.0%}]{primary_marker}")

        # ADVANCED DATE EXTRACTION
        date_extraction = self.extract_dates_advanced(text, sections)
        
        # Use advanced dates if available, otherwise fallback to basic extraction
        if date_extraction.get('available'):
            dob_std = date_extraction.get('date_of_birth')
            admission_std = date_extraction.get('admission_date')
            discharge_std = date_extraction.get('discharge_date')
            length_of_stay = date_extraction.get('length_of_stay')
            policy_start = date_extraction.get('policy_start')
            policy_end = date_extraction.get('policy_end')
        else:
            # Fallback to basic date standardization
            dob_raw = extracted.get('date_of_birth')
            dob_std = self.standardize_date(dob_raw) if dob_raw else None
            
            admission_raw = extracted.get('date_of_admission')
            admission_std = self.standardize_date(admission_raw) if admission_raw else None
            
            discharge_raw = extracted.get('date_of_discharge')
            discharge_std = self.standardize_date(discharge_raw) if discharge_raw else None
            
            length_of_stay = None
            policy_start = None
            policy_end = None
            
            # Calculate length of stay if both dates available
            if admission_std and discharge_std:
                try:
                    from datetime import datetime
                    adm = datetime.strptime(admission_std, '%Y-%m-%d')
                    dis = datetime.strptime(discharge_std, '%Y-%m-%d')
                    length_of_stay = (dis - adm).days
                except:
                    pass

        # Combine diagnosis (ICD code + text description)
        diagnosis_combined = None
        if primary_diagnosis:
            # Use primary diagnosis from advanced extractor
            diagnosis_combined = f"{primary_diagnosis.get('code', '')} - {primary_diagnosis.get('description', '') or primary_diagnosis.get('category', '')}"
        elif icd_codes_str:
            diagnosis_combined = icd_codes_str
            if extracted.get('diagnosis_text'):
                diagnosis_combined += f" - {extracted.get('diagnosis_text')}"
        elif extracted.get('diagnosis_text'):
            diagnosis_combined = extracted.get('diagnosis_text')

        # Build sections summary
        sections_found = list(sections.keys())

        # Build result dictionary
        result = {
            'success': True,
            'raw_text_preview': text[:1500],  # Reduced preview
            
            # Basic Info
            'policy_number': extracted.get('policy_number'),
            'claim_number': extracted.get('claim_number'),
            'member_id': extracted.get('member_id'),
            'group_number': extracted.get('group_number'),
            
            # Patient Info
            'claimant_name': extracted.get('claimant_name'),
            'dob': dob_std,
            'dob_raw': extracted.get('date_of_birth'),
            
            # Dates (Standardized with Advanced Parser)
            'date_of_admission': admission_std,
            'date_of_discharge': discharge_std,
            'date_of_service': extracted.get('date_of_service'),
            'length_of_stay': length_of_stay,
            'policy_start_date': policy_start,
            'policy_end_date': policy_end,
            'date_of_loss': admission_std or extracted.get('date_of_service'),
            
            # Provider Info
            'provider': extracted.get('provider_name'),
            'hospital_name': extracted.get('hospital_name'),
            'npi_number': extracted.get('npi_number'),
            'insurance_company': extracted.get('insurance_company'),
            
            # Medical Info (ICD-10 Enhanced with Advanced Extraction)
            'diagnosis': diagnosis_combined,
            'diagnosis_icd_codes': icd_codes,
            'diagnosis_icd_list': icd_with_desc,
            'primary_diagnosis': primary_diagnosis,
            'procedure_code': extracted.get('procedure_code'),
            'procedure_name': extracted.get('procedure_name'),
            'procedure_codes_extracted': procedure_codes,  # CPT codes from advanced extractor
            'medical_code_confidence': medical_codes.get('extraction_confidence', 0.0),
            
            # SMART BILLING (Filtered)
            'billed_amount': total_amount or extracted.get('billed_amount'),
            'billing_items': [asdict(item) for item in billing_items] if billing_items else [],
            'billing_summary': billing_summary[:10],  # Top 10 items
            'billing_by_category': category_summary,
            'total_itemized': sum(item.amount for item in billing_items) if billing_items else None,
            
            # Additional
            'room_type': extracted.get('room_type'),
            
            # Form sections found
            'form_sections': sections_found,
            
            # Legacy field (for backward compatibility)
            'amounts_found': f"Total: ₹{total_amount:,.2f}" if total_amount else None,
        }
        
        # Add advanced form section data if available
        if advanced_sections and advanced_sections.get('sections'):
            result['form_sections_detailed'] = advanced_sections['sections']
            result['form_type'] = advanced_sections.get('form_type', 'Unknown')
            result['section_parse_confidence'] = advanced_sections.get('parse_confidence', 0.0)
        
        # Apply fuzzy matching to standardize hospital, provider, and insurance names
        result = self.fuzzy_match_all(result)
        
        # Apply data validation to filter gibberish and invalid amounts
        if DATA_VALIDATOR_AVAILABLE and DataValidator:
            result = self._validate_and_clean_result(result)
        
        return result
    
    def _validate_and_clean_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean extracted data
        Removes gibberish text and invalid amounts
        """
        if not DATA_VALIDATOR_AVAILABLE or not DataValidator:
            return result
        
        validator = DataValidator(ValidationLevel.MODERATE)
        
        # Validate billing items
        if 'billing_items' in result and result['billing_items']:
            validated_items = []
            for item in result['billing_items']:
                if isinstance(item, dict):
                    desc = item.get('description', '')
                    amt = item.get('amount', 0)
                    
                    # Validate the item
                    validation = validator.validate_billing_item(desc, amt)
                    if validation.is_valid:
                        validated_items.append(item)
                    else:
                        # Log rejected item for debugging
                        pass  # Could add logging here
            
            result['billing_items'] = validated_items
            result['billing_items_validated'] = True
            
            # Recalculate totals based on validated items
            if validated_items:
                result['total_itemized'] = sum(item.get('amount', 0) for item in validated_items)
                result['billing_summary'] = [
                    f"{item.get('description', 'Unknown')}: ₹{item.get('amount', 0):,.2f}"
                    for item in validated_items[:10]
                ]
        
        # Validate billed_amount
        if 'billed_amount' in result and result['billed_amount']:
            amount_validation = validator.validate_amount(
                result['billed_amount'], 
                'billed_amount',
                'total_bill'
            )
            if not amount_validation.is_valid:
                # If total is invalid, use sum of validated items instead
                if result.get('total_itemized'):
                    result['billed_amount'] = result['total_itemized']
                    result['billed_amount_source'] = 'calculated_from_items'
        
        # Clean text fields that might have gibberish
        text_fields = ['claimant_name', 'hospital_name', 'provider', 'diagnosis']
        for field in text_fields:
            if field in result and result[field]:
                value = str(result[field])
                is_gibberish, conf = validator.is_gibberish(value)
                if is_gibberish and conf > 0.6:
                    result[field] = None
                    result[f'{field}_rejected'] = f'Gibberish detected ({conf:.0%})'
        
        result['data_validated'] = True
        return result

    def extract_all_fields(self, pdf_path: str) -> Dict[str, Any]:
        """Extract all fields from a PDF claim document (does its own OCR)"""
        # Get text from PDF
        text, ocr_confidence = self.extract_text_from_pdf(pdf_path)

        if not text.strip():
            return {
                'success': False,
                'error': 'Could not extract text from PDF',
                'ocr_confidence': 0.0,
                'fields': {},
            }

        # Extract each field
        fields = {}
        field_names = [
            'policy_number', 'claim_number', 'claimant_name', 'date_of_birth',
            'date_of_service', 'provider_name', 'npi_number', 'diagnosis',
            'procedure_code', 'billed_amount', 'insurance_company',
            'member_id', 'group_number',
        ]

        for field_name in field_names:
            result = self.extract_field(text, field_name)
            fields[field_name] = asdict(result)

        # Calculate overall confidence
        confidences = [f['confidence'] for f in fields.values() if f['confidence'] > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            'success': True,
            'ocr_confidence': ocr_confidence,
            'extraction_confidence': avg_confidence,
            'raw_text': text[:2000],  # First 2000 chars for reference
            'fields': fields,
        }

    def extract_all(self, pdf_path: str) -> Dict[str, Any]:
        """
        Complete extraction pipeline with structured sections
        
        Pipeline Steps:
        1. Convert PDF to high-quality images and run OCR
        2. Parse form sections (Part A, B, C, D, E)
        3. Extract structured data per section
        4. Apply fuzzy matching for entity standardization
        5. Validate and score extraction confidence
        
        Returns:
            {
                'success': bool,
                'data': {...extracted fields...},
                'validation': {...validation results...},
                'confidence': {...confidence scores...}
            }
        """
        # Step 1: Convert PDF and run OCR with confidence scores
        raw_text, ocr_confidence = self.extract_text_from_pdf(pdf_path)
        
        if not raw_text.strip():
            return {
                'success': False,
                'error': 'Could not extract text from PDF',
                'data': {},
                'validation': {'status': 'failed', 'errors': ['No text extracted']},
                'confidence': {'ocr': 0.0, 'overall': 0.0}
            }
        
        # Step 2: Separate form sections using advanced parser
        sections = self.parse_form_sections(raw_text)
        
        # Use advanced form parser if available
        advanced_sections = None
        if FORM_PARSER_AVAILABLE:
            advanced_sections = self.parse_form_sections_advanced(raw_text)
        
        # Step 3: Extract structured data per section
        extracted = {
            'insured': self._extract_insured_details(raw_text, sections),
            'hospitalization': self._extract_hospitalization_details(raw_text, sections),
            'billing': self._extract_billing_details(raw_text, sections),
            'medical': self._extract_medical_details(raw_text, sections),
            'dates': self._extract_and_standardize_dates(raw_text),
        }
        
        # Merge in advanced section data if available
        if advanced_sections and advanced_sections.get('sections'):
            extracted['form_sections'] = advanced_sections['sections']
            extracted['form_type'] = advanced_sections.get('form_type', 'Unknown')
            extracted['section_parse_confidence'] = advanced_sections.get('parse_confidence', 0.0)
        
        # Step 4: Apply fuzzy matching for entity standardization
        extracted = self._apply_fuzzy_matching(extracted)
        
        # Step 5: Validate extracted data and calculate confidence
        validation = self._validate_extraction(extracted)
        confidence = self._calculate_extraction_confidence(extracted, ocr_confidence)
        
        return {
            'success': True,
            'data': extracted,
            'validation': validation,
            'confidence': confidence,
            'raw_text_preview': raw_text[:1500],
            'sections_found': list(sections.keys()) if sections else []
        }

    def _extract_insured_details(self, text: str, sections: Dict) -> Dict[str, Any]:
        """Extract insured/patient details from Part A or full text"""
        # Try to use Part A section if available
        section_text = text
        if sections and 'A' in sections:
            section_text = sections['A'].content + '\n' + text
        
        return {
            'policy_number': self.extract_field(section_text, 'policy_number').value,
            'claim_number': self.extract_field(section_text, 'claim_number').value,
            'member_id': self.extract_field(section_text, 'member_id').value,
            'group_number': self.extract_field(section_text, 'group_number').value,
            'claimant_name': self.extract_field(section_text, 'claimant_name').value,
            'date_of_birth': self.extract_field(section_text, 'date_of_birth').value,
            'insurance_company': self.extract_field(section_text, 'insurance_company').value,
        }

    def _extract_hospitalization_details(self, text: str, sections: Dict) -> Dict[str, Any]:
        """Extract hospitalization details from Part B or full text"""
        section_text = text
        if sections and 'B' in sections:
            section_text = sections['B'].content + '\n' + text
        
        admission = self.extract_field(section_text, 'date_of_admission').value
        discharge = self.extract_field(section_text, 'date_of_discharge').value
        
        # Calculate length of stay if both dates available
        los = None
        if admission and discharge:
            try:
                if DATEUTIL_AVAILABLE and date_parser:
                    adm_date = date_parser.parse(admission)
                    dis_date = date_parser.parse(discharge)
                    los = (dis_date - adm_date).days
            except:
                pass
        
        return {
            'hospital_name': self.extract_field(section_text, 'hospital_name').value,
            'provider_name': self.extract_field(section_text, 'provider_name').value,
            'npi_number': self.extract_field(section_text, 'npi_number').value,
            'date_of_admission': admission,
            'date_of_discharge': discharge,
            'length_of_stay': los,
            'room_type': self.extract_field(section_text, 'room_type').value,
        }

    def _extract_billing_details(self, text: str, sections: Dict) -> Dict[str, Any]:
        """
        Extract billing details with smart filtering
        Uses the dedicated BillExtractor to filter out false positives
        (addresses, phone numbers, pin codes, etc.)
        """
        section_text = text
        if sections and 'C' in sections:
            section_text = sections['C'].content + '\n' + text
        
        # Use smart BillExtractor if available
        if BILL_EXTRACTOR_AVAILABLE and BillExtractor:
            bill_extractor = BillExtractor()
            billing_summary = bill_extractor.extract_bills(section_text)
            
            # Convert to our format
            item_details = []
            by_category = {}
            
            for item in billing_summary.items:
                item_dict = item.to_dict()
                item_details.append({
                    'description': item_dict['description'],
                    'amount': item_dict['amount'],
                    'category': item_dict['category']
                })
                
                cat = item_dict['category']
                if cat not in by_category:
                    by_category[cat] = 0
                by_category[cat] += item_dict['amount']
            
            return {
                'total_billed': billing_summary.total or billing_summary.subtotal,
                'itemized_total': billing_summary.subtotal,
                'items': item_details[:20],
                'by_category': by_category,
                'item_count': len(billing_summary.items),
                'discount': billing_summary.discount,
                'tax': billing_summary.tax,
                'discrepancy': abs((billing_summary.total or 0) - billing_summary.subtotal) if billing_summary.total else None,
                'extraction_confidence': billing_summary.extraction_confidence
            }
        
        # Fallback to legacy extraction
        billing_items = self.extract_billing_items(section_text)
        total_amount = self.extract_total_amount(section_text)
        
        # Categorize billing items
        by_category = {}
        item_details = []
        
        for item in billing_items:
            if item.category not in by_category:
                by_category[item.category] = {'total': 0, 'items': []}
            by_category[item.category]['total'] += item.amount
            by_category[item.category]['items'].append({
                'description': item.description,
                'amount': item.amount,
                'quantity': item.quantity,
                'unit_price': item.unit_price
            })
            item_details.append({
                'description': item.description,
                'amount': item.amount,
                'category': item.category
            })
        
        # Calculate totals
        itemized_total = sum(item.amount for item in billing_items)
        
        return {
            'total_billed': total_amount or itemized_total,
            'itemized_total': itemized_total,
            'items': item_details[:20],  # Top 20 items
            'by_category': {k: v['total'] for k, v in by_category.items()},
            'item_count': len(billing_items),
            'discrepancy': abs((total_amount or 0) - itemized_total) if total_amount else None
        }

    def _extract_medical_details(self, text: str, sections: Dict) -> Dict[str, Any]:
        """Extract medical codes and diagnoses from Part D or full text"""
        section_text = text
        if sections and 'D' in sections:
            section_text = sections['D'].content + '\n' + text
        
        # Extract ICD-10 codes
        icd_codes = self.extract_icd10_codes(section_text)
        icd_with_desc = []
        for code in icd_codes:
            desc = self.get_icd_description(code, section_text)
            icd_with_desc.append({'code': code, 'description': desc})
        
        return {
            'diagnosis_text': self.extract_field(section_text, 'diagnosis_text').value,
            'icd_codes': icd_codes,
            'icd_details': icd_with_desc,
            'procedure_code': self.extract_field(section_text, 'procedure_code').value,
            'procedure_name': self.extract_field(section_text, 'procedure_name').value,
        }

    def _extract_and_standardize_dates(self, text: str) -> Dict[str, Any]:
        """Extract and standardize all dates in the document"""
        dates = {}
        date_fields = ['date_of_birth', 'date_of_service', 'date_of_admission', 'date_of_discharge']
        
        for field in date_fields:
            raw = self.extract_field(text, field).value
            standardized = self.standardize_date(raw) if raw else None
            dates[field] = {
                'raw': raw,
                'standardized': standardized,
                'valid': standardized is not None
            }
        
        return dates

    def _apply_fuzzy_matching(self, extracted: Dict) -> Dict:
        """Apply fuzzy matching to standardize entity names"""
        # Match hospital name
        if extracted.get('hospitalization', {}).get('hospital_name'):
            matched, conf = self.fuzzy_match_hospital(extracted['hospitalization']['hospital_name'])
            if conf >= 70:
                extracted['hospitalization']['hospital_name_matched'] = matched
                extracted['hospitalization']['hospital_match_confidence'] = conf
        
        # Match provider name
        if extracted.get('hospitalization', {}).get('provider_name'):
            cleaned, is_doc = self.fuzzy_match_provider(extracted['hospitalization']['provider_name'])
            extracted['hospitalization']['provider_name_cleaned'] = cleaned
            extracted['hospitalization']['is_doctor'] = is_doc
        
        # Match insurance company
        if extracted.get('insured', {}).get('insurance_company'):
            matched, conf = self.fuzzy_match_insurance(extracted['insured']['insurance_company'])
            if conf >= 70:
                extracted['insured']['insurance_company_matched'] = matched
                extracted['insured']['insurance_match_confidence'] = conf
        
        return extracted

    def _validate_extraction(self, extracted: Dict) -> Dict[str, Any]:
        """Validate all extracted fields and return validation results"""
        errors = []
        warnings = []
        
        # Validate required fields
        required = [
            ('insured.policy_number', extracted.get('insured', {}).get('policy_number')),
            ('insured.claimant_name', extracted.get('insured', {}).get('claimant_name')),
            ('billing.total_billed', extracted.get('billing', {}).get('total_billed')),
        ]
        
        for field_path, value in required:
            if not value:
                warnings.append(f"Missing: {field_path}")
        
        # Validate dates
        dates = extracted.get('dates', {})
        for date_field, date_info in dates.items():
            if date_info.get('raw') and not date_info.get('valid'):
                errors.append(f"Invalid date format: {date_field}")
        
        # Validate billing consistency
        billing = extracted.get('billing', {})
        if billing.get('discrepancy') and billing['discrepancy'] > 1000:
            warnings.append(f"Billing discrepancy: ₹{billing['discrepancy']:,.2f}")
        
        # Validate ICD codes
        medical = extracted.get('medical', {})
        icd_codes = medical.get('icd_codes', [])
        for code in icd_codes:
            if code and code[0].upper() not in self.icd_codes:
                errors.append(f"Invalid ICD prefix: {code}")
        
        # Validate NPI (US format - 10 digits)
        npi = extracted.get('hospitalization', {}).get('npi_number')
        if npi and not re.match(r'^\d{10}$', str(npi)):
            warnings.append(f"Invalid NPI format: {npi}")
        
        return {
            'status': 'error' if errors else 'warning' if warnings else 'valid',
            'errors': errors,
            'warnings': warnings,
            'fields_extracted': self._count_extracted_fields(extracted),
            'completeness': self._calculate_completeness(extracted)
        }

    def _calculate_extraction_confidence(self, extracted: Dict, ocr_confidence: float) -> Dict[str, float]:
        """Calculate confidence scores for extraction"""
        # Count filled fields
        total_fields = 0
        filled_fields = 0
        
        for section_name, section_data in extracted.items():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if not key.endswith('_matched') and not key.endswith('_confidence'):
                        total_fields += 1
                        if value is not None and value != '' and value != []:
                            filled_fields += 1
        
        extraction_confidence = filled_fields / total_fields if total_fields > 0 else 0.0
        
        # Overall confidence combines OCR and extraction
        overall = (ocr_confidence * 0.4) + (extraction_confidence * 0.6)
        
        return {
            'ocr': round(ocr_confidence, 3),
            'extraction': round(extraction_confidence, 3),
            'overall': round(overall, 3),
            'fields_total': total_fields,
            'fields_extracted': filled_fields
        }

    def _count_extracted_fields(self, extracted: Dict) -> int:
        """Count number of successfully extracted fields"""
        count = 0
        for section_data in extracted.values():
            if isinstance(section_data, dict):
                for key, value in section_data.items():
                    if value is not None and value != '' and value != []:
                        count += 1
        return count

    def _calculate_completeness(self, extracted: Dict) -> float:
        """Calculate extraction completeness percentage"""
        critical_fields = [
            extracted.get('insured', {}).get('policy_number'),
            extracted.get('insured', {}).get('claimant_name'),
            extracted.get('hospitalization', {}).get('hospital_name'),
            extracted.get('billing', {}).get('total_billed'),
            extracted.get('medical', {}).get('diagnosis_text') or extracted.get('medical', {}).get('icd_codes'),
        ]
        
        filled = sum(1 for f in critical_fields if f)
        return round(filled / len(critical_fields), 2)

    def validate_and_correct(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted fields and apply corrections"""
        fields = extracted_data.get('fields', {})
        corrections = []

        # Validate dates
        for date_field in ['date_of_birth', 'date_of_service']:
            if date_field in fields and fields[date_field].get('value'):
                try:
                    value = fields[date_field]['value']
                    if DATEUTIL_AVAILABLE and date_parser:
                        parsed = date_parser.parse(value)
                        # Check reasonable date range
                        if parsed.year < 1900 or parsed.year > 2030:
                            fields[date_field]['validation_status'] = 'invalid_date_range'
                            corrections.append(f"{date_field}: date out of range")
                except Exception:
                    fields[date_field]['validation_status'] = 'parse_error'

        # Validate amounts
        if 'billed_amount' in fields and fields['billed_amount'].get('value'):
            try:
                amount = float(fields['billed_amount']['value'].replace(',', ''))
                if amount <= 0 or amount > 10000000:
                    fields['billed_amount']['validation_status'] = 'suspicious_amount'
                    corrections.append("billed_amount: suspicious value")
            except ValueError:
                fields['billed_amount']['validation_status'] = 'parse_error'

        # Validate NPI (must be 10 digits)
        if 'npi_number' in fields and fields['npi_number'].get('value'):
            npi = fields['npi_number']['value']
            if not re.match(r'^\d{10}$', npi):
                fields['npi_number']['validation_status'] = 'invalid_format'
                corrections.append("npi_number: invalid format")

        # Validate ICD codes
        if 'diagnosis' in fields and fields['diagnosis'].get('value'):
            icd = fields['diagnosis']['value']
            if icd and icd[0].upper() not in self.icd_codes:
                fields['diagnosis']['validation_status'] = 'invalid_icd_prefix'
                corrections.append("diagnosis: invalid ICD prefix")

        extracted_data['fields'] = fields
        extracted_data['corrections'] = corrections
        extracted_data['validation_complete'] = True

        return extracted_data


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def extract_claim_data(pdf_path: str, use_ml: bool = False) -> Dict[str, Any]:
    """
    Main entry point for claim extraction
    
    Args:
        pdf_path: Path to PDF file
        use_ml: Whether to use ML models (spaCy NER)
    
    Returns:
        Dictionary with extracted and validated fields
    """
    extractor = EnhancedClaimExtractor(use_ml_models=use_ml)
    result = extractor.extract_all_fields(pdf_path)
    if result['success']:
        result = extractor.validate_and_correct(result)
    return result


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        result = extract_claim_data(pdf_file)
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Usage: python extractor_enhanced.py <pdf_path>")
