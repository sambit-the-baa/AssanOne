# agent/extractors/medical_code_extractor.py
"""
Medical Code Extractor for ICD-10, CPT, and other medical codes
Accurately extracts and validates medical codes from claim documents
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class CodeType(Enum):
    """Types of medical codes"""
    ICD10_DIAGNOSIS = "ICD-10-CM Diagnosis"
    ICD10_PROCEDURE = "ICD-10-PCS Procedure"
    CPT = "CPT Procedure"
    HCPCS = "HCPCS"
    NDC = "NDC Drug Code"
    UNKNOWN = "Unknown"


@dataclass
class MedicalCode:
    """Represents an extracted medical code"""
    code: str
    code_type: CodeType
    description: str = ""
    category: str = ""
    confidence: float = 0.0
    context: str = ""  # Surrounding text where code was found
    is_primary: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'code': self.code,
            'code_type': self.code_type.value,
            'description': self.description,
            'category': self.category,
            'confidence': self.confidence,
            'context': self.context[:100] if self.context else "",
            'is_primary': self.is_primary
        }


@dataclass
class MedicalCodeSummary:
    """Summary of all extracted medical codes"""
    diagnosis_codes: List[MedicalCode] = field(default_factory=list)
    procedure_codes: List[MedicalCode] = field(default_factory=list)
    other_codes: List[MedicalCode] = field(default_factory=list)
    primary_diagnosis: Optional[MedicalCode] = None
    extraction_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'diagnosis_codes': [c.to_dict() for c in self.diagnosis_codes],
            'procedure_codes': [c.to_dict() for c in self.procedure_codes],
            'other_codes': [c.to_dict() for c in self.other_codes],
            'primary_diagnosis': self.primary_diagnosis.to_dict() if self.primary_diagnosis else None,
            'total_codes': len(self.diagnosis_codes) + len(self.procedure_codes) + len(self.other_codes),
            'extraction_confidence': self.extraction_confidence
        }


class MedicalCodeExtractor:
    """
    Extracts and validates medical codes from claim text
    Supports ICD-10-CM, ICD-10-PCS, CPT, HCPCS codes
    """
    
    # ICD-10-CM Diagnosis Code Categories (first letter)
    ICD10_CATEGORIES = {
        'A': 'Infectious and parasitic diseases',
        'B': 'Infectious and parasitic diseases',
        'C': 'Neoplasms (Cancer)',
        'D': 'Blood diseases / Neoplasms',
        'E': 'Endocrine, nutritional, metabolic diseases',
        'F': 'Mental and behavioral disorders',
        'G': 'Nervous system diseases',
        'H': 'Eye and ear diseases',
        'I': 'Circulatory system diseases',
        'J': 'Respiratory system diseases',
        'K': 'Digestive system diseases',
        'L': 'Skin and subcutaneous tissue diseases',
        'M': 'Musculoskeletal system diseases',
        'N': 'Genitourinary system diseases',
        'O': 'Pregnancy, childbirth, puerperium',
        'P': 'Perinatal conditions',
        'Q': 'Congenital malformations',
        'R': 'Symptoms and abnormal findings',
        'S': 'Injuries - single body region',
        'T': 'Injuries - multiple/unspecified',
        'U': 'Special purpose codes',
        'V': 'External causes - transport',
        'W': 'External causes - other',
        'X': 'External causes - other',
        'Y': 'External causes - other',
        'Z': 'Factors influencing health status',
    }
    
    # Common ICD-10 codes with descriptions (Indian healthcare context)
    COMMON_ICD10_CODES = {
        # Infectious diseases
        'A09': 'Infectious gastroenteritis and colitis',
        'A15': 'Respiratory tuberculosis',
        'A16': 'Respiratory tuberculosis (without bacteriological confirmation)',
        'A90': 'Dengue fever',
        'A91': 'Dengue hemorrhagic fever',
        'B15': 'Acute hepatitis A',
        'B16': 'Acute hepatitis B',
        'B17': 'Other acute viral hepatitis',
        'B20': 'HIV disease',
        'B50': 'Plasmodium falciparum malaria',
        'B51': 'Plasmodium vivax malaria',
        'B54': 'Unspecified malaria',
        
        # Diabetes and metabolic
        'E10': 'Type 1 diabetes mellitus',
        'E11': 'Type 2 diabetes mellitus',
        'E13': 'Other specified diabetes mellitus',
        'E66': 'Overweight and obesity',
        'E78': 'Disorders of lipoprotein metabolism',
        
        # Cardiovascular
        'I10': 'Essential (primary) hypertension',
        'I11': 'Hypertensive heart disease',
        'I20': 'Angina pectoris',
        'I21': 'Acute myocardial infarction',
        'I25': 'Chronic ischemic heart disease',
        'I48': 'Atrial fibrillation and flutter',
        'I50': 'Heart failure',
        'I63': 'Cerebral infarction (Stroke)',
        'I64': 'Stroke, not specified',
        
        # Respiratory
        'J06': 'Acute upper respiratory infection',
        'J10': 'Influenza due to identified virus',
        'J11': 'Influenza, virus not identified',
        'J12': 'Viral pneumonia',
        'J15': 'Bacterial pneumonia',
        'J18': 'Pneumonia, unspecified organism',
        'J20': 'Acute bronchitis',
        'J44': 'Chronic obstructive pulmonary disease',
        'J45': 'Asthma',
        'J96': 'Respiratory failure',
        
        # Digestive
        'K20': 'Esophagitis',
        'K21': 'Gastro-esophageal reflux disease',
        'K25': 'Gastric ulcer',
        'K26': 'Duodenal ulcer',
        'K29': 'Gastritis and duodenitis',
        'K35': 'Acute appendicitis',
        'K40': 'Inguinal hernia',
        'K80': 'Cholelithiasis (Gallstones)',
        'K81': 'Cholecystitis',
        'K85': 'Acute pancreatitis',
        
        # Genitourinary
        'N10': 'Acute pyelonephritis',
        'N17': 'Acute kidney failure',
        'N18': 'Chronic kidney disease',
        'N20': 'Calculus of kidney and ureter',
        'N39': 'Urinary tract infection',
        'N40': 'Benign prostatic hyperplasia',
        
        # Pregnancy
        'O00': 'Ectopic pregnancy',
        'O03': 'Spontaneous abortion',
        'O14': 'Pre-eclampsia',
        'O24': 'Diabetes mellitus in pregnancy',
        'O36': 'Maternal care for fetal problems',
        'O42': 'Premature rupture of membranes',
        'O60': 'Preterm labor',
        'O80': 'Single spontaneous delivery',
        'O82': 'Cesarean delivery',
        
        # Injuries
        'S00': 'Superficial injury of head',
        'S06': 'Intracranial injury',
        'S22': 'Fracture of rib(s), sternum, thoracic spine',
        'S32': 'Fracture of lumbar spine and pelvis',
        'S42': 'Fracture of shoulder and upper arm',
        'S52': 'Fracture of forearm',
        'S72': 'Fracture of femur',
        'S82': 'Fracture of lower leg',
        
        # External causes (COVID-19)
        'U07': 'COVID-19',
        'U07.1': 'COVID-19, virus identified',
        'U07.2': 'COVID-19, virus not identified',
        
        # Symptoms
        'R00': 'Abnormalities of heart beat',
        'R05': 'Cough',
        'R06': 'Abnormalities of breathing',
        'R07': 'Pain in throat and chest',
        'R10': 'Abdominal and pelvic pain',
        'R50': 'Fever of other and unknown origin',
        'R51': 'Headache',
        'R53': 'Malaise and fatigue',
        
        # Health status
        'Z00': 'General examination',
        'Z01': 'Other special examinations',
        'Z23': 'Immunization',
        'Z38': 'Liveborn infants',
        'Z51': 'Encounter for other aftercare',
        'Z96': 'Presence of functional implants',
    }
    
    # False positive patterns (things that look like ICD codes but aren't)
    FALSE_POSITIVE_PATTERNS = [
        r'PIN\s*(?:CODE)?:?\s*\d{6}',  # PIN codes
        r'(?:MOBILE|PHONE|TEL|FAX)[\s:]*[A-Z]?\d{2}',  # Phone numbers
        r'(?:PLOT|SHOP|FLAT|HOUSE)\s*(?:NO)?[\s:.]*[A-Z]?\d{2}',  # Address numbers
        r'SECTOR[\s-]*[A-Z]?\d{2}',  # Sector numbers
        r'(?:BLOCK|WING|TOWER)[\s-]*[A-Z]?\d{2}',  # Building blocks
        r'(?:BRANCH|BRN)[\s:]*[A-Z]?\d{2}',  # Branch codes
        r'(?:ACCOUNT|A/C|AC)[\s:]*[A-Z]?\d{2}',  # Account numbers
        r'IFSC[\s:]*[A-Z]{4}\d{7}',  # IFSC codes
        r'PAN[\s:]*[A-Z]{5}\d{4}[A-Z]',  # PAN numbers
        r'(?:PAGE|PG)[\s:]*\d+\s*(?:OF|/)\s*\d+',  # Page numbers
        r'(?:SERIAL|SR|SL)[\s:.]*(?:NO)?[\s:.]*[A-Z]?\d{2}',  # Serial numbers
        r'(?:REG|REGISTRATION)[\s:.]*(?:NO)?[\s:.]*[A-Z]?\d{2}',  # Registration numbers (non-medical)
        r'(?:INVOICE|INV|BILL)[\s:.]*(?:NO)?[\s:.]*[A-Z]?\d{2}',  # Invoice numbers
        r'(?:ROOM|BED)[\s:.]*(?:NO)?[\s:.]*[A-Z]?\d{2}',  # Room/bed numbers
        r'(?:POLICY|CLAIM|TPA)[\s:.]*(?:NO)?[\s:.]*[A-Z]?\d{2}',  # Policy/claim numbers
    ]
    
    # Primary diagnosis indicators
    PRIMARY_INDICATORS = [
        r'(?:PRIMARY|PRINCIPAL|MAIN|CHIEF)\s*(?:DIAGNOSIS|DX|COMPLAINT)',
        r'(?:DIAGNOSIS|DX)\s*(?:CODE)?\s*(?:1|ONE|I)\b',
        r'ADMITTING\s*(?:DIAGNOSIS|DX)',
        r'REASON\s*FOR\s*(?:ADMISSION|HOSPITALIZATION)',
    ]
    
    # Context patterns that indicate valid medical codes
    VALID_CONTEXT_PATTERNS = [
        r'(?:ICD|DIAGNOSIS|DX|DISEASE|CONDITION)',
        r'(?:PROCEDURE|SURGERY|OPERATION)',
        r'(?:MEDICAL|CLINICAL|HOSPITAL)',
        r'(?:TREATMENT|THERAPY)',
        r'(?:PRIMARY|SECONDARY|PRINCIPAL)',
        r'(?:ADMIT|DISCHARGE|FINAL)',
    ]
    
    def __init__(self):
        # Compile patterns
        self.false_positive_re = [re.compile(p, re.IGNORECASE) for p in self.FALSE_POSITIVE_PATTERNS]
        self.primary_re = [re.compile(p, re.IGNORECASE) for p in self.PRIMARY_INDICATORS]
        self.valid_context_re = [re.compile(p, re.IGNORECASE) for p in self.VALID_CONTEXT_PATTERNS]
        
        # ICD-10-CM pattern: Letter + 2-7 characters (digits, letters, dots)
        # Valid: A00, A00.0, A00.00, S72.001A
        self.icd10_pattern = re.compile(
            r'\b([A-TV-Z]\d{2}(?:\.\d{1,4})?[A-Z]?)\b',
            re.IGNORECASE
        )
        
        # ICD-10-PCS pattern: 7 alphanumeric characters
        self.icd10_pcs_pattern = re.compile(
            r'\b([0-9A-HJ-NP-Z]{7})\b',
            re.IGNORECASE
        )
        
        # CPT pattern: 5 digits, optionally with modifier
        self.cpt_pattern = re.compile(
            r'\b(\d{5})(?:\s*-\s*([A-Z0-9]{2}))?\b'
        )
        
        # HCPCS pattern: Letter + 4 digits
        self.hcpcs_pattern = re.compile(
            r'\b([A-V]\d{4})\b',
            re.IGNORECASE
        )
    
    def _is_false_positive(self, code: str, context: str) -> bool:
        """Check if a potential code is actually a false positive"""
        full_text = f"{context} {code}"
        
        for pattern in self.false_positive_re:
            if pattern.search(full_text):
                return True
        
        # Check for common non-medical patterns
        code_upper = code.upper()
        
        # Filter out obvious non-codes
        if code_upper.startswith(('PIN', 'TEL', 'FAX', 'MOB')):
            return True
        
        # Very short codes without medical context are suspect
        if len(code) <= 3:
            has_medical_context = any(p.search(context) for p in self.valid_context_re)
            if not has_medical_context:
                # Allow common ICD codes even without explicit context
                if code_upper[:3] not in self.COMMON_ICD10_CODES:
                    return True
        
        return False
    
    def _get_context(self, text: str, match_start: int, match_end: int, window: int = 50) -> str:
        """Extract context around a match"""
        start = max(0, match_start - window)
        end = min(len(text), match_end + window)
        return text[start:end].strip()
    
    def _is_primary_diagnosis(self, context: str) -> bool:
        """Check if context indicates this is a primary diagnosis"""
        for pattern in self.primary_re:
            if pattern.search(context):
                return True
        return False
    
    def _get_icd10_description(self, code: str) -> Tuple[str, str]:
        """Get description and category for ICD-10 code"""
        code_upper = code.upper().replace('.', '')
        
        # Try exact match first (with 3-char prefix)
        code_prefix = code_upper[:3]
        if code_prefix in self.COMMON_ICD10_CODES:
            return self.COMMON_ICD10_CODES[code_prefix], self.ICD10_CATEGORIES.get(code_upper[0], 'Unknown')
        
        # Try with full code
        if code_upper in self.COMMON_ICD10_CODES:
            return self.COMMON_ICD10_CODES[code_upper], self.ICD10_CATEGORIES.get(code_upper[0], 'Unknown')
        
        # Return category only
        category = self.ICD10_CATEGORIES.get(code_upper[0], 'Unknown category')
        return '', category
    
    def _calculate_confidence(self, code: str, context: str, code_type: CodeType) -> float:
        """Calculate confidence score for extracted code"""
        confidence = 0.5  # Base confidence
        
        # Boost for known codes
        code_upper = code.upper()
        if code_upper[:3] in self.COMMON_ICD10_CODES or code_upper in self.COMMON_ICD10_CODES:
            confidence += 0.3
        
        # Boost for medical context
        if any(p.search(context) for p in self.valid_context_re):
            confidence += 0.15
        
        # Boost for proper format
        if '.' in code and code_type == CodeType.ICD10_DIAGNOSIS:
            confidence += 0.05  # Properly formatted with decimal
        
        # Penalize if context has non-medical terms (more aggressive)
        non_medical_terms = ['address', 'plot', 'shop', 'sector', 'branch', 'account', 'invoice',
                           'phone', 'mobile', 'email', 'website', 'registration', 'receipt',
                           'bill no', 'invoice no', 'reg no', 'indoor no', 'date:', 'time:']
        if any(term in context.lower() for term in non_medical_terms):
            confidence -= 0.4  # Increased penalty from 0.3 to 0.4
        
        # Additional penalty for codes in form headers/metadata context
        form_metadata = ['patient name', 'date of birth', 'policy number', 'claim number',
                        'member id', 'indoor no', 'reg. no', 'admission date', 'discharge date']
        if any(meta in context.lower() for meta in form_metadata):
            confidence -= 0.3
        
        return min(1.0, max(0.0, confidence))
    
    def extract_icd10_codes(self, text: str) -> List[MedicalCode]:
        """Extract ICD-10 diagnosis codes from text"""
        codes = []
        seen = set()
        
        for match in self.icd10_pattern.finditer(text):
            code = match.group(1).upper()
            
            # Skip if already seen
            if code in seen:
                continue
            
            # Get context
            context = self._get_context(text, match.start(), match.end())
            
            # Skip false positives
            if self._is_false_positive(code, context):
                continue
            
            # Get description and category
            description, category = self._get_icd10_description(code)
            
            # Calculate confidence
            confidence = self._calculate_confidence(code, context, CodeType.ICD10_DIAGNOSIS)
            
            # Skip low confidence codes (increased threshold from 0.3 to 0.65 for better accuracy)
            if confidence < 0.65:
                continue
            
            # Check if primary
            is_primary = self._is_primary_diagnosis(context)
            
            codes.append(MedicalCode(
                code=code,
                code_type=CodeType.ICD10_DIAGNOSIS,
                description=description,
                category=category,
                confidence=confidence,
                context=context,
                is_primary=is_primary
            ))
            
            seen.add(code)
        
        # Sort by confidence (descending) and primary status
        codes.sort(key=lambda x: (x.is_primary, x.confidence), reverse=True)
        
        return codes
    
    def extract_cpt_codes(self, text: str) -> List[MedicalCode]:
        """Extract CPT procedure codes from text"""
        codes = []
        seen = set()
        
        # Common CPT code ranges
        # 00100-01999: Anesthesia
        # 10021-69990: Surgery
        # 70010-79999: Radiology
        # 80047-89398: Pathology/Lab
        # 90281-99607: Medicine
        
        for match in self.cpt_pattern.finditer(text):
            code = match.group(1)
            modifier = match.group(2) if match.group(2) else ''
            
            # Skip if already seen
            if code in seen:
                continue
            
            # Validate CPT range (must be in valid ranges)
            code_int = int(code)
            valid_ranges = [
                (100, 1999),     # Anesthesia
                (10021, 69990),  # Surgery
                (70010, 79999),  # Radiology
                (80047, 89398),  # Pathology/Lab
                (90281, 99607),  # Medicine
            ]
            
            is_valid_cpt = any(start <= code_int <= end for start, end in valid_ranges)
            if not is_valid_cpt:
                continue
            
            context = self._get_context(text, match.start(), match.end())
            
            # Skip false positives
            if self._is_false_positive(code, context):
                continue
            
            confidence = self._calculate_confidence(code, context, CodeType.CPT)
            
            full_code = f"{code}-{modifier}" if modifier else code
            
            codes.append(MedicalCode(
                code=full_code,
                code_type=CodeType.CPT,
                description='',
                category='Procedure',
                confidence=confidence,
                context=context,
                is_primary=False
            ))
            
            seen.add(code)
        
        return codes
    
    def extract_all_codes(self, text: str) -> MedicalCodeSummary:
        """Extract all medical codes from text"""
        if not text or not text.strip():
            return MedicalCodeSummary()
        
        # Extract different code types
        diagnosis_codes = self.extract_icd10_codes(text)
        procedure_codes = self.extract_cpt_codes(text)
        
        # Determine primary diagnosis
        primary = None
        for code in diagnosis_codes:
            if code.is_primary:
                primary = code
                break
        
        # If no explicit primary, use highest confidence
        if not primary and diagnosis_codes:
            primary = diagnosis_codes[0]
        
        # Calculate overall confidence
        all_codes = diagnosis_codes + procedure_codes
        if all_codes:
            avg_confidence = sum(c.confidence for c in all_codes) / len(all_codes)
        else:
            avg_confidence = 0.0
        
        return MedicalCodeSummary(
            diagnosis_codes=diagnosis_codes,
            procedure_codes=procedure_codes,
            other_codes=[],
            primary_diagnosis=primary,
            extraction_confidence=avg_confidence
        )
    
    def format_codes_for_display(self, summary: MedicalCodeSummary) -> Dict[str, Any]:
        """Format extracted codes for display in dashboard"""
        result = {
            'diagnosis': [],
            'procedures': [],
            'primary_diagnosis': None,
            'icd_codes_str': '',
            'summary_text': ''
        }
        
        # Format diagnosis codes
        for code in summary.diagnosis_codes:
            entry = {
                'code': code.code,
                'description': code.description or code.category,
                'confidence': f"{code.confidence:.0%}",
                'is_primary': code.is_primary
            }
            result['diagnosis'].append(entry)
        
        # Format procedure codes
        for code in summary.procedure_codes:
            entry = {
                'code': code.code,
                'type': code.code_type.value,
                'confidence': f"{code.confidence:.0%}"
            }
            result['procedures'].append(entry)
        
        # Primary diagnosis
        if summary.primary_diagnosis:
            pd = summary.primary_diagnosis
            result['primary_diagnosis'] = f"{pd.code} - {pd.description or pd.category}"
        
        # ICD codes string (for backward compatibility)
        if summary.diagnosis_codes:
            result['icd_codes_str'] = ', '.join(c.code for c in summary.diagnosis_codes[:5])
        
        # Summary text
        diag_count = len(summary.diagnosis_codes)
        proc_count = len(summary.procedure_codes)
        result['summary_text'] = f"{diag_count} diagnosis code(s), {proc_count} procedure code(s)"
        
        return result


def extract_medical_codes(text: str) -> MedicalCodeSummary:
    """Convenience function to extract all medical codes"""
    extractor = MedicalCodeExtractor()
    return extractor.extract_all_codes(text)


# Test function
if __name__ == "__main__":
    test_text = """
    CLAIM FORM - PART D: MEDICAL DETAILS
    
    Primary Diagnosis: Acute Myocardial Infarction
    ICD-10 Code: I21.0
    
    Secondary Diagnosis:
    1. Type 2 Diabetes Mellitus (E11.9)
    2. Essential Hypertension - I10
    3. Chronic Kidney Disease Stage 3 (N18.3)
    
    Procedures Performed:
    - Coronary Angiography (CPT: 93458)
    - Percutaneous Coronary Intervention (93454)
    
    Patient Address: SHOP NO 10, PLOT NO 20, SECTOR-6
    PIN CODE: 400706
    
    Hospital Registration No: H12345
    """
    
    extractor = MedicalCodeExtractor()
    summary = extractor.extract_all_codes(test_text)
    
    print("=" * 60)
    print("MEDICAL CODE EXTRACTION TEST")
    print("=" * 60)
    
    print(f"\nDiagnosis Codes ({len(summary.diagnosis_codes)}):")
    for code in summary.diagnosis_codes:
        primary_marker = " [PRIMARY]" if code.is_primary else ""
        print(f"  - {code.code}: {code.description or code.category} ({code.confidence:.0%}){primary_marker}")
    
    print(f"\nProcedure Codes ({len(summary.procedure_codes)}):")
    for code in summary.procedure_codes:
        print(f"  - {code.code}: {code.code_type.value} ({code.confidence:.0%})")
    
    print(f"\nPrimary Diagnosis: {summary.primary_diagnosis.code if summary.primary_diagnosis else 'None'}")
    print(f"Overall Confidence: {summary.extraction_confidence:.0%}")
    
    # Test formatting
    formatted = extractor.format_codes_for_display(summary)
    print(f"\nFormatted ICD String: {formatted['icd_codes_str']}")
    print(f"Summary: {formatted['summary_text']}")
