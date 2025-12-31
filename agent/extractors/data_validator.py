# agent/extractors/data_validator.py
"""
Data Validation & Cleaning for Insurance Claims
Filters out OCR garbage, validates amounts, and cleans extracted data
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum


class ValidationLevel(Enum):
    """Validation strictness levels"""
    STRICT = "strict"      # Reject anything suspicious
    MODERATE = "moderate"  # Allow some uncertainty
    LENIENT = "lenient"    # Accept most data


@dataclass
class ValidationResult:
    """Result of validation check"""
    is_valid: bool
    original_value: Any
    cleaned_value: Any = None
    issues: List[str] = field(default_factory=list)
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'original_value': str(self.original_value)[:100] if self.original_value else None,
            'cleaned_value': str(self.cleaned_value)[:100] if self.cleaned_value else None,
            'issues': self.issues,
            'confidence': self.confidence
        }


class DataValidator:
    """
    Validates and cleans extracted claim data
    Filters OCR garbage, validates amounts, dates, and text fields
    """
    
    # Gibberish detection patterns - text that looks like OCR errors
    GIBBERISH_PATTERNS = [
        r'[A-Z]{2,}[a-z]{1,2}[A-Z]{2,}',  # Mixed case gibberish like "TDIHiVIA"
        r'[bcdfghjklmnpqrstvwxyz]{5,}',    # 5+ consonants in a row (case insensitive)
        r'[BCDFGHJKLMNPQRSTVWXYZ]{5,}',    # 5+ uppercase consonants
        r'\b[A-Z][a-z][A-Z][a-z][A-Z]',    # Alternating case pattern
        r'[|!]{2,}',                        # Multiple pipes/exclamation (OCR artifacts)
        r'[\[\]{}]{2,}',                    # Multiple brackets
        r'[0-9][A-Za-z][0-9][A-Za-z]',     # Alternating digit-letter pattern
    ]
    
    # Valid text patterns - things that should NOT be flagged as gibberish
    VALID_TEXT_PATTERNS = [
        r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+',  # Names with titles
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+',               # Normal names
        r'\b(?:Room|ICU|OT|Ward|Bed)\s*(?:No\.?|#)?\s*\d+',  # Room numbers
        r'(?:ICD|CPT|HCPCS)[-\s]*\d+',                # Medical codes
        r'\b[A-Z]\d{2}(?:\.\d+)?',                    # ICD-10 codes
    ]
    
    # Amount validation - reasonable ranges for Indian healthcare
    AMOUNT_RANGES = {
        'room_charge': (100, 50000),          # Per day room charge
        'icu_charge': (1000, 200000),         # Per day ICU
        'surgery': (5000, 2000000),           # Surgery costs
        'medicine': (10, 500000),             # Medicines
        'consultation': (100, 10000),         # Doctor fees
        'investigation': (50, 100000),        # Lab/radiology
        'total_bill': (500, 10000000),        # Total bill (up to 1 crore)
        'line_item': (1, 1000000),            # Individual line items
    }
    
    # False positive amount patterns - things that look like amounts but aren't
    # These patterns check the CONTEXT around the number, not the number itself
    FALSE_AMOUNT_CONTEXT_PATTERNS = [
        r'PIN[\s:]*(?:CODE)?[\s:]*$',          # PIN code context (at end)
        r'(?:MOBILE|PHONE|TEL|FAX)[\s:]*$',   # Phone context (at end)
        r'(?:A/C|Account|Acc)[\s:.]*(?:No\.?)?[\s:]*$',  # Account context
        r'(?:Policy|Member)[\s:.]*(?:No\.?|Number)?[\s:]*$',  # Policy/Member number
        r'(?:Indoor|IP|Reg|Registration)[\s:.]*(?:No\.?)?[\s:]*$',  # Registration context
        r'(?:Sr|Sl|Serial)[\s:.]*(?:No\.?)?[\s:]*$',  # Serial number context
        r'(?:Page|Pg)[\s:]*$',                 # Page context
        r'IFSC[\s:]*$',                        # IFSC context
    ]
    
    # Minimum word length for valid text (to filter OCR fragments)
    MIN_WORD_LENGTH = 2
    MIN_VALID_WORDS_RATIO = 0.5  # At least 50% of words should be valid
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        self.level = level
        self.gibberish_re = [re.compile(p, re.IGNORECASE) for p in self.GIBBERISH_PATTERNS]
        self.valid_text_re = [re.compile(p, re.IGNORECASE) for p in self.VALID_TEXT_PATTERNS]
        self.false_amount_context_re = [re.compile(p, re.IGNORECASE) for p in self.FALSE_AMOUNT_CONTEXT_PATTERNS]
    
    def is_gibberish(self, text: str) -> Tuple[bool, float]:
        """
        Check if text is likely OCR gibberish
        Returns (is_gibberish, confidence)
        """
        if not text or len(text.strip()) < 3:
            return False, 0.0
        
        text = str(text).strip()
        gibberish_score = 0.0
        
        # Check 1: Consonant clusters (4+ consonants without vowels)
        consonant_clusters = re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{4,}', text)
        if consonant_clusters:
            gibberish_score += 1.5
        
        # Check 2: Mixed case gibberish patterns (like yEomate, ChequetD0, eAge)
        # Lowercase followed by uppercase in same word
        mixed_case_words = re.findall(r'\b\w*[a-z][A-Z]\w*\b', text)
        if len(mixed_case_words) >= 2:  # Multiple mixed case words is very suspicious
            gibberish_score += 2.5
        elif mixed_case_words:
            gibberish_score += 1.5
        
        # Check 3: Digit-letter mixing in words (like D0, l1, O0)
        digit_letter_mix = re.findall(r'\b[A-Za-z]+[0-9][A-Za-z]*\b|\b[A-Za-z]*[0-9][A-Za-z]+\b', text)
        if digit_letter_mix:
            gibberish_score += 1.0
        
        # Check 4: Very low vowel ratio
        letters = re.findall(r'[A-Za-z]', text)
        if len(letters) >= 5:
            vowels = re.findall(r'[AEIOUaeiou]', text)
            vowel_ratio = len(vowels) / len(letters)
            if vowel_ratio < 0.15:  # Very low vowels
                gibberish_score += 1.5
            elif vowel_ratio < 0.22:  # Low vowels
                gibberish_score += 0.75
        
        # Check 5: Word validity - check each word (now stricter)
        words = [w.strip('.,!?;:()[]{}') for w in text.split() if len(w.strip('.,!?;:()[]{}')) >= 4]
        if words:
            invalid_words = 0
            for word in words:
                if not self._is_valid_word(word):
                    invalid_words += 1
            invalid_ratio = invalid_words / len(words) if words else 0
            if invalid_ratio > 0.4:  # More than 40% invalid words
                gibberish_score += 2.0  # Stronger signal
            elif invalid_ratio > 0.2:  # More than 20% invalid words
                gibberish_score += 1.0
        
        # Check 6: ALL CAPS nonsense words (like TDIHIVIA, IRVINIAITIM)
        all_caps_words = re.findall(r'\b[A-Z]{5,}\b', text)
        valid_acronyms = {'HOSPITAL', 'MEDICAL', 'PATIENT', 'HEALTH', 'INSURANCE',
                        'CLAIM', 'DOCTOR', 'MEDICINE', 'SURGERY', 'TOTAL', 'AMOUNT',
                        'INDIA', 'DELHI', 'MUMBAI', 'KOLKATA', 'CHENNAI', 'BANGALORE',
                        'BLOOD', 'URINE', 'LIPID', 'KIDNEY', 'LIVER', 'HEART',
                        'HDFC', 'ICICI', 'AXIS', 'STATE', 'UNION', 'CENTRAL', 
                        'STAIN', 'ACCOUNT', 'DETAILS', 'BRANCH', 'AADHAAR'}
        caps_nonsense = 0
        for caps_word in all_caps_words:
            if caps_word in valid_acronyms:
                continue
            # Check for unpronounceable patterns
            if re.search(r'[BCDFGHJKLMNPQRSTVWXYZ]{3,}', caps_word):
                caps_nonsense += 1
            # Check for repeated vowels or unusual vowel patterns
            elif re.search(r'([AEIOU])\1|I[AI]I|AI[AI]', caps_word):
                caps_nonsense += 1
            # Long words that aren't in valid set
            elif len(caps_word) > 7:
                caps_nonsense += 1
        if caps_nonsense >= 2:
            gibberish_score += 2.0
        elif caps_nonsense >= 1:
            gibberish_score += 1.0
        
        # Check 7: Known OCR garbage patterns
        ocr_garbage = [
            r'[Il1|]{3,}',  # III, lll, |||, 111 mixed
            r'[O0]{2,}',    # OO, 00
            r'\bt[DT][IiLl1]',  # tDI, TDI type patterns
            r'[A-Z]{2,}[a-z][A-Z]{2,}',  # Mixed case blocks
        ]
        for pattern in ocr_garbage:
            if re.search(pattern, text):
                gibberish_score += 1.0
                break
        
        # Check 8: Unusual punctuation patterns (common in OCR errors)
        if re.search(r'[.]{3,}|[,]{2,}|[;:]{2,}', text):
            gibberish_score += 0.5
        
        # Check 9: Short all-lowercase words that look like OCR fragments
        short_nonsense = re.findall(r'\b[a-z]{2,4}\b', text)
        valid_short = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 
                      'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day',
                      'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new',
                      'now', 'old', 'see', 'way', 'who', 'boy', 'did', 'own',
                      'say', 'she', 'too', 'use', 'room', 'bed', 'icu', 'ward',
                      'dr', 'mr', 'mrs', 'ms', 'no', 'rs', 'per', 'day', 'qty',
                      'amt', 'date', 'name', 'age', 'sex', 'dob', 'mob', 'tel'}
        if short_nonsense:
            invalid_short = sum(1 for w in short_nonsense if w not in valid_short)
            if invalid_short / max(len(short_nonsense), 1) > 0.6:
                gibberish_score += 0.75
        
        # Check 10: 3-letter all caps that aren't acronyms (like RAG, QWE, UIO, XYZ)
        three_letter_caps = re.findall(r'\b[A-Z]{3}\b', text)
        valid_three = {'ICU', 'OPD', 'IPD', 'MRI', 'ECG', 'EKG', 'HIV', 'BMI', 'DNA', 'RNA',
                      'TPA', 'MLC', 'DOB', 'DOA', 'DOD', 'GST', 'PAN', 'PIN', 'UPI', 'EMI',
                      'ENT', 'OPD', 'ICU', 'OT', 'MRD', 'BPL', 'APL', 'SSC', 'HSC', 'PDF',
                      'RAG', 'THE', 'AND', 'FOR', 'ARE', 'NOT', 'BUT', 'YOU', 'ALL'}
        invalid_three = [w for w in three_letter_caps if w not in valid_three]
        if len(invalid_three) >= 2:  # 2+ invalid 3-letter caps is suspicious
            gibberish_score += 2.0
        
        # Calculate if gibberish (threshold of 2.0)
        is_gibberish = gibberish_score >= 2.0
        confidence = min(1.0, gibberish_score / 3.0)
        
        return is_gibberish, confidence
    
    def _is_valid_word(self, word: str) -> bool:
        """Check if a word is valid (not gibberish)"""
        word = word.strip('.,!?;:()[]{}')
        
        if len(word) < self.MIN_WORD_LENGTH:
            return True  # Short words are ok
        
        # Pure numbers are valid
        if word.replace(',', '').replace('.', '').isdigit():
            return True
        
        # Extended dictionary of common valid words
        valid_words = {
            # Medical terms
            'room', 'bed', 'ward', 'icu', 'ot', 'charges', 'total', 'amount',
            'medicine', 'pharmacy', 'drug', 'tablet', 'injection', 'dose',
            'consultation', 'doctor', 'surgeon', 'nursing', 'care', 'nurse',
            'lab', 'test', 'xray', 'scan', 'report', 'discharge', 'admit',
            'admission', 'patient', 'hospital', 'clinic', 'insurance', 'treatment',
            'claim', 'policy', 'member', 'date', 'birth', 'name', 'diagnosis',
            'address', 'phone', 'mobile', 'email', 'signature', 'blood', 'urine',
            'surgery', 'operation', 'procedure', 'anesthesia', 'oxygen', 'dialysis',
            # Common English
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
            'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how',
            'its', 'may', 'new', 'now', 'old', 'see', 'way', 'who', 'boy', 'did',
            'own', 'say', 'she', 'too', 'use', 'this', 'that', 'with', 'have',
            'from', 'they', 'been', 'said', 'each', 'what', 'when', 'will', 'more',
            'write', 'like', 'time', 'very', 'after', 'words', 'called', 'just',
            'where', 'most', 'know', 'take', 'people', 'into', 'year', 'your',
            'good', 'some', 'could', 'them', 'make', 'than', 'first', 'water',
            'been', 'call', 'made', 'find', 'long', 'down', 'side', 'been',
            # Indian names/places (common)
            'aadhaar', 'aadhar', 'india', 'delhi', 'mumbai', 'kolkata', 'chennai',
            'bangalore', 'hyderabad', 'pune', 'jaipur', 'kumar', 'singh', 'sharma',
            'gupta', 'patel', 'khan', 'verma', 'jain', 'mehta', 'shah', 'bhatt',
            # Banking/finance
            'hdfc', 'icici', 'axis', 'bank', 'account', 'branch', 'ifsc', 'cheque',
            'payment', 'amount', 'rupees', 'neft', 'rtgs', 'imps', 'upi',
            # Numbers as words
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        }
        
        word_lower = word.lower()
        if word_lower in valid_words:
            return True
        
        # Check consonant clusters - too many consonants = gibberish
        consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]', word_lower))
        vowels = len(re.findall(r'[aeiou]', word_lower))
        
        if len(word) > 3:
            # No vowels at all = gibberish
            if vowels == 0 and consonants > 2:
                return False
            # Very high consonant ratio = gibberish
            if consonants > 0 and consonants / len(word) > 0.85:
                return False
            # Multiple consonant clusters = gibberish
            clusters = re.findall(r'[bcdfghjklmnpqrstvwxyz]{3,}', word_lower)
            if len(clusters) >= 2:
                return False
        
        # For ALL CAPS words longer than 5 chars, be more suspicious
        if word.isupper() and len(word) > 5:
            # Check if it has repeating vowel patterns (like IRVINIAITIM)
            vowel_seq = ''.join(re.findall(r'[AEIOU]', word))
            if len(vowel_seq) >= 4:
                # Check for unusual patterns - real words don't typically have IIAI, AIIE patterns
                if re.search(r'([AEIOU])\1', vowel_seq):  # Repeated vowels
                    return False
                if re.search(r'I[AI]I|AI[AI]|[AE]I[AE]', vowel_seq):  # Strange I-heavy patterns
                    return False
        
        return True
    
    def validate_amount(self, value: Any, context: str = "", amount_type: str = "line_item") -> ValidationResult:
        """
        Validate if a value is a legitimate billing amount
        """
        issues = []
        
        # Convert to float
        try:
            if isinstance(value, str):
                # Remove currency symbols and commas
                cleaned = re.sub(r'[₹$,\s]', '', value)
                amount = float(cleaned)
            else:
                amount = float(value)
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                original_value=value,
                issues=['Cannot parse as number'],
                confidence=0.0
            )
        
        # Check for false positives using context keywords
        context_lower = context.lower()
        for pattern in self.false_amount_context_re:
            if pattern.search(context_lower):
                issues.append(f'Context suggests non-amount ({context[:30]})')
                return ValidationResult(
                    is_valid=False,
                    original_value=value,
                    issues=issues,
                    confidence=0.2
                )
        
        # Check reasonable range
        min_val, max_val = self.AMOUNT_RANGES.get(amount_type, (1, 1000000))
        
        if amount < min_val:
            issues.append(f'Amount too low ({amount} < {min_val})')
        elif amount > max_val:
            issues.append(f'Amount too high ({amount} > {max_val})')
        
        # Check for suspicious patterns in the number itself
        amount_str = str(int(amount))
        
        # 6-digit numbers are often PIN codes (in India, PIN codes are 6 digits starting with 1-8)
        # But NOT if they look like round amounts (e.g., 150000, 200000)
        if len(amount_str) == 6 and amount == int(amount):  # Exact 6 digits, no decimals
            if amount_str[0] in '12345678':
                # Round amounts like 100000, 150000, 200000 are valid billing amounts
                # PIN codes don't typically end with multiple zeros
                if not (amount % 10000 == 0 or amount % 5000 == 0):
                    # Check if context suggests it's actually a billing amount
                    billing_context = ['charge', 'amount', 'total', 'bill', 'fee', 'cost', 
                                      'surgery', 'medicine', 'room', 'icu', 'ot']
                    if not any(bc in context.lower() for bc in billing_context):
                        issues.append('Looks like PIN code (6 digits)')
        
        # 8-digit numbers starting with date patterns
        if len(amount_str) == 8:
            # DDMMYYYY or YYYYMMDD patterns
            if (amount_str[:2] in [f'{d:02d}' for d in range(1, 32)] and 
                amount_str[2:4] in [f'{m:02d}' for m in range(1, 13)]):
                issues.append('Looks like date (DDMMYYYY)')
            elif (amount_str[:4] in [str(y) for y in range(2020, 2030)] and
                  amount_str[4:6] in [f'{m:02d}' for m in range(1, 13)]):
                issues.append('Looks like date (YYYYMMDD)')
        
        # 10-digit numbers are phone numbers
        if len(amount_str) == 10 and amount_str[0] in '6789':
            issues.append('Looks like phone number')
        
        # Very large round numbers might be ID numbers (but not if they're billing amounts)
        # This check is only for numbers > 1 million that look suspiciously like IDs
        if amount > 1000000 and amount == int(amount):
            # Check if it's suspiciously round - only applies to very large numbers
            if str(int(amount)).count('0') >= 5:  # 5+ zeros is suspicious
                issues.append('Suspiciously round number')
        
        is_valid = len(issues) == 0
        confidence = 1.0 - (len(issues) * 0.25)
        
        return ValidationResult(
            is_valid=is_valid,
            original_value=value,
            cleaned_value=amount if is_valid else None,
            issues=issues,
            confidence=max(0.0, confidence)
        )
    
    def validate_billing_item(self, description: str, amount: float, context: str = "") -> ValidationResult:
        """
        Validate a complete billing line item (description + amount)
        """
        issues = []
        
        # First: Check for obvious non-billing descriptions
        desc_lower = (description or "").lower().strip()
        
        # Reject descriptions that are clearly metadata, not billing items
        metadata_patterns = [
            r'^reg\.?\s*no\.?',                # Registration number
            r'^r\.?\s*no\.?',                   # R. No
            r'regn\.?\s*no',                   # Regn. No (common OCR garbage)
            r'^[1il]\.?\s*p\.?\s*d\.?',        # I.P.D./1.P.D. number (OCR reads I as 1)
            r'^indoor\s*no',                   # Indoor number
            r'^dt\.?\s*of\s*(admission|discharge)',  # Date fields
            r'^d\.?o\.?a\.?[\s\-]',            # DOA
            r'^d\.?o\.?d\.?[\s\-]',            # DOD  
            r'^date\s*:?\s*\d',                # Date: 10/02/...
            r'^net\s*amt',                     # Net Amt (subtotal marker)
            r'^\d{4,}\s*-',                    # Number followed by dash (like "82682-")
            r'^[a-z]\s*reg',                   # "i Reg. No", "b- Reg. No"
            r'^=+\s*reg',                      # "=e Reg. No"
            r'^[b\-]+\s*reg',                  # "b- Reg. No"
            r'civil\s*surgeon',               # Civil Surgeon (authority, not service)
            r'nursing\s*homes?\s*registration\s*act',  # Legal registration act
            r'bombay\s*nursing',              # Bombay Nursing Homes Act
            r'bombay.*act',                   # Bombay ... Act
            r'^\d+\s*in\s*\d+',               # "34 IN 30044" pattern
            r'satara|vaduj',                  # Location names as descriptions
            r'^\@\s*\d',                       # @ 0-231333
            r'indoor\s*no',                   # Indoor No.
            r'[1il]\.?\s*p\.?\s*d\.?\s*no',   # I.P.D. NO. (anywhere in text)
            r'p\.?\s*d\.?\s*no\.?\s*[\-\.]',  # P.D. NO. - pattern
            # Additional metadata patterns
            r'iso\s*certified',               # ISO Certified header
            r'@\w+\.\w+',                     # Email addresses
            r'@gmail|@yahoo|@hotmail',        # Email domains
            r'^bill\s*no\b',                  # Bill No. (metadata)
            r'^receipt\s*no\b',               # Receipt No. (metadata)
            r'^amount\s*:\s*rs',              # Amount : Rs (header)
            r'hospital\d*@',                  # Hospital email patterns
            r'\d{5,}\s*@',                    # Numbers followed by @
            r'^total\s*:?\s*$',               # Just "Total:" without amount
            r'^sub\s*total',                  # Sub total header
            r'^grand\s*total',                # Grand total header
            r'page\s*\d+\s*of\s*\d+',         # Page X of Y
            r'^sr\.?\s*no\.?',                # Serial number
            r'^sl\.?\s*no\.?',                # Sl. No.
            r'^particulars?\s*$',             # Particulars header
            r'^amount\s*$',                   # Amount header
            r'^rate\s*$',                     # Rate header
            r'^qty\s*$',                      # Qty header
            r'basilhospital',                 # Specific OCR garbage
            r'^\d+\.\s*$',                    # Just "1." or "2." (numbered list markers)
            r'^[a-z]\s*\d{5,}',               # Letter followed by long number
        ]
        
        for pattern in metadata_patterns:
            if re.search(pattern, desc_lower):
                issues.append(f'Description is metadata, not a billing item')
                break
        
        # Check for OCR garbage patterns in description
        if not issues:
            ocr_garbage_patterns = [
                r'[&+]{2,}',                   # Multiple & or +
                r'\\[a-z]',                    # Escaped characters like \g
                r'[\ufffd\u00ef\u00bf\u00bd]', # Unicode replacement characters
                r'zippja|gicbiz|ccameal|shbeses',  # Known OCR garbage words
                r'\b[a-z]\d+[a-z]\d+',         # Alternating letter-digit patterns
                r'fo>aee|fates',               # More OCR garbage
                r'hegunation',                 # Misspelled "registration"
                r'NMEDE|MEDENI',               # OCR errors for "NEEDLE"
                r'NONOCEF',                    # OCR error for "MONOCEF"
                r'=O\s+SPIN',                  # "=O SPIN" pattern
                r'SONS\s+\d',                  # "SONS 03/28"
                r'^=e\s+',                     # "=e Reg. No"
                r'semen\s*analyser',           # Likely OCR error (sensitive test)
                r'NEDMOL',                     # OCR error
                # More OCR garbage patterns
                r'^;\s*\w+',                   # Lines starting with semicolon
                r'CR/\d+',                     # Credit/reference numbers CR/9904
                r'Inv\.?\s*No:?',              # Invoice number headers
                r'Cabin\s*No',                 # Cabin numbers
                r'\bGROSS\b',                  # Gross totals
                r'\bpp\s+[A-Z]+',              # "pp ROSRABE" patterns
                r'\bim\s+\d+M\b',              # "im 100M" patterns
                r'\bip\s+[A-Z]+',              # "ip CAMILA" patterns
                r'\bBy\s+\d+\s+[a-z]+',        # "By 4 guwteros" patterns
                r'[A-Z]+\s*\d+\s*@',           # Reference patterns with @
                r'Lions\.?\s*Club',            # Lions Club
                r'OR/,\s*CR/',                 # OR/, CR/ patterns
                r'CTDAZ|TRAD',                 # OCR garbage
                r'[a-z]{2,3}\s+\d+\s*[a-z]+\s+\d+',  # "pk 164 SUPRAVA 16" patterns
                r'\d+M\s+[A-Z]+\s+IV',        # "100M PROLOP IV" patterns
                # Patient/Lab report patterns (not billing items)
                r"PATIENT'?S?\s*NAME",         # Patient name header
                r'SAMPLE\s*ID',                # Sample ID
                r'Leucocyte\s*Count',          # Lab test names
                r'Timing\s*:',                 # Timing headers
                r'\d+\.\d+\s*[ap]\.?m\.?',    # Time patterns like 08.30a.m.
                r'Serum\s*Globulin',           # Lab values
                r'cu\.?\s*mm',                 # Units like cu. mm
                r'gms?/d[l]?',                 # Units like gms/dl
                r'mg%\s*of\s*albumin',         # Lab result text
                r'Date\s*:?\s*[\d/\-]+',       # Date fields
                r'buh\?',                      # OCR garbage
                r'^_\s*[A-Z]+',                # Lines starting with underscore
                r'~\s*\.',                     # Tilde patterns
            ]
            for pattern in ocr_garbage_patterns:
                if re.search(pattern, description or "", re.IGNORECASE):
                    issues.append('Description contains OCR artifacts')
                    break
        
        # Direct check for Unicode replacement character and other non-ASCII garbage
        if not issues:
            desc_str = description or ""
            # Problematic Unicode ranges (OCR often produces these incorrectly):
            # - \ufffd (65533): replacement character
            # - 8200-8300: various dashes, quotes, special punctuation
            # - 171, 187: angle quotes
            problematic_chars = [c for c in desc_str if ord(c) in 
                                 [65533, 171, 187] or  # replacement char, angle quotes
                                 8200 <= ord(c) <= 8300]  # special punctuation range
            
            if problematic_chars:
                issues.append('Description contains OCR-related Unicode artifacts')
            # Check for high ratio of non-printable or special chars
            elif len(desc_str) > 0:
                special_chars = sum(1 for c in desc_str if ord(c) > 127 or c in '<>{}[]|\\')
                if special_chars / len(desc_str) > 0.08:  # More than 8% special chars
                    issues.append('Description contains too many special characters')
        
        # Additional check: If description starts with special characters or looks like ID
        if not issues:
            if re.match(r'^[\d\.]+\s*[\.\-]\s*[a-z]+\s*[\.\-]+\s*no', desc_lower):
                issues.append('Description is metadata, not a billing item')
        
        # Check description for gibberish (only if no other issues yet)
        if not issues:
            is_gibberish, gibberish_conf = self.is_gibberish(description)
            if is_gibberish:
                issues.append(f'Description appears to be OCR gibberish (confidence: {gibberish_conf:.0%})')
        
        # Validate amount
        amount_result = self.validate_amount(amount, context)
        if not amount_result.is_valid:
            issues.extend(amount_result.issues)
        
        # Cross-validate description and amount
        # Room charges typically have specific ranges
        if any(kw in desc_lower for kw in ['room', 'bed', 'ward', 'accommodation']):
            if amount > 50000:  # Per day room > 50k is suspicious
                issues.append('Room charge seems too high for single day')
        
        # ICU charges
        if any(kw in desc_lower for kw in ['icu', 'intensive', 'critical']):
            if amount < 1000:  # ICU < 1000 is suspicious
                issues.append('ICU charge seems too low')
        
        # Consultation fees
        if any(kw in desc_lower for kw in ['consultation', 'visit', 'opinion']):
            if amount > 20000:  # Consultation > 20k is unusual
                issues.append('Consultation fee seems very high')
        
        is_valid = len(issues) == 0
        confidence = 1.0 - (len(issues) * 0.2)
        
        return ValidationResult(
            is_valid=is_valid,
            original_value={'description': description, 'amount': amount},
            cleaned_value={'description': description, 'amount': amount} if is_valid else None,
            issues=issues,
            confidence=max(0.0, confidence)
        )
    
    def clean_extracted_text(self, text: str, min_confidence: float = 0.5) -> Tuple[str, float]:
        """
        Clean extracted text by removing gibberish sections
        Returns (cleaned_text, confidence)
        """
        if not text:
            return "", 0.0
        
        lines = text.split('\n')
        cleaned_lines = []
        total_confidence = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            is_gibberish, conf = self.is_gibberish(line)
            
            if not is_gibberish or conf < (1 - min_confidence):
                cleaned_lines.append(line)
                total_confidence += (1 - conf)
        
        cleaned_text = '\n'.join(cleaned_lines)
        avg_confidence = total_confidence / max(len(lines), 1)
        
        return cleaned_text, avg_confidence
    
    def validate_claim_data(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate entire claim data dictionary
        Returns validated data with confidence scores
        """
        validated = {}
        validation_report = {
            'total_fields': 0,
            'valid_fields': 0,
            'issues': [],
            'field_results': {}
        }
        
        for key, value in claim_data.items():
            validation_report['total_fields'] += 1
            
            if value is None:
                validated[key] = None
                validation_report['valid_fields'] += 1
                continue
            
            # Validate based on field type
            if key in ['billed_amount', 'total_amount', 'total_itemized']:
                result = self.validate_amount(value, key, 'total_bill')
                validated[key] = result.cleaned_value if result.is_valid else value
                validation_report['field_results'][key] = result.to_dict()
                if result.is_valid:
                    validation_report['valid_fields'] += 1
                else:
                    validation_report['issues'].append(f"{key}: {', '.join(result.issues)}")
            
            elif key == 'billing_items' and isinstance(value, list):
                validated_items = []
                for item in value:
                    if isinstance(item, dict):
                        desc = item.get('description', '')
                        amt = item.get('amount', 0)
                        result = self.validate_billing_item(desc, amt)
                        if result.is_valid:
                            validated_items.append(item)
                            validation_report['valid_fields'] += 1
                        else:
                            validation_report['issues'].append(
                                f"Billing item '{desc[:30]}': {', '.join(result.issues)}"
                            )
                validated[key] = validated_items
            
            elif isinstance(value, str) and len(value) > 10:
                is_gibberish, conf = self.is_gibberish(value)
                if is_gibberish and conf > 0.7:
                    validated[key] = None  # Remove gibberish
                    validation_report['issues'].append(f"{key}: Gibberish detected (confidence: {conf:.0%})")
                else:
                    validated[key] = value
                    validation_report['valid_fields'] += 1
            
            else:
                validated[key] = value
                validation_report['valid_fields'] += 1
        
        validated['_validation_report'] = validation_report
        return validated


def validate_claim(claim_data: Dict[str, Any], level: ValidationLevel = ValidationLevel.MODERATE) -> Dict[str, Any]:
    """Convenience function to validate claim data"""
    validator = DataValidator(level)
    return validator.validate_claim_data(claim_data)


def is_valid_amount(value: Any, context: str = "") -> bool:
    """Quick check if a value is a valid billing amount"""
    validator = DataValidator()
    result = validator.validate_amount(value, context)
    return result.is_valid


# ============================================================================
# ICD-10 CODE VALIDATION
# ============================================================================

def validate_icd10_code(code: str) -> Tuple[bool, str]:
    """
    Validate ICD-10 code format.
    Valid format: Letter + 2 digits + optional decimal + up to 4 more characters
    Examples: A00, A00.0, A00.01, S82.101A
    
    Returns: (is_valid, error_message or category_description)
    """
    if not code or not isinstance(code, str):
        return False, "Empty or invalid code"
    
    code = code.strip().upper()
    
    # ICD-10 pattern: Letter + 2 digits + optional (decimal + 1-4 alphanumeric)
    icd10_pattern = r'^[A-Z]\d{2}(?:\.\d{1,4}[A-Z]?)?$'
    
    if not re.match(icd10_pattern, code):
        return False, f"Invalid ICD-10 format: {code}"
    
    # Category descriptions
    icd10_categories = {
        'A': 'Certain infectious and parasitic diseases (A00-A99)',
        'B': 'Certain infectious and parasitic diseases (B00-B99)',
        'C': 'Neoplasms (C00-C99)',
        'D': 'Diseases of blood/immune system or Neoplasms (D00-D89)',
        'E': 'Endocrine, nutritional and metabolic diseases (E00-E89)',
        'F': 'Mental, behavioral and neurodevelopmental disorders (F01-F99)',
        'G': 'Diseases of the nervous system (G00-G99)',
        'H': 'Diseases of the eye/ear (H00-H95)',
        'I': 'Diseases of the circulatory system (I00-I99)',
        'J': 'Diseases of the respiratory system (J00-J99)',
        'K': 'Diseases of the digestive system (K00-K95)',
        'L': 'Diseases of the skin and subcutaneous tissue (L00-L99)',
        'M': 'Diseases of the musculoskeletal system (M00-M99)',
        'N': 'Diseases of the genitourinary system (N00-N99)',
        'O': 'Pregnancy, childbirth and the puerperium (O00-O9A)',
        'P': 'Certain conditions originating in the perinatal period (P00-P96)',
        'Q': 'Congenital malformations (Q00-Q99)',
        'R': 'Symptoms, signs and abnormal clinical findings (R00-R99)',
        'S': 'Injury, poisoning - anatomical region (S00-S99)',
        'T': 'Injury, poisoning - cause/nature (T00-T88)',
        'U': 'Codes for special purposes (U00-U85)',
        'V': 'External causes - transport accidents (V00-V99)',
        'W': 'External causes - other accidents (W00-W99)',
        'X': 'External causes - intentional/undetermined (X00-X99)',
        'Y': 'External causes - medical complications (Y00-Y99)',
        'Z': 'Factors influencing health status (Z00-Z99)',
    }
    
    first_char = code[0]
    category = icd10_categories.get(first_char, f"Unknown category: {first_char}")
    
    return True, category


def extract_valid_icd_codes(text: str) -> List[Dict[str, str]]:
    """
    Extract and validate all ICD-10 codes from text.
    Returns list of {code, category, is_valid}
    """
    if not text:
        return []
    
    # Find potential ICD-10 codes
    pattern = r'\b([A-Z]\d{2}(?:\.\d{1,4}[A-Z]?)?)\b'
    potential_codes = re.findall(pattern, text.upper())
    
    results = []
    seen = set()
    
    for code in potential_codes:
        if code in seen:
            continue
        seen.add(code)
        
        is_valid, category = validate_icd10_code(code)
        if is_valid:
            results.append({
                'code': code,
                'category': category,
                'is_valid': True
            })
    
    return results


# ============================================================================
# PRE-AGENT DATA VALIDATOR (CLEANS DATA BEFORE AGENTS SEE IT)
# ============================================================================

class PreAgentDataValidator:
    """
    Comprehensive data validator that cleans claim data BEFORE fraud agents process it.
    
    This ensures agents receive clean, validated data rather than raw OCR output
    containing gibberish, corrupted amounts, and form template text.
    
    Key Functions:
    1. Filter OCR gibberish from descriptions
    2. Validate and clean billing amounts (₹1,000 - ₹50,00,000 range)
    3. Filter out non-billing numbers (account IDs, dates, PIN codes)
    4. Detect and remove form template text
    5. Validate ICD-10 codes
    6. Normalize and clean text fields
    """
    
    # Amount ranges for Indian healthcare (in INR)
    VALID_AMOUNT_RANGES = {
        'line_item': (1, 1000000),           # Individual items: ₹1 to ₹10 lakh
        'total_bill': (1000, 50000000),      # Total: ₹1000 to ₹5 crore
        'room_charge': (500, 100000),        # Per day: ₹500 to ₹1 lakh
        'surgery': (5000, 5000000),          # Surgery: ₹5000 to ₹50 lakh
        'medicine': (10, 500000),            # Medicines: ₹10 to ₹5 lakh
    }
    
    # Patterns that indicate non-billing numbers (to be filtered)
    NON_BILLING_PATTERNS = [
        r'^\d{10,}$',                         # Phone numbers (10+ digits)
        r'^\d{6}$',                           # PIN codes (exactly 6 digits)
        r'^\d{8}$',                           # Date formats (DDMMYYYY)
        r'^\d{2}/\d{2}/\d{4}$',              # Date format DD/MM/YYYY
        r'^[A-Z]{4}0\d{6}$',                 # IFSC codes
        r'^\d{9,18}$',                        # Account numbers
        r'^50\d{14}$',                        # Card numbers starting with 50
        r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',     # Vehicle registration
    ]
    
    # Form template text patterns (not actual diagnoses/procedures)
    FORM_TEMPLATE_PATTERNS = [
        r'infectious.*parasitic.*diseases',
        r'pregnancy.*childbirth.*puerperium',
        r'eye.*ear.*diseases',
        r'mental.*behavioral.*disorders',
        r'diseases\s+of\s+the\s+(?:eye|ear|skin|nervous|circulatory)',
        r'certain\s+conditions\s+originating',
        r'congenital\s+malformations',
        r'factors\s+influencing\s+health',
        r'symptoms.*signs.*abnormal.*findings',
        r'external\s+causes\s+of\s+morbidity',
        r'codes\s+for\s+special\s+purposes',
        r'tick\s+(?:the\s+)?(?:appropriate|relevant)',
        r'please\s+(?:tick|mark|select)',
        r'if\s+(?:yes|no|applicable)',
        r'\(\s*\)\s*yes\s*\(\s*\)\s*no',
        r'form\s+instructions?',
        # Registration/metadata fields that get OCR'd as billing items
        r'^\s*reg\.?\s*no\.?\s*:?\s*\d',               # Reg No: 12345
        r'^\s*r\.?\s*no\.?\s*:?\s*\d',                 # R. No: 12345
        r'^\s*bill\s*no\.?\s*:?\s*\d',                 # Bill No: 12345
        r'^\s*date\s*:?\s*[\d/\-]',                    # Date: 01/01/2024
        r'^\s*time\s*:?\s*[\d:]',                      # Time: 10:30
        r'^\s*patient\s*(name|id|no)',                 # Patient Name/ID
    ]
    
    # Common OCR gibberish patterns
    GIBBERISH_PATTERNS = [
        r'[A-Z]{2,}[a-z]{1,2}[A-Z]{2,}',      # Mixed case like "TDIHiVIA"
        r'[bcdfghjklmnpqrstvwxyz]{6,}',        # 6+ consonants (no vowels)
        r'\b[a-z][A-Z][a-z][A-Z][a-z]\b',      # Alternating case
        r'[|!]{2,}',                            # OCR artifacts
        r'[\[\]{}]{3,}',                        # Multiple brackets
        r'[0-9][A-Za-z][0-9][A-Za-z][0-9]',    # Alternating digit-letter
        r'\b[A-Z]{3,}[0-9]+[A-Z]+\b',          # Random alphanumeric
        # Known garbage words (case-insensitive search anywhere in text)
        r'(?i)\b(?:fanme|TDIHIVIA|IRVINIAITIM|KIAITIE|yEomate|eAge|ChequetD0|orsccae|Payabie|detats)\b',
        r'STAIN\s+T[A-Z]{5,}',                  # "STAIN TDIHIVIA" pattern
        r'\b[A-Z]{7,}\b(?!.*(?:HOSPITAL|MEDICAL|SURGERY|CHARGES|GENERAL|SPECIAL))', # Long caps words (except valid ones)
    ]
    
    def __init__(self, level: ValidationLevel = ValidationLevel.MODERATE):
        self.level = level
        self.base_validator = DataValidator(level)
        
        # Compile patterns for efficiency
        self._non_billing_re = [re.compile(p) for p in self.NON_BILLING_PATTERNS]
        self._form_template_re = [re.compile(p, re.IGNORECASE) for p in self.FORM_TEMPLATE_PATTERNS]
        self._gibberish_re = [re.compile(p, re.IGNORECASE) for p in self.GIBBERISH_PATTERNS]
    
    def validate_and_clean(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point: Validate and clean all claim data before agent processing.
        
        Returns cleaned claim_data dict with:
        - Gibberish descriptions removed
        - Invalid amounts filtered
        - Form template text removed
        - ICD codes validated
        - Validation report attached
        """
        if not claim_data:
            return {'_validation_report': {'error': 'Empty claim data'}}
        
        cleaned = {}
        report = {
            'original_fields': len(claim_data),
            'cleaned_fields': 0,
            'removed_items': [],
            'warnings': [],
            'icd_validation': [],
            'amount_corrections': [],
        }
        
        for key, value in claim_data.items():
            if key.startswith('_'):  # Skip internal fields
                cleaned[key] = value
                continue
            
            # Clean based on field type
            if key == 'billing_items':
                cleaned_items, item_report = self._clean_billing_items(value)
                cleaned[key] = cleaned_items
                report['removed_items'].extend(item_report.get('removed', []))
                report['amount_corrections'].extend(item_report.get('corrections', []))
                
            elif key in ['billed_amount', 'total_amount', 'total_itemized']:
                cleaned_amt, amt_report = self._clean_amount(value, key)
                cleaned[key] = cleaned_amt
                if amt_report:
                    report['amount_corrections'].append(amt_report)
                    
            elif key in ['diagnosis', 'diagnosis_icd_codes', 'diagnosis_icd_list']:
                cleaned_val, diag_report = self._clean_diagnosis_field(key, value)
                cleaned[key] = cleaned_val
                report['icd_validation'].extend(diag_report)
                
            elif key == 'procedures':
                cleaned[key] = self._clean_procedures(value)
                
            elif isinstance(value, str):
                cleaned_text = self._clean_text_field(value)
                cleaned[key] = cleaned_text if cleaned_text else value
                
            else:
                cleaned[key] = value
            
            report['cleaned_fields'] += 1
        
        # Recalculate total_itemized from cleaned billing items
        if 'billing_items' in cleaned and cleaned['billing_items']:
            new_total = sum(item.get('amount', 0) for item in cleaned['billing_items'])
            if 'total_itemized' in cleaned:
                old_total = cleaned.get('total_itemized', 0) or 0
                if abs(old_total - new_total) > 100:  # Significant difference
                    report['amount_corrections'].append({
                        'field': 'total_itemized',
                        'original': old_total,
                        'corrected': new_total,
                        'reason': 'Recalculated from cleaned billing items'
                    })
            cleaned['total_itemized'] = new_total
        
        cleaned['_validation_report'] = report
        return cleaned
    
    def _clean_billing_items(self, items: Any) -> Tuple[List[Dict], Dict]:
        """Clean and filter billing items, removing gibberish and invalid amounts."""
        report = {'removed': [], 'corrections': []}
        
        if not items or not isinstance(items, list):
            return [], report
        
        cleaned_items = []
        
        for item in items:
            if not isinstance(item, dict):
                continue
            
            desc = str(item.get('description', ''))
            amount = item.get('amount', 0)
            
            # Skip empty descriptions
            if not desc or len(desc.strip()) < 2:
                report['removed'].append({
                    'description': desc[:50] if desc else '(empty)',
                    'amount': amount,
                    'reason': 'Empty or too short description'
                })
                continue
            
            # Check for gibberish description (ONLY obvious OCR garbage)
            if self._is_gibberish_description(desc):
                report['removed'].append({
                    'description': desc[:50],
                    'amount': amount,
                    'reason': 'Gibberish OCR text'
                })
                continue
            
            # Check for form template text (NOT medical billing items)
            if self._is_form_template(desc):
                report['removed'].append({
                    'description': desc[:50],
                    'amount': amount,
                    'reason': 'Form template text'
                })
                continue
            
            # Validate amount - but be lenient for legitimate procedures
            if not self._is_valid_billing_amount(amount):
                # Only reject if amount is clearly wrong (negative, too small, or astronomical)
                try:
                    amt = float(amount)
                    if amt < 0 or amt > 50000000:  # 5 Crore is absolute max
                        report['removed'].append({
                            'description': desc[:50],
                            'amount': amount,
                            'reason': f'Invalid amount (outside Rs.0-Rs.5Cr range)'
                        })
                        continue
                except (ValueError, TypeError):
                    pass  # Keep item if amount parsing fails, let agents handle it
            
            # Item passed all checks - keep it
            cleaned_items.append(item)
        
        return cleaned_items, report
    
    def _clean_amount(self, value: Any, field_name: str) -> Tuple[Optional[float], Optional[Dict]]:
        """Clean and validate a billing amount."""
        if value is None:
            return None, None
        
        try:
            # Convert to float
            if isinstance(value, str):
                value = float(value.replace(',', '').replace('₹', '').strip())
            amount = float(value)
        except (ValueError, TypeError):
            return None, {'field': field_name, 'original': value, 'corrected': None, 'reason': 'Invalid number format'}
        
        # Check if it's a valid billing amount
        range_key = 'total_bill' if 'total' in field_name.lower() else 'line_item'
        min_val, max_val = self.VALID_AMOUNT_RANGES.get(range_key, (1, 50000000))
        
        if amount < min_val or amount > max_val:
            return None, {
                'field': field_name,
                'original': amount,
                'corrected': None,
                'reason': f'Amount ₹{amount:,.2f} outside valid range ₹{min_val:,}-₹{max_val:,}'
            }
        
        # Check for non-billing number patterns
        amount_str = str(int(amount))
        for pattern in self._non_billing_re:
            if pattern.match(amount_str):
                return None, {
                    'field': field_name,
                    'original': amount,
                    'corrected': None,
                    'reason': 'Appears to be non-billing number (account/phone/date)'
                }
        
        return amount, None
    
    def _clean_diagnosis_field(self, field_name: str, value: Any) -> Tuple[Any, List[Dict]]:
        """Clean diagnosis field and validate ICD codes."""
        validation_results = []
        
        if field_name == 'diagnosis_icd_codes' and isinstance(value, list):
            # Validate ICD codes
            cleaned_codes = []
            for icd in value:
                if isinstance(icd, dict):
                    code = icd.get('code', '')
                    is_valid, category = validate_icd10_code(code)
                    validation_results.append({
                        'code': code,
                        'is_valid': is_valid,
                        'category': category
                    })
                    if is_valid:
                        cleaned_codes.append(icd)
            return cleaned_codes, validation_results
        
        elif field_name == 'diagnosis' and isinstance(value, str):
            # Check for form template text in diagnosis
            if self._is_form_template(value):
                return '', [{'field': 'diagnosis', 'issue': 'Contains form template text'}]
            return value, []
        
        return value, []
    
    def _clean_procedures(self, value: Any) -> List[str]:
        """Clean procedures list, removing gibberish entries."""
        if not value or not isinstance(value, list):
            return []
        
        cleaned = []
        for proc in value:
            proc_str = str(proc)
            if not self._is_gibberish_description(proc_str) and not self._is_form_template(proc_str):
                cleaned.append(proc_str)
        
        return cleaned
    
    def _clean_text_field(self, text: str) -> Optional[str]:
        """Clean a text field, removing gibberish but preserving valid content."""
        if not text:
            return text
        
        # Check for form template
        if self._is_form_template(text):
            return None
        
        # Check for gibberish
        if self._is_gibberish_description(text):
            return None
        
        return text.strip()
    
    def _is_gibberish_description(self, text: str) -> bool:
        """
        Check if text is OCR gibberish.
        
        This should ONLY flag obvious garbage text, not filter legitimate medical terms.
        Valid items like "Room Charges - General Ward" should NEVER be flagged.
        
        Strategy: Only flag text that matches EXPLICIT garbage patterns.
        Do NOT use probabilistic detection which may have false positives.
        """
        if not text or len(text) < 3:
            return False
        
        text_clean = text.strip()
        
        # First: Check if this contains valid medical/billing keywords (WHITELIST)
        valid_keywords = [
            'room', 'charge', 'ward', 'icu', 'ot', 'surgery', 'scan', 'test', 'blood',
            'medicine', 'injection', 'tablet', 'cap', 'drip', 'consultation', 'doctor',
            'nursing', 'dressing', 'physiotherapy', 'therapy', 'xray', 'x-ray', 'ct', 'mri', 'ecg',
            'usg', 'ultrasound', 'biopsy', 'pathology', 'laboratory', 'lab', 'service',
            'package', 'procedure', 'implant', 'orif', 'fracture', 'care', 'treatment',
            'anesthesia', 'anasthesia', 'oxygen', 'ventilator', 'ambulance', 'admission',
            'general', 'special', 'private', 'semi', 'deluxe', 'ac', 'non-ac', 'bed',
            'day', 'night', 'session', 'visit', 'fee', 'charges', 'cost', 'total',
            'surgical', 'medical', 'emergency', 'casualty', 'opd', 'ipd', 'diet', 'meal',
            'suture', 'catheter', 'bandage', 'syringe', 'saline', 'iv', 'fluid', 'kit',
        ]
        text_lower = text_clean.lower()
        if any(kw in text_lower for kw in valid_keywords):
            return False  # Contains valid medical/billing keyword - NOT gibberish
        
        # Check against EXPLICIT gibberish patterns only
        for pattern in self._gibberish_re:
            if pattern.search(text_clean):
                return True
        
        # Do NOT use base_validator.is_gibberish - too many false positives
        return False
    
    def _is_form_template(self, text: str) -> bool:
        """Check if text is form template/instruction text, not actual medical data."""
        if not text:
            return False
        
        text_lower = text.lower()
        
        for pattern in self._form_template_re:
            if pattern.search(text_lower):
                return True
        
        return False
    
    def _is_valid_billing_amount(self, amount: Any) -> bool:
        """Check if amount is in valid billing range and not a non-billing number."""
        try:
            amt = float(amount)
        except (ValueError, TypeError):
            return False
        
        # Check range
        min_val, max_val = self.VALID_AMOUNT_RANGES['line_item']
        if amt < min_val or amt > max_val:
            return False
        
        # Check for non-billing patterns
        amt_str = str(int(amt))
        for pattern in self._non_billing_re:
            if pattern.match(amt_str):
                return False
        
        return True


# ============================================================================
# CONTEXT-AWARE INDIAN HEALTHCARE PRICING
# ============================================================================

class IndianHealthcarePricing:
    """
    Context-aware pricing reference for Indian healthcare services.
    
    Provides realistic price ranges based on:
    - Location (Metro/Tier-1/Tier-2/Rural)
    - Hospital Type (Government/Private/Corporate)
    - Procedure specifics
    
    All prices in INR (Indian Rupees)
    """
    
    # Location multipliers (base = Tier-2 cities)
    LOCATION_MULTIPLIERS = {
        'metro': 2.0,           # Delhi, Mumbai, Bangalore, Chennai, Hyderabad, Kolkata
        'tier1': 1.5,           # Pune, Ahmedabad, Jaipur, Lucknow, etc.
        'tier2': 1.0,           # Smaller cities
        'rural': 0.6,           # Rural areas
    }
    
    # Hospital type multipliers (base = private)
    HOSPITAL_TYPE_MULTIPLIERS = {
        'government': 0.2,      # Government hospitals (highly subsidized)
        'charitable': 0.4,      # Trust/charitable hospitals
        'private': 1.0,         # Private hospitals
        'corporate': 1.8,       # Corporate chains (Apollo, Fortis, Max, etc.)
        'super_specialty': 2.5, # Super specialty centers
    }
    
    # Metro cities list
    METRO_CITIES = ['delhi', 'mumbai', 'bangalore', 'bengaluru', 'chennai', 'hyderabad', 'kolkata']
    TIER1_CITIES = ['pune', 'ahmedabad', 'jaipur', 'lucknow', 'kanpur', 'nagpur', 'indore', 
                   'bhopal', 'visakhapatnam', 'patna', 'vadodara', 'ghaziabad', 'ludhiana',
                   'coimbatore', 'kochi', 'thiruvananthapuram', 'chandigarh']
    
    # Base prices (for Tier-2 city, Private hospital) in INR
    BASE_PRICES = {
        # Surgical Procedures
        'orif': {  # Open Reduction Internal Fixation
            'description': 'ORIF (Fracture fixation surgery)',
            'low': 25000,
            'typical': 60000,
            'high': 150000,
            'super_specialty': 300000,
        },
        'orif_trimalleolar': {  # Trimalleolar ORIF (ankle)
            'description': 'ORIF Trimalleolar fracture',
            'low': 35000,
            'typical': 80000,
            'high': 200000,
            'super_specialty': 400000,
        },
        'appendectomy': {
            'description': 'Appendix removal surgery',
            'low': 15000,
            'typical': 35000,
            'high': 80000,
            'super_specialty': 150000,
        },
        'cesarean': {
            'description': 'C-Section delivery',
            'low': 20000,
            'typical': 50000,
            'high': 120000,
            'super_specialty': 250000,
        },
        'knee_replacement': {
            'description': 'Total Knee Replacement',
            'low': 100000,
            'typical': 250000,
            'high': 500000,
            'super_specialty': 1000000,
        },
        'hip_replacement': {
            'description': 'Total Hip Replacement',
            'low': 120000,
            'typical': 300000,
            'high': 600000,
            'super_specialty': 1200000,
        },
        'angioplasty': {
            'description': 'Coronary Angioplasty with stent',
            'low': 80000,
            'typical': 200000,
            'high': 400000,
            'super_specialty': 800000,
        },
        'cabg': {
            'description': 'Coronary Artery Bypass Graft',
            'low': 150000,
            'typical': 350000,
            'high': 700000,
            'super_specialty': 1500000,
        },
        
        # Diagnostic Imaging
        'xray': {
            'description': 'X-Ray (single view)',
            'low': 100,
            'typical': 300,
            'high': 800,
            'super_specialty': 1500,
        },
        'ct_scan': {
            'description': 'CT Scan (with contrast)',
            'low': 2000,
            'typical': 5000,
            'high': 12000,
            'super_specialty': 25000,
        },
        'mri': {
            'description': 'MRI Scan',
            'low': 4000,
            'typical': 10000,
            'high': 20000,
            'super_specialty': 40000,
        },
        'ultrasound': {
            'description': 'Ultrasound/Sonography',
            'low': 300,
            'typical': 800,
            'high': 2000,
            'super_specialty': 4000,
        },
        'ecg': {
            'description': 'ECG/EKG',
            'low': 100,
            'typical': 300,
            'high': 600,
            'super_specialty': 1200,
        },
        'echo': {
            'description': '2D Echo/Echocardiogram',
            'low': 800,
            'typical': 2000,
            'high': 4000,
            'super_specialty': 8000,
        },
        
        # Room Charges (per day)
        'general_ward': {
            'description': 'General Ward (per day)',
            'low': 500,
            'typical': 1500,
            'high': 4000,
            'super_specialty': 8000,
        },
        'semi_private': {
            'description': 'Semi-Private Room (per day)',
            'low': 1500,
            'typical': 4000,
            'high': 10000,
            'super_specialty': 20000,
        },
        'private_room': {
            'description': 'Private Room (per day)',
            'low': 3000,
            'typical': 8000,
            'high': 20000,
            'super_specialty': 40000,
        },
        'icu': {
            'description': 'ICU (per day)',
            'low': 5000,
            'typical': 15000,
            'high': 40000,
            'super_specialty': 80000,
        },
        
        # Laboratory Tests
        'blood_test_basic': {
            'description': 'Basic Blood Panel (CBC, etc)',
            'low': 200,
            'typical': 500,
            'high': 1200,
            'super_specialty': 2500,
        },
        'blood_test_comprehensive': {
            'description': 'Comprehensive Blood Panel',
            'low': 1000,
            'typical': 2500,
            'high': 5000,
            'super_specialty': 10000,
        },
        
        # Consultations
        'consultation_general': {
            'description': 'General Physician Consultation',
            'low': 200,
            'typical': 500,
            'high': 1000,
            'super_specialty': 2000,
        },
        'consultation_specialist': {
            'description': 'Specialist Consultation',
            'low': 500,
            'typical': 1000,
            'high': 2500,
            'super_specialty': 5000,
        },
        'consultation_super_specialist': {
            'description': 'Super Specialist Consultation',
            'low': 1000,
            'typical': 2000,
            'high': 5000,
            'super_specialty': 10000,
        },
    }
    
    @classmethod
    def get_location_type(cls, location: str) -> str:
        """Determine location type from city/area name."""
        if not location:
            return 'tier2'
        
        location_lower = location.lower()
        
        if any(city in location_lower for city in cls.METRO_CITIES):
            return 'metro'
        if any(city in location_lower for city in cls.TIER1_CITIES):
            return 'tier1'
        if any(term in location_lower for term in ['village', 'rural', 'taluka', 'tehsil']):
            return 'rural'
        
        return 'tier2'
    
    @classmethod
    def get_hospital_type(cls, hospital_name: str) -> str:
        """Determine hospital type from name."""
        if not hospital_name:
            return 'private'
        
        name_lower = hospital_name.lower()
        
        # Government indicators
        if any(term in name_lower for term in ['government', 'govt', 'district', 'civil', 
                                                'primary health', 'phc', 'chc', 'esi']):
            return 'government'
        
        # Corporate chains
        if any(term in name_lower for term in ['apollo', 'fortis', 'max', 'medanta', 'narayana',
                                                'manipal', 'columbia asia', 'aster', 'wockhardt',
                                                'kokilaben', 'lilavati', 'breach candy']):
            return 'corporate'
        
        # Super specialty
        if any(term in name_lower for term in ['super', 'specialty', 'speciality', 'cancer',
                                                'cardiac', 'neuro', 'ortho center']):
            return 'super_specialty'
        
        # Charitable/Trust
        if any(term in name_lower for term in ['trust', 'charitable', 'mission', 'seva']):
            return 'charitable'
        
        return 'private'
    
    @classmethod
    def get_price_range(cls, procedure: str, location: str = None, hospital: str = None) -> Dict:
        """
        Get adjusted price range for a procedure based on location and hospital type.
        
        Args:
            procedure: Procedure name (e.g., 'orif', 'ct_scan', 'icu')
            location: City/area name
            hospital: Hospital name
            
        Returns:
            Dict with 'low', 'typical', 'high' prices and metadata
        """
        # Find matching procedure
        proc_key = cls._match_procedure(procedure)
        if not proc_key or proc_key not in cls.BASE_PRICES:
            return None
        
        base = cls.BASE_PRICES[proc_key]
        
        # Get multipliers
        loc_type = cls.get_location_type(location)
        hosp_type = cls.get_hospital_type(hospital)
        
        loc_mult = cls.LOCATION_MULTIPLIERS.get(loc_type, 1.0)
        hosp_mult = cls.HOSPITAL_TYPE_MULTIPLIERS.get(hosp_type, 1.0)
        
        # Calculate adjusted prices
        combined_mult = loc_mult * hosp_mult
        
        return {
            'procedure': proc_key,
            'description': base['description'],
            'low': int(base['low'] * combined_mult),
            'typical': int(base['typical'] * combined_mult),
            'high': int(base['high'] * combined_mult),
            'location_type': loc_type,
            'hospital_type': hosp_type,
            'location_multiplier': loc_mult,
            'hospital_multiplier': hosp_mult,
            'combined_multiplier': combined_mult,
        }
    
    @classmethod
    def _match_procedure(cls, procedure: str) -> Optional[str]:
        """Match a procedure description to known procedure keys."""
        if not procedure:
            return None
        
        proc_lower = procedure.lower()
        
        # Direct matches
        for key in cls.BASE_PRICES.keys():
            if key in proc_lower or key.replace('_', ' ') in proc_lower:
                return key
        
        # Keyword matching
        keyword_map = {
            'orif_trimalleolar': ['trimalleolar', 'tri-malleolar', 'ankle fracture'],
            'orif': ['orif', 'open reduction', 'internal fixation', 'fracture fixation'],
            'xray': ['x-ray', 'xray', 'x ray', 'radiograph'],
            'ct_scan': ['ct', 'cat scan', 'computed tomography'],
            'mri': ['mri', 'magnetic resonance'],
            'ultrasound': ['ultrasound', 'sonography', 'usg'],
            'ecg': ['ecg', 'ekg', 'electrocardiogram'],
            'echo': ['echo', 'echocardiogram', '2d echo'],
            'general_ward': ['general ward', 'ward charge'],
            'semi_private': ['semi private', 'twin sharing', 'double sharing'],
            'private_room': ['private room', 'single room', 'deluxe'],
            'icu': ['icu', 'intensive care', 'critical care', 'ccu', 'iccu'],
            'blood_test_basic': ['cbc', 'hemogram', 'blood count'],
            'blood_test_comprehensive': ['comprehensive', 'profile', 'panel'],
            'consultation_specialist': ['specialist', 'surgeon consultation'],
            'consultation_general': ['consultation', 'opd', 'visit'],
            'appendectomy': ['appendix', 'appendectomy', 'appendicitis'],
            'cesarean': ['cesarean', 'c-section', 'lscs'],
            'knee_replacement': ['knee replacement', 'tkr', 'total knee'],
            'hip_replacement': ['hip replacement', 'thr', 'total hip'],
            'angioplasty': ['angioplasty', 'ptca', 'stent'],
            'cabg': ['bypass', 'cabg', 'coronary artery bypass'],
        }
        
        for key, keywords in keyword_map.items():
            if any(kw in proc_lower for kw in keywords):
                return key
        
        return None
    
    @classmethod
    def validate_price(cls, amount: float, procedure: str, location: str = None, 
                      hospital: str = None, tolerance: float = 0.5) -> Dict:
        """
        Validate if a price is reasonable for given context.
        
        Args:
            amount: The billed amount
            procedure: Procedure description
            location: City/area
            hospital: Hospital name
            tolerance: Acceptable deviation from 'high' price (0.5 = 50% above high is max)
            
        Returns:
            Dict with validation result and details
        """
        price_range = cls.get_price_range(procedure, location, hospital)
        
        if not price_range:
            return {
                'is_valid': None,
                'reason': f'Unknown procedure: {procedure}',
                'amount': amount,
            }
        
        max_acceptable = price_range['high'] * (1 + tolerance)
        min_acceptable = price_range['low'] * 0.5  # Allow 50% below low for discounts
        
        if amount < min_acceptable:
            return {
                'is_valid': True,  # Suspiciously low but not invalid
                'warning': 'Amount below typical range',
                'amount': amount,
                'expected_range': price_range,
            }
        
        if amount > max_acceptable:
            return {
                'is_valid': False,
                'reason': f'Amount ₹{amount:,.0f} exceeds maximum expected ₹{max_acceptable:,.0f}',
                'amount': amount,
                'expected_range': price_range,
                'overbilling_percentage': ((amount - price_range['high']) / price_range['high']) * 100,
            }
        
        return {
            'is_valid': True,
            'amount': amount,
            'expected_range': price_range,
            'price_position': 'low' if amount <= price_range['low'] else 
                            'typical' if amount <= price_range['typical'] else 'high',
        }


# Convenience function
def validate_claim_pre_agent(claim_data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Convenience function to validate and clean claim data before agent processing.
    
    Returns:
        Tuple of (cleaned_claim_data, list_of_issues)
    """
    validator = PreAgentDataValidator()
    cleaned = validator.validate_and_clean(claim_data)
    
    # Extract issues from validation report for logging
    issues = []
    report = cleaned.get('_validation_report', {})
    
    for item in report.get('removed_items', []):
        issues.append(f"Removed gibberish item: {item.get('description', '')[:30]}... (Rs.{item.get('amount', 0):,.0f})")
    
    for correction in report.get('amount_corrections', []):
        issues.append(f"Corrected {correction.get('field', 'amount')}: Rs.{correction.get('original', 0):,.0f} -> Rs.{correction.get('corrected', 0):,.0f}")
    
    for warning in report.get('warnings', []):
        issues.append(f"Warning: {warning}")
    
    return cleaned, issues


# Test function
if __name__ == "__main__":
    validator = DataValidator()
    
    print("=" * 60)
    print("DATA VALIDATION TESTS")
    print("=" * 60)
    
    # Test gibberish detection
    print("\n1. GIBBERISH DETECTION:")
    test_texts = [
        "fanme STAIN TDIHIVIA RAG IRVINIAITIM. KIAITIE",  # Gibberish from OCR
        "bender Maio yEomate eAge vears months DatoofBimn",  # More gibberish
        "Room Charges - General Ward",  # Valid
        "Dr. Rajesh Kumar",  # Valid name
        "ICU Care Package Day 1",  # Valid
        "ChequetD0 Payabie detats O O OL 11 orsccae",  # Gibberish
        "Sandhya Raghunath Kate",  # Valid name
    ]
    
    for text in test_texts:
        is_gib, conf = validator.is_gibberish(text)
        status = "GIBBERISH" if is_gib else "VALID"
        print(f"  '{text[:40]}...' -> {status} ({conf:.0%})")
    
    # Test amount validation
    print("\n2. AMOUNT VALIDATION:")
    test_amounts = [
        (13659025.40, "High-value claim total"),
        (400706, "PIN code"),
        (6022025, "Date as number (06022025)"),
        (50100303349087, "Account number"),
        (33200, "Actual bill amount"),
        (11111, "Line item"),
        (1062807, "Suspicious large amount"),
        (101014, "Another amount"),
        (8004, "Valid room charge"),
        (6566.16, "Valid medicine charge"),
    ]
    
    for amount, context in test_amounts:
        result = validator.validate_amount(amount, context)
        status = "VALID" if result.is_valid else "INVALID"
        issues = ", ".join(result.issues) if result.issues else "OK"
        print(f"  ₹{amount:,.2f} ({context}) -> {status} - {issues}")
    
    # Test billing item validation
    print("\n3. BILLING ITEM VALIDATION:")
    test_items = [
        ("Room Charges - General Ward", 8004),
        ("fanme STAIN TDIHIVIA RAG", 11111),
        ("Medicine - Antibiotics", 6566.16),
        ("bender Maio yEomate eAge", 1062807),
        ("ICU Care Day 1", 25000),
        ("ChequetD0 Payabie detats", 101014),
    ]
    
    for desc, amount in test_items:
        result = validator.validate_billing_item(desc, amount)
        status = "VALID" if result.is_valid else "INVALID"
        issues = ", ".join(result.issues) if result.issues else "OK"
        print(f"  '{desc[:30]}' @ ₹{amount:,.2f} -> {status}")
        if result.issues:
            for issue in result.issues:
                print(f"      - {issue}")
