"""
Text Postprocessing Module
- Text normalization
- Typo correction heuristics
- Domain-specific corrections
- OCR error pattern fixes
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PostprocessingConfig:
    """Configuration for text postprocessing"""
    # Enable/disable specific corrections
    fix_ocr_patterns: bool = True
    normalize_whitespace: bool = True
    normalize_case: bool = False  # Keep original case by default
    fix_common_typos: bool = True
    domain_corrections: bool = True
    
    # Confidence threshold for fuzzy matching
    fuzzy_threshold: float = 0.8
    
    # Custom corrections
    custom_replacements: Dict[str, str] = field(default_factory=dict)


class OCRErrorPatterns:
    """Common OCR misrecognition patterns"""
    
    # Character substitution errors (what OCR reads -> correct char)
    CHAR_SUBSTITUTIONS = {
        # Letters confused with numbers
        'O': ['0'],
        '0': ['O', 'Q', 'D'],
        'I': ['1', 'l', '|'],
        '1': ['I', 'l', '|', 'i'],
        'l': ['1', 'I', '|', 'i'],
        'S': ['5', '$'],
        '5': ['S'],
        'Z': ['2'],
        '2': ['Z'],
        'B': ['8', '3'],
        '8': ['B'],
        'G': ['6', '9'],
        '6': ['G'],
        
        # Letters confused with each other
        'rn': ['m'],
        'm': ['rn', 'nn'],
        'cl': ['d'],
        'd': ['cl'],
        'li': ['h', 'b'],
        'h': ['li', 'b', 'n'],
        'n': ['h', 'ri'],
        'ri': ['n'],
        'vv': ['w'],
        'w': ['vv', 'uu'],
        
        # Common OCR artifacts
        '|': ['I', 'l', '1'],
        '.': [',', "'"],
        ',': ['.', "'"],
        "'": [',', '`'],
        '"': ['``', "''"],
    }
    
    # Common OCR garbage patterns
    GARBAGE_PATTERNS = [
        r'[\[\]{}|\\]',  # Brackets often misread
        r'[<>]',  # Angle brackets
        r'\s+',  # Multiple whitespace
        r'[^\x00-\x7F]+',  # Non-ASCII (optional)
    ]
    
    # Patterns that indicate form field labels (to be removed from values)
    LABEL_PATTERNS = [
        r'^name\s*[:\-]?\s*',
        r'^patient\s+name\s*[:\-]?\s*',
        r'^policy\s*#?\s*[:\-]?\s*',
        r'^date\s*[:\-]?\s*',
        r'^hospital\s*[:\-]?\s*',
        r'^diagnosis\s*[:\-]?\s*',
        r'^claim\s*(?:no|number|#)?\s*[:\-]?\s*',
    ]


class TextNormalizer:
    """Basic text normalization"""
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace: multiple spaces to single, trim"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        return text.strip()
    
    @staticmethod
    def normalize_newlines(text: str) -> str:
        """Normalize newlines"""
        # Replace Windows newlines with Unix
        text = text.replace('\r\n', '\n')
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text
    
    @staticmethod
    def remove_control_chars(text: str) -> str:
        """Remove control characters except newlines and tabs"""
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    @staticmethod
    def fix_spacing_around_punctuation(text: str) -> str:
        """Fix spacing around punctuation"""
        # Remove space before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        # Add space after punctuation if missing
        text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
        return text
    
    @staticmethod
    def normalize_quotes(text: str) -> str:
        """Normalize various quote characters"""
        # Normalize single quotes (curly to straight)
        text = text.replace('\u2018', "'")  # left single quote
        text = text.replace('\u2019', "'")  # right single quote
        text = text.replace('`', "'")
        # Normalize double quotes (curly to straight)
        text = text.replace('\u201c', '"')  # left double quote
        text = text.replace('\u201d', '"')  # right double quote
        return text


class DomainCorrector:
    """Domain-specific corrections for insurance/medical forms"""
    
    # Common hospital/medical terms
    MEDICAL_TERMS = {
        'hospita1': 'hospital',
        'hospitel': 'hospital',
        'hosptial': 'hospital',
        'hosp1tal': 'hospital',
        'medica1': 'medical',
        'medicel': 'medical',
        'c1aim': 'claim',
        'po1icy': 'policy',
        'policiy': 'policy',
        'insurence': 'insurance',
        'insuranse': 'insurance',
        'insurnace': 'insurance',
        'pat1ent': 'patient',
        'patiant': 'patient',
        'diagnoisis': 'diagnosis',
        'diagnisis': 'diagnosis',
        'treatmant': 'treatment',
        'treatement': 'treatment',
        'amoun1': 'amount',
        'arnount': 'amount',
        'numbor': 'number',
        'nurnber': 'number',
    }
    
    # Common name suffixes/prefixes
    NAME_PARTS = {
        'mr': 'Mr.',
        'mrs': 'Mrs.',
        'ms': 'Ms.',
        'dr': 'Dr.',
        'shri': 'Shri',
        'smt': 'Smt.',
        'kumari': 'Kumari',
    }
    
    # Indian states (common misspellings)
    STATES = {
        'maharastra': 'Maharashtra',
        'maharashtr': 'Maharashtra',
        'gujrat': 'Gujarat',
        'karnatka': 'Karnataka',
        'tamilnadu': 'Tamil Nadu',
        'rajastan': 'Rajasthan',
        'madhyapradesh': 'Madhya Pradesh',
    }
    
    def correct_medical_terms(self, text: str) -> str:
        """Correct common medical/insurance term misspellings"""
        result = text
        for wrong, correct in self.MEDICAL_TERMS.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            result = pattern.sub(correct, result)
        return result
    
    def correct_name_parts(self, text: str) -> str:
        """Standardize name prefixes"""
        words = text.split()
        corrected = []
        
        for word in words:
            lower = word.lower().rstrip('.')
            if lower in self.NAME_PARTS:
                corrected.append(self.NAME_PARTS[lower])
            else:
                corrected.append(word)
        
        return ' '.join(corrected)
    
    def correct_states(self, text: str) -> str:
        """Correct state name misspellings"""
        result = text
        for wrong, correct in self.STATES.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            result = pattern.sub(correct, result)
        return result


class OCRErrorCorrector:
    """Fix common OCR errors"""
    
    def __init__(self):
        self.char_subs = OCRErrorPatterns.CHAR_SUBSTITUTIONS
    
    def fix_number_letter_confusion(self, text: str, context: str = 'general') -> str:
        """Fix common number/letter confusion based on context"""
        if context == 'number':
            # In numeric context, prefer numbers
            replacements = {
                'O': '0', 'o': '0',
                'I': '1', 'l': '1',
                'S': '5', 's': '5',
                'Z': '2', 'z': '2',
                'B': '8',
            }
        elif context == 'alpha':
            # In alphabetic context, prefer letters
            replacements = {
                '0': 'O',
                '1': 'I',
                '5': 'S',
                '2': 'Z',
                '8': 'B',
            }
        else:
            return text
        
        result = text
        for wrong, correct in replacements.items():
            result = result.replace(wrong, correct)
        return result
    
    def fix_rn_m_confusion(self, text: str) -> str:
        """Fix common rn->m OCR error"""
        # Common words where this occurs
        rn_words = {
            'narne': 'name',
            'rnedical': 'medical',
            'nurnber': 'number',
            'arnount': 'amount',
            'forni': 'form',
            'inforniation': 'information',
            'treatrnent': 'treatment',
            'govemrnent': 'government',
        }
        
        result = text
        for wrong, correct in rn_words.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            result = pattern.sub(correct, result)
        
        return result
    
    def remove_ocr_garbage(self, text: str) -> str:
        """Remove common OCR garbage characters"""
        # Remove isolated special characters
        text = re.sub(r'(?<!\w)[|\\]{1,3}(?!\w)', '', text)
        # Remove repeated punctuation
        text = re.sub(r'([.,;:]){2,}', r'\1', text)
        # Remove orphan brackets
        text = re.sub(r'[\[\]{}]', '', text)
        return text
    
    def fix_split_words(self, text: str) -> str:
        """Fix words split by OCR (spaces in middle)"""
        # Common patterns
        patterns = [
            (r'h\s+o\s+s\s+p\s+i\s+t\s+a\s+l', 'hospital'),
            (r'p\s+a\s+t\s+i\s+e\s+n\s+t', 'patient'),
            (r'n\s+a\s+m\s+e', 'name'),
        ]
        
        result = text
        for pattern, replacement in patterns:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result


class FieldSpecificPostprocessor:
    """Postprocessing tailored to specific field types"""
    
    @staticmethod
    def process_name(text: str) -> str:
        """Clean up extracted name"""
        # Remove common prefixes that might be labels
        text = re.sub(r'^(?:patient|name|insured|claimant)\s*[:\-]?\s*', '', text, flags=re.IGNORECASE)
        
        # Remove Mr./Mrs. etc at start (optional - keep if needed)
        # text = re.sub(r'^(?:mr|mrs|ms|dr|shri|smt)\.?\s+', '', text, flags=re.IGNORECASE)
        
        # Fix common OCR issues in names
        text = re.sub(r'[|1l](?=[a-z])', 'I', text)  # | or 1 before lowercase = I
        text = re.sub(r'(?<=[A-Z])[0O](?=[a-z])', 'o', text)  # 0 between upper and lower
        
        # Capitalize properly (Title Case)
        text = ' '.join(word.capitalize() for word in text.split())
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def process_policy_number(text: str) -> str:
        """Clean up policy number"""
        # Remove label
        text = re.sub(r'^(?:policy|pol)\.?\s*(?:no|number|#)?\.?\s*[:\-]?\s*', '', text, flags=re.IGNORECASE)
        
        # Keep only alphanumeric and common separators
        text = re.sub(r'[^A-Za-z0-9\-/]', '', text)
        
        # Convert to uppercase
        text = text.upper()
        
        # Fix common OCR errors in policy numbers (context: alphanumeric)
        # In policy numbers, usually expect specific patterns
        
        return text
    
    @staticmethod
    def process_date(text: str) -> str:
        """Clean up date field"""
        # Remove label
        text = re.sub(r'^(?:date|doa|dod|admission)\s*[:\-]?\s*', '', text, flags=re.IGNORECASE)
        
        # Keep only digits and date separators
        text = re.sub(r'[^0-9/\-.]', '', text)
        
        # Normalize separators to /
        text = re.sub(r'[\-.]', '/', text)
        
        # Try to extract date pattern
        date_match = re.search(r'(\d{1,2})[/](\d{1,2})[/](\d{2,4})', text)
        if date_match:
            day, month, year = date_match.groups()
            # Ensure 4-digit year
            if len(year) == 2:
                year = '20' + year if int(year) < 50 else '19' + year
            return f"{day.zfill(2)}/{month.zfill(2)}/{year}"
        
        return text
    
    @staticmethod
    def process_amount(text: str) -> str:
        """Clean up monetary amount"""
        # Remove currency symbols and labels
        text = re.sub(r'^(?:rs\.?|inr|amount|₹)\s*', '', text, flags=re.IGNORECASE)
        
        # Remove non-numeric except decimal point and comma
        text = re.sub(r'[^0-9.,]', '', text)
        
        # Handle Indian number format (commas as thousand separators)
        # Remove commas for clean number
        text = text.replace(',', '')
        
        # Ensure single decimal point
        parts = text.split('.')
        if len(parts) > 2:
            text = parts[0] + '.' + ''.join(parts[1:])
        
        return text
    
    @staticmethod
    def process_diagnosis(text: str) -> str:
        """Clean up diagnosis text"""
        # Remove label
        text = re.sub(r'^(?:diagnosis|disease|condition)\s*[:\-]?\s*', '', text, flags=re.IGNORECASE)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Title case for medical terms
        # (Could add medical term dictionary here)
        
        return text


class TextPostprocessor:
    """Main postprocessing pipeline"""
    
    def __init__(self, config: PostprocessingConfig = None):
        self.config = config or PostprocessingConfig()
        self.normalizer = TextNormalizer()
        self.domain_corrector = DomainCorrector()
        self.ocr_corrector = OCRErrorCorrector()
        self.field_processor = FieldSpecificPostprocessor()
    
    def postprocess(self, text: str, field_type: str = 'general') -> Dict[str, Any]:
        """
        Full postprocessing pipeline.
        
        Args:
            text: Raw OCR text
            field_type: One of 'name', 'policy_number', 'date', 'amount', 'diagnosis', 'general'
        
        Returns:
            Dictionary with processed text and metadata
        """
        original = text
        
        # Step 1: Basic normalization
        if self.config.normalize_whitespace:
            text = self.normalizer.normalize_whitespace(text)
            text = self.normalizer.remove_control_chars(text)
        
        # Step 2: Fix OCR errors
        if self.config.fix_ocr_patterns:
            text = self.ocr_corrector.remove_ocr_garbage(text)
            text = self.ocr_corrector.fix_rn_m_confusion(text)
            text = self.ocr_corrector.fix_split_words(text)
        
        # Step 3: Domain corrections
        if self.config.domain_corrections:
            text = self.domain_corrector.correct_medical_terms(text)
        
        # Step 4: Field-specific processing
        if field_type == 'name':
            text = self.field_processor.process_name(text)
        elif field_type == 'policy_number':
            text = self.field_processor.process_policy_number(text)
        elif field_type == 'date':
            text = self.field_processor.process_date(text)
        elif field_type == 'amount':
            text = self.field_processor.process_amount(text)
        elif field_type == 'diagnosis':
            text = self.field_processor.process_diagnosis(text)
        
        # Step 5: Custom replacements
        for pattern, replacement in self.config.custom_replacements.items():
            text = text.replace(pattern, replacement)
        
        # Step 6: Final cleanup
        text = self.normalizer.normalize_whitespace(text)
        text = self.normalizer.fix_spacing_around_punctuation(text)
        
        return {
            'original': original,
            'processed': text,
            'changed': original != text,
            'field_type': field_type
        }
    
    def postprocess_batch(self, texts: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """
        Process multiple texts.
        
        Args:
            texts: List of (text, field_type) tuples
        
        Returns:
            List of processed results
        """
        return [self.postprocess(text, field_type) for text, field_type in texts]


class SpellCorrector:
    """Simple spell correction using word frequency and edit distance"""
    
    def __init__(self, word_list: List[str] = None):
        self.word_freq = defaultdict(int)
        
        if word_list:
            for word in word_list:
                self.word_freq[word.lower()] += 1
    
    def add_words(self, words: List[str]):
        """Add words to dictionary"""
        for word in words:
            self.word_freq[word.lower()] += 1
    
    def correct(self, word: str, max_distance: int = 2) -> str:
        """Correct a single word"""
        if word.lower() in self.word_freq:
            return word
        
        candidates = []
        for dict_word in self.word_freq:
            ratio = SequenceMatcher(None, word.lower(), dict_word).ratio()
            if ratio > 0.7:
                candidates.append((dict_word, ratio, self.word_freq[dict_word]))
        
        if candidates:
            # Sort by similarity, then frequency
            candidates.sort(key=lambda x: (-x[1], -x[2]))
            corrected = candidates[0][0]
            
            # Preserve original case
            if word.isupper():
                return corrected.upper()
            elif word[0].isupper():
                return corrected.capitalize()
            return corrected
        
        return word


if __name__ == "__main__":
    # Test the postprocessor
    processor = TextPostprocessor()
    
    # Test cases
    test_cases = [
        ("SANDHYA RAGHUNATH KATE", "name"),
        ("narne: John Srnith", "name"),
        ("Po1icy No: ABC12345678", "policy_number"),
        ("Date: 15/O8/2024", "date"),
        ("Rs. 1,50,000.00", "amount"),
        ("hospita1 treatrnent for fever", "diagnosis"),
    ]
    
    print("=" * 60)
    print("Text Postprocessing Test")
    print("=" * 60)
    
    for text, field_type in test_cases:
        result = processor.postprocess(text, field_type)
        print(f"\nField Type: {field_type}")
        print(f"Original: '{result['original']}'")
        print(f"Processed: '{result['processed']}'")
        print(f"Changed: {result['changed']}")
