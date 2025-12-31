# agent/extractors/date_parser.py
"""
Date Parser for Insurance Claim Documents
Standardizes dates from various Indian/international formats to ISO format (YYYY-MM-DD)
"""

import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class DateType(Enum):
    """Types of dates in insurance claims"""
    ADMISSION = "Date of Admission"
    DISCHARGE = "Date of Discharge"
    BIRTH = "Date of Birth"
    POLICY_START = "Policy Start Date"
    POLICY_END = "Policy End Date"
    CLAIM = "Claim Date"
    SERVICE = "Date of Service"
    INJURY = "Date of Injury/Illness"
    SURGERY = "Surgery Date"
    UNKNOWN = "Unknown"


@dataclass
class ParsedDate:
    """Represents a parsed date with metadata"""
    original: str
    standardized: str  # ISO format YYYY-MM-DD
    date_type: DateType = DateType.UNKNOWN
    confidence: float = 0.0
    context: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'original': self.original,
            'standardized': self.standardized,
            'date_type': self.date_type.value,
            'confidence': self.confidence,
            'context': self.context[:50] if self.context else ""
        }


@dataclass
class DateExtractionResult:
    """Result of date extraction from document"""
    all_dates: List[ParsedDate] = field(default_factory=list)
    key_dates: Dict[str, str] = field(default_factory=dict)
    admission_date: Optional[str] = None
    discharge_date: Optional[str] = None
    date_of_birth: Optional[str] = None
    policy_start: Optional[str] = None
    policy_end: Optional[str] = None
    length_of_stay: Optional[int] = None
    extraction_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'all_dates': [d.to_dict() for d in self.all_dates],
            'key_dates': self.key_dates,
            'admission_date': self.admission_date,
            'discharge_date': self.discharge_date,
            'date_of_birth': self.date_of_birth,
            'policy_start': self.policy_start,
            'policy_end': self.policy_end,
            'length_of_stay': self.length_of_stay,
            'extraction_confidence': self.extraction_confidence
        }


class DateParser:
    """
    Parse and standardize dates from various formats found in Indian insurance claims
    
    Supported formats:
    - DD-MM-YYYY (06-02-2025)
    - DD/MM/YYYY (06/02/2025)
    - DDMMYYYY (06022025)
    - DD.MM.YYYY (06.02.2025)
    - DD Month YYYY (06 February 2025)
    - DD Mon YYYY (06 Feb 2025)
    - YYYY-MM-DD (2025-02-06) - ISO format
    - Month DD, YYYY (February 06, 2025)
    """
    
    # Month names for parsing
    MONTHS_FULL = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    
    MONTHS_SHORT = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]
    
    # Date type patterns - keywords that indicate what type of date follows
    DATE_TYPE_PATTERNS = {
        DateType.ADMISSION: [
            r'(?:date\s+of\s+)?admission',
            r'd\.?o\.?a\.?',
            r'admitted\s+(?:on|date)',
            r'admission\s+date',
            r'hospitali[sz]ation\s+(?:from|date)',
        ],
        DateType.DISCHARGE: [
            r'(?:date\s+of\s+)?discharge',
            r'd\.?o\.?d\.?',
            r'discharged?\s+(?:on|date)',
            r'discharge\s+date',
            r'hospitali[sz]ation\s+(?:to|end)',
        ],
        DateType.BIRTH: [
            r'(?:date\s+of\s+)?birth',
            r'd\.?o\.?b\.?',
            r'born\s+(?:on)?',
            r'birth\s+date',
            r'age.*born',
        ],
        DateType.POLICY_START: [
            r'policy\s+(?:start|commencement|from)',
            r'(?:coverage|insurance)\s+(?:start|from)',
            r'effective\s+(?:from|date)',
            r'valid\s+from',
        ],
        DateType.POLICY_END: [
            r'policy\s+(?:end|expiry|to)',
            r'(?:coverage|insurance)\s+(?:end|to|expiry)',
            r'valid\s+(?:to|till|until)',
            r'expiry\s+date',
        ],
        DateType.CLAIM: [
            r'claim\s+date',
            r'date\s+of\s+claim',
            r'claimed\s+(?:on)?',
        ],
        DateType.SERVICE: [
            r'(?:date\s+of\s+)?service',
            r'd\.?o\.?s\.?',
            r'service\s+date',
            r'treatment\s+date',
        ],
        DateType.INJURY: [
            r'(?:date\s+of\s+)?(?:injury|illness|accident)',
            r'd\.?o\.?i\.?',
            r'injured\s+(?:on)?',
            r'illness\s+(?:started|began)',
        ],
        DateType.SURGERY: [
            r'(?:date\s+of\s+)?(?:surgery|operation|procedure)',
            r'operated\s+(?:on)?',
            r'surgery\s+date',
        ],
    }
    
    # False positive patterns (things that look like dates but aren't)
    FALSE_POSITIVE_PATTERNS = [
        r'(?:PIN|PINCODE|ZIP)[\s:]*\d{6}',  # PIN codes
        r'(?:MOBILE|PHONE|TEL|FAX)[\s:]*\d{10}',  # Phone numbers
        r'(?:POLICY|CLAIM|MEMBER|ID)[\s:]*(?:NO\.?)?[\s:]*\d{8,}',  # ID numbers
        r'(?:ACCOUNT|A/C)[\s:]*\d{8,}',  # Account numbers
        r'IFSC[\s:]*[A-Z]{4}\d{7}',  # IFSC codes
        r'\d{2}:\d{2}(?::\d{2})?',  # Time formats
    ]
    
    def __init__(self):
        # Compile false positive patterns
        self.false_positive_re = [re.compile(p, re.IGNORECASE) for p in self.FALSE_POSITIVE_PATTERNS]
        
        # Compile date type patterns
        self.date_type_re = {}
        for date_type, patterns in self.DATE_TYPE_PATTERNS.items():
            combined = '|'.join(f'(?:{p})' for p in patterns)
            self.date_type_re[date_type] = re.compile(combined, re.IGNORECASE)
    
    def _is_false_positive(self, date_str: str, context: str) -> bool:
        """Check if a potential date is actually a false positive"""
        full_text = f"{context}"
        
        for pattern in self.false_positive_re:
            match = pattern.search(full_text)
            if match and date_str in match.group(0):
                return True
        
        return False
    
    def _get_context(self, text: str, pos: int, window: int = 60) -> str:
        """Extract context around a position"""
        start = max(0, pos - window)
        end = min(len(text), pos + window)
        return text[start:end]
    
    def _detect_date_type(self, context: str) -> DateType:
        """Detect the type of date based on surrounding context"""
        context_lower = context.lower()
        
        for date_type, pattern in self.date_type_re.items():
            if pattern.search(context_lower):
                return date_type
        
        return DateType.UNKNOWN
    
    def _validate_date(self, day: int, month: int, year: int) -> bool:
        """Validate date components"""
        # Basic validation
        if month < 1 or month > 12:
            return False
        if day < 1 or day > 31:
            return False
        if year < 1900 or year > 2100:
            return False
        
        # Month-specific day validation
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
        # Leap year check
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            days_in_month[1] = 29
        
        if day > days_in_month[month - 1]:
            return False
        
        return True
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse single date string to YYYY-MM-DD format"""
        date_str = date_str.strip()
        
        # Pattern 1: DDMMYYYY (8 digits)
        if len(date_str) == 8 and date_str.isdigit():
            try:
                day = int(date_str[0:2])
                month = int(date_str[2:4])
                year = int(date_str[4:8])
                if self._validate_date(day, month, year):
                    return datetime(year, month, day).strftime('%Y-%m-%d')
            except ValueError:
                pass
        
        # Pattern 2: DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        for separator in ['-', '/', '.']:
            if separator in date_str:
                parts = date_str.split(separator)
                if len(parts) == 3:
                    try:
                        # Try DD-MM-YYYY
                        day = int(parts[0])
                        month = int(parts[1])
                        year = int(parts[2])
                        
                        # Handle 2-digit year
                        if year < 100:
                            year = 2000 + year if year < 50 else 1900 + year
                        
                        if self._validate_date(day, month, year):
                            return datetime(year, month, day).strftime('%Y-%m-%d')
                        
                        # Try YYYY-MM-DD (ISO format)
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        
                        if self._validate_date(day, month, year):
                            return datetime(year, month, day).strftime('%Y-%m-%d')
                    except ValueError:
                        continue
        
        return None
    
    def _parse_month_name(self, month_name: str) -> Optional[int]:
        """Convert month name to number"""
        month_name = month_name.strip().capitalize()
        
        # Full month names
        if month_name in self.MONTHS_FULL:
            return self.MONTHS_FULL.index(month_name) + 1
        
        # Short month names
        month_short = month_name[:3]
        if month_short in self.MONTHS_SHORT:
            return self.MONTHS_SHORT.index(month_short) + 1
        
        return None
    
    def standardize_all_dates(self, text: str) -> Dict[str, str]:
        """
        Find and standardize all dates in text
        Returns dict mapping original date strings to ISO format
        """
        dates_found = {}
        
        # Pattern 1: DD-MM-YYYY, DD/MM/YYYY, DD.MM.YYYY
        pattern1 = r'(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})'
        for match in re.finditer(pattern1, text):
            day, month, year = match.groups()
            day, month, year = int(day), int(month), int(year)
            
            # Handle 2-digit year
            if year < 100:
                year = 2000 + year if year < 50 else 1900 + year
            
            context = self._get_context(text, match.start())
            if self._is_false_positive(match.group(0), context):
                continue
            
            if self._validate_date(day, month, year):
                try:
                    iso_date = datetime(year, month, day).strftime('%Y-%m-%d')
                    dates_found[match.group(0)] = iso_date
                except ValueError:
                    pass
        
        # Pattern 2: DDMMYYYY (8 consecutive digits)
        pattern2 = r'\b(\d{2})(\d{2})(\d{4})\b'
        for match in re.finditer(pattern2, text):
            day, month, year = match.groups()
            day, month, year = int(day), int(month), int(year)
            
            context = self._get_context(text, match.start())
            if self._is_false_positive(match.group(0), context):
                continue
            
            if self._validate_date(day, month, year):
                try:
                    iso_date = datetime(year, month, day).strftime('%Y-%m-%d')
                    dates_found[match.group(0)] = iso_date
                except ValueError:
                    pass
        
        # Pattern 3: DD Month YYYY or DD Mon YYYY
        month_pattern = '|'.join(self.MONTHS_FULL + self.MONTHS_SHORT)
        pattern3 = rf'(\d{{1,2}})\s*(?:st|nd|rd|th)?\s*({month_pattern})[,\s]+(\d{{4}})'
        
        for match in re.finditer(pattern3, text, re.IGNORECASE):
            day, month_name, year = match.groups()
            month = self._parse_month_name(month_name)
            
            if month:
                day, year = int(day), int(year)
                if self._validate_date(day, month, year):
                    try:
                        iso_date = datetime(year, month, day).strftime('%Y-%m-%d')
                        dates_found[match.group(0)] = iso_date
                    except ValueError:
                        pass
        
        # Pattern 4: Month DD, YYYY
        pattern4 = rf'({month_pattern})\s+(\d{{1,2}})(?:st|nd|rd|th)?[,\s]+(\d{{4}})'
        
        for match in re.finditer(pattern4, text, re.IGNORECASE):
            month_name, day, year = match.groups()
            month = self._parse_month_name(month_name)
            
            if month:
                day, year = int(day), int(year)
                if self._validate_date(day, month, year):
                    try:
                        iso_date = datetime(year, month, day).strftime('%Y-%m-%d')
                        dates_found[match.group(0)] = iso_date
                    except ValueError:
                        pass
        
        return dates_found
    
    def extract_key_dates(self, text: str, sections: Optional[Dict] = None) -> Dict[str, str]:
        """
        Extract important dates like admission, discharge, DOB, etc.
        Uses both text patterns and section-based extraction
        """
        key_dates = {}
        
        # Combine sections if provided
        full_text = text
        if sections:
            for section_name, section_text in sections.items():
                if isinstance(section_text, str):
                    full_text += "\n" + section_text
        
        # Generic date pattern
        date_pattern = r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{8}|\d{1,2}\s+(?:' + '|'.join(self.MONTHS_FULL + self.MONTHS_SHORT) + r')\s+\d{4})'
        
        # Admission date patterns
        admission_patterns = [
            r'(?:date\s+of\s+)?admission[\s:]+' + date_pattern,
            r'd\.?o\.?a\.?[\s:]+' + date_pattern,
            r'admitted\s+(?:on)?[\s:]+' + date_pattern,
        ]
        for pattern in admission_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                parsed = self._parse_date(match.group(1))
                if parsed:
                    key_dates['admission_date'] = parsed
                    break
        
        # Discharge date patterns
        discharge_patterns = [
            r'(?:date\s+of\s+)?discharge[\s:]+' + date_pattern,
            r'd\.?o\.?d\.?[\s:]+' + date_pattern,
            r'discharged?\s+(?:on)?[\s:]+' + date_pattern,
        ]
        for pattern in discharge_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                parsed = self._parse_date(match.group(1))
                if parsed:
                    key_dates['discharge_date'] = parsed
                    break
        
        # Date of birth patterns
        dob_patterns = [
            r'(?:date\s+of\s+)?birth[\s:]+' + date_pattern,
            r'd\.?o\.?b\.?[\s:]+' + date_pattern,
            r'born[\s:]+(?:on)?[\s:]*' + date_pattern,
        ]
        for pattern in dob_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                parsed = self._parse_date(match.group(1))
                if parsed:
                    key_dates['date_of_birth'] = parsed
                    break
        
        # Policy dates
        policy_start_patterns = [
            r'policy\s+(?:start|from|commencement)[\s:]+' + date_pattern,
            r'(?:coverage|insurance)\s+from[\s:]+' + date_pattern,
            r'valid\s+from[\s:]+' + date_pattern,
        ]
        for pattern in policy_start_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                parsed = self._parse_date(match.group(1))
                if parsed:
                    key_dates['policy_start'] = parsed
                    break
        
        policy_end_patterns = [
            r'policy\s+(?:end|to|expiry)[\s:]+' + date_pattern,
            r'(?:coverage|insurance)\s+(?:to|till)[\s:]+' + date_pattern,
            r'valid\s+(?:to|till|until)[\s:]+' + date_pattern,
            r'expiry[\s:]+' + date_pattern,
        ]
        for pattern in policy_end_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                parsed = self._parse_date(match.group(1))
                if parsed:
                    key_dates['policy_end'] = parsed
                    break
        
        return key_dates
    
    def calculate_length_of_stay(self, admission_date: str, discharge_date: str) -> Optional[int]:
        """Calculate hospital length of stay in days"""
        try:
            admission = datetime.strptime(admission_date, '%Y-%m-%d')
            discharge = datetime.strptime(discharge_date, '%Y-%m-%d')
            
            delta = (discharge - admission).days
            
            # Validation: length should be reasonable (0 to 365 days)
            if 0 <= delta <= 365:
                return delta
        except (ValueError, TypeError):
            pass
        
        return None
    
    def calculate_age(self, dob: str, reference_date: Optional[str] = None) -> Optional[int]:
        """Calculate age from date of birth"""
        try:
            birth_date = datetime.strptime(dob, '%Y-%m-%d')
            
            if reference_date:
                ref_date = datetime.strptime(reference_date, '%Y-%m-%d')
            else:
                ref_date = datetime.now()
            
            age = ref_date.year - birth_date.year
            
            # Adjust if birthday hasn't occurred yet this year
            if (ref_date.month, ref_date.day) < (birth_date.month, birth_date.day):
                age -= 1
            
            # Validation: age should be reasonable (0 to 120 years)
            if 0 <= age <= 120:
                return age
        except (ValueError, TypeError):
            pass
        
        return None
    
    def extract_all(self, text: str, sections: Optional[Dict] = None) -> DateExtractionResult:
        """
        Complete date extraction with all dates and key dates identified
        """
        result = DateExtractionResult()
        
        # Get all dates
        all_dates_dict = self.standardize_all_dates(text)
        
        # Convert to ParsedDate objects with context
        for original, standardized in all_dates_dict.items():
            # Find position in text for context
            pos = text.find(original)
            context = self._get_context(text, pos) if pos >= 0 else ""
            date_type = self._detect_date_type(context)
            
            # Calculate confidence based on context
            confidence = 0.7  # Base confidence
            if date_type != DateType.UNKNOWN:
                confidence += 0.2  # Higher if we identified the type
            if len(original) >= 8:
                confidence += 0.1  # Higher for complete date formats
            
            result.all_dates.append(ParsedDate(
                original=original,
                standardized=standardized,
                date_type=date_type,
                confidence=min(1.0, confidence),
                context=context
            ))
        
        # Get key dates
        result.key_dates = self.extract_key_dates(text, sections)
        
        # Set specific date fields
        result.admission_date = result.key_dates.get('admission_date')
        result.discharge_date = result.key_dates.get('discharge_date')
        result.date_of_birth = result.key_dates.get('date_of_birth')
        result.policy_start = result.key_dates.get('policy_start')
        result.policy_end = result.key_dates.get('policy_end')
        
        # Calculate length of stay
        if result.admission_date and result.discharge_date:
            result.length_of_stay = self.calculate_length_of_stay(
                result.admission_date, result.discharge_date
            )
        
        # Overall confidence
        if result.all_dates:
            result.extraction_confidence = sum(d.confidence for d in result.all_dates) / len(result.all_dates)
        
        return result
    
    def format_for_display(self, result: DateExtractionResult) -> Dict[str, Any]:
        """Format extraction result for dashboard display"""
        return {
            'admission_date': result.admission_date,
            'discharge_date': result.discharge_date,
            'date_of_birth': result.date_of_birth,
            'length_of_stay': f"{result.length_of_stay} days" if result.length_of_stay else None,
            'policy_period': f"{result.policy_start} to {result.policy_end}" if result.policy_start and result.policy_end else None,
            'all_dates_count': len(result.all_dates),
            'extraction_confidence': f"{result.extraction_confidence:.0%}" if result.extraction_confidence else "0%"
        }


def parse_dates(text: str, sections: Optional[Dict] = None) -> DateExtractionResult:
    """Convenience function to parse all dates from text"""
    parser = DateParser()
    return parser.extract_all(text, sections)


# Test function
if __name__ == "__main__":
    test_text = """
    CLAIM FORM - PART A: INSURED DETAILS
    
    Name: Rajesh Kumar
    Date of Birth: 15-06-1985
    DOB: 15/06/1985
    
    PART B: HOSPITALIZATION DETAILS
    
    Date of Admission: 06-02-2025
    D.O.A.: 06022025
    
    Date of Discharge: 10-02-2025
    D.O.D.: 10/02/2025
    
    Hospital Address:
    Plot No 20, Sector-6
    PIN CODE: 400706
    Mobile: 9876543210
    
    Policy Details:
    Policy Start: 01 January 2025
    Valid Till: 31 December 2025
    Policy No: 12345678
    
    Surgery Date: February 07, 2025
    """
    
    parser = DateParser()
    result = parser.extract_all(test_text)
    
    print("=" * 60)
    print("DATE EXTRACTION TEST")
    print("=" * 60)
    
    print(f"\nAll Dates Found ({len(result.all_dates)}):")
    for date in result.all_dates:
        print(f"  - {date.original} -> {date.standardized} [{date.date_type.value}] ({date.confidence:.0%})")
    
    print(f"\nKey Dates:")
    print(f"  Admission: {result.admission_date}")
    print(f"  Discharge: {result.discharge_date}")
    print(f"  DOB: {result.date_of_birth}")
    print(f"  Policy Start: {result.policy_start}")
    print(f"  Policy End: {result.policy_end}")
    print(f"  Length of Stay: {result.length_of_stay} days")
    
    print(f"\nOverall Confidence: {result.extraction_confidence:.0%}")
