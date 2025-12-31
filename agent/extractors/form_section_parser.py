# agent/extractors/form_section_parser.py
"""
Form Section Parser for Multi-Part Insurance Claim Forms
Separates Part A (Insured), Part B (Hospital), Part C (Claims), Part D (Medical), Part E (Declaration)
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class FormSection(Enum):
    """Standard insurance claim form sections"""
    PART_A = "Part A - Insured Details"
    PART_B = "Part B - Hospital Details"
    PART_C = "Part C - Claim Details"
    PART_D = "Part D - Medical Details"
    PART_E = "Part E - Declaration"
    HEADER = "Header"
    FOOTER = "Footer"
    UNKNOWN = "Unknown"


@dataclass
class SectionInfo:
    """Information about a parsed section"""
    section_type: FormSection
    title: str
    content: str
    start_line: int
    end_line: int
    confidence: float = 0.0
    fields_found: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'section_type': self.section_type.value,
            'title': self.title,
            'content': self.content[:500] + '...' if len(self.content) > 500 else self.content,
            'start_line': self.start_line,
            'end_line': self.end_line,
            'confidence': self.confidence,
            'fields_found': self.fields_found,
            'line_count': self.end_line - self.start_line
        }


@dataclass
class ParsedForm:
    """Complete parsed form with all sections"""
    sections: Dict[str, SectionInfo] = field(default_factory=dict)
    raw_text: str = ""
    form_type: str = "Unknown"
    total_lines: int = 0
    parse_confidence: float = 0.0
    
    def get_section(self, section_type: FormSection) -> Optional[SectionInfo]:
        """Get a specific section by type"""
        for section in self.sections.values():
            if section.section_type == section_type:
                return section
        return None
    
    def get_section_content(self, section_type: FormSection) -> str:
        """Get content of a specific section"""
        section = self.get_section(section_type)
        return section.content if section else ""
    
    def to_dict(self) -> Dict:
        return {
            'form_type': self.form_type,
            'total_lines': self.total_lines,
            'parse_confidence': self.parse_confidence,
            'sections': {k: v.to_dict() for k, v in self.sections.items()},
            'section_count': len(self.sections)
        }


class FormSectionParser:
    """
    Parser for multi-section insurance claim forms
    Handles Indian TPA claim forms with Parts A, B, C, D, E
    """
    
    # Section header patterns
    SECTION_PATTERNS = {
        FormSection.PART_A: [
            r'PART\s*[-]?\s*A\b',
            r'SECTION\s*[-]?\s*A\b',
            r'A\s*[-\.]\s*(?:DETAILS?\s+OF\s+)?(?:PRIMARY\s+)?INSURED',
            r'(?:PRIMARY\s+)?INSURED\s+(?:DETAILS?|INFORMATION|PARTICULARS)',
            r'PATIENT\s+(?:DETAILS?|INFORMATION|PARTICULARS)',
            r'POLICYHOLDER\s+(?:DETAILS?|INFORMATION)',
            r'TO\s+BE\s+FILLED\s+(?:IN\s+)?BY\s+(?:THE\s+)?INSURED',
            r'MEMBER\s+(?:DETAILS?|INFORMATION)',
        ],
        FormSection.PART_B: [
            r'PART\s*[-]?\s*B\b',
            r'SECTION\s*[-]?\s*B\b',
            r'B\s*[-\.]\s*(?:DETAILS?\s+OF\s+)?HOSPITAL',
            r'HOSPITAL\s+(?:DETAILS?|INFORMATION|PARTICULARS)',
            r'(?:TREATING\s+)?HOSPITAL',
            r'TO\s+BE\s+FILLED\s+(?:IN\s+)?BY\s+(?:THE\s+)?HOSPITAL',
            r'PROVIDER\s+(?:DETAILS?|INFORMATION)',
            r'ADMISSION\s+(?:DETAILS?|INFORMATION)',
            r'HOSPITALI[ZS]ATION\s+(?:DETAILS?|INFORMATION)',
        ],
        FormSection.PART_C: [
            r'PART\s*[-]?\s*C\b',
            r'SECTION\s*[-]?\s*C\b',
            r'C\s*[-\.]\s*(?:DETAILS?\s+OF\s+)?CLAIM',
            r'CLAIM\s+(?:DETAILS?|INFORMATION|PARTICULARS)',
            r'BILLING\s+(?:DETAILS?|INFORMATION|PARTICULARS)',
            r'BILL\s+(?:OF\s+)?PARTICULARS',
            r'EXPENSE\s+(?:DETAILS?|BREAKDOWN)',
            r'ITEMIZED\s+(?:BILL|CHARGES?)',
            r'CHARGES?\s+(?:DETAILS?|BREAKDOWN)',
            r'AMOUNT\s+CLAIMED',
        ],
        FormSection.PART_D: [
            r'PART\s*[-]?\s*D\b',
            r'SECTION\s*[-]?\s*D\b',
            r'D\s*[-\.]\s*(?:DETAILS?\s+OF\s+)?(?:MEDICAL|TREATMENT)',
            r'MEDICAL\s+(?:DETAILS?|INFORMATION|RECORDS?)',
            r'TREATMENT\s+(?:DETAILS?|INFORMATION)',
            r'DIAGNOSIS\s+(?:DETAILS?|INFORMATION)',
            r'CLINICAL\s+(?:DETAILS?|INFORMATION|SUMMARY)',
            r'ATTENDING\s+(?:DOCTOR|PHYSICIAN)',
            r"DOCTOR'?S?\s+(?:DETAILS?|INFORMATION|CERTIFICATE)",
        ],
        FormSection.PART_E: [
            r'PART\s*[-]?\s*E\b',
            r'SECTION\s*[-]?\s*E\b',
            r'E\s*[-\.]\s*DECLARATION',
            r'DECLARATION\s+(?:BY|OF)\s+(?:THE\s+)?(?:INSURED|PATIENT)',
            r'AUTHORIZATION',
            r'CONSENT',
            r'SIGNATURE\s+OF\s+(?:THE\s+)?(?:INSURED|PATIENT)',
            r'I\s+(?:HEREBY\s+)?DECLARE',
            r'UNDERTAKING',
        ],
        FormSection.HEADER: [
            r'CLAIM\s+FORM',
            r'HEALTH\s+INSURANCE\s+CLAIM',
            r'MEDICLAIM\s+FORM',
            r'CASHLESS\s+(?:REQUEST|CLAIM)\s+FORM',
            r'PRE[-\s]?AUTH(?:ORIZATION)?\s+(?:REQUEST\s+)?FORM',
            r'REIMBURSEMENT\s+CLAIM\s+FORM',
        ],
    }
    
    # Field patterns for each section (to identify section by content)
    SECTION_FIELD_PATTERNS = {
        FormSection.PART_A: [
            r'policy\s*(?:no\.?|number)',
            r'name\s+of\s+(?:the\s+)?(?:insured|patient|member)',
            r'date\s+of\s+birth',
            r'(?:age|gender|sex)',
            r'address',
            r'contact\s*(?:no\.?|number)',
            r'email',
            r'member\s*(?:id|no\.?)',
            r'employee\s*(?:id|no\.?)',
            r'sum\s+insured',
            r'(?:uhid|mrn)',
        ],
        FormSection.PART_B: [
            r'hospital\s+name',
            r'(?:hospital\s+)?address',
            r'date\s+of\s+admission',
            r'date\s+of\s+discharge',
            r'room\s+(?:type|category)',
            r'(?:ipd|opd)\s*(?:no\.?|number)',
            r'treating\s+doctor',
            r'(?:registration|rohini)\s*(?:no\.?|number)',
            r'type\s+of\s+(?:admission|treatment)',
        ],
        FormSection.PART_C: [
            r'(?:total|billed?)\s+amount',
            r'room\s+(?:rent|charges?)',
            r'(?:icu|iccu)\s+charges?',
            r'(?:ot|operation|surgery)\s+charges?',
            r'medicine\s+(?:charges?|cost)',
            r'(?:lab|laboratory|pathology)\s+charges?',
            r'(?:consultation|doctor)\s+(?:fee|charges?)',
            r'(?:misc|miscellaneous|other)\s+charges?',
            r'(?:discount|deduction)',
            r'(?:net|payable)\s+amount',
        ],
        FormSection.PART_D: [
            r'diagnosis',
            r'(?:icd|icd-?10)\s*(?:code)?',
            r'(?:procedure|cpt)\s*(?:code)?',
            r'nature\s+of\s+illness',
            r'(?:history|complaint)',
            r'(?:findings?|investigation)',
            r'treatment\s+(?:given|details?)',
            r'(?:operation|surgery)\s+(?:performed|details?)',
            r'(?:date|duration)\s+of\s+(?:illness|symptoms?)',
        ],
        FormSection.PART_E: [
            r'signature',
            r'date',
            r'place',
            r'(?:i\s+)?(?:hereby\s+)?declare',
            r'(?:i\s+)?(?:hereby\s+)?authorize',
            r'consent',
            r'witness',
            r'stamp',
            r'seal',
        ],
    }
    
    def __init__(self):
        # Compile patterns for efficiency
        self.compiled_section_patterns = {}
        for section, patterns in self.SECTION_PATTERNS.items():
            self.compiled_section_patterns[section] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        self.compiled_field_patterns = {}
        for section, patterns in self.SECTION_FIELD_PATTERNS.items():
            self.compiled_field_patterns[section] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def parse(self, text: str) -> ParsedForm:
        """
        Parse a claim form into sections
        
        Args:
            text: OCR text from claim document
            
        Returns:
            ParsedForm with all identified sections
        """
        lines = text.split('\n')
        
        # Step 1: Find section boundaries
        section_boundaries = self._find_section_boundaries(lines)
        
        # Step 2: If no explicit sections found, try content-based detection
        if len(section_boundaries) <= 1:
            section_boundaries = self._detect_sections_by_content(lines)
        
        # Step 3: Extract sections
        sections = self._extract_sections(lines, section_boundaries)
        
        # Step 4: Identify form type
        form_type = self._identify_form_type(text)
        
        # Step 5: Calculate confidence
        confidence = self._calculate_parse_confidence(sections)
        
        return ParsedForm(
            sections=sections,
            raw_text=text,
            form_type=form_type,
            total_lines=len(lines),
            parse_confidence=confidence
        )
    
    def _find_section_boundaries(self, lines: List[str]) -> List[Tuple[int, FormSection, str]]:
        """Find section headers and their line numbers"""
        boundaries = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check against section patterns
            for section, patterns in self.compiled_section_patterns.items():
                for pattern in patterns:
                    if pattern.search(line_stripped):
                        boundaries.append((i, section, line_stripped))
                        break
                else:
                    continue
                break
        
        # Sort by line number
        boundaries.sort(key=lambda x: x[0])
        
        return boundaries
    
    def _detect_sections_by_content(self, lines: List[str]) -> List[Tuple[int, FormSection, str]]:
        """Detect sections by analyzing content patterns"""
        boundaries = []
        
        # Scan through document in chunks
        chunk_size = 20
        current_section = None
        
        for i in range(0, len(lines), chunk_size // 2):
            chunk = '\n'.join(lines[i:i + chunk_size])
            
            # Score each section type for this chunk
            scores = {}
            for section, patterns in self.compiled_field_patterns.items():
                score = sum(1 for p in patterns if p.search(chunk))
                if score > 0:
                    scores[section] = score
            
            if scores:
                best_section = max(scores, key=scores.get)
                if best_section != current_section and scores[best_section] >= 2:
                    # Found a new section
                    boundaries.append((i, best_section, f"Detected {best_section.value}"))
                    current_section = best_section
        
        return boundaries
    
    def _extract_sections(self, lines: List[str], 
                          boundaries: List[Tuple[int, FormSection, str]]) -> Dict[str, SectionInfo]:
        """Extract section content based on boundaries"""
        sections = {}
        
        if not boundaries:
            # No sections found - treat entire document as one section
            content = '\n'.join(lines)
            section_type = self._guess_section_type(content)
            sections['MAIN'] = SectionInfo(
                section_type=section_type,
                title="Main Content",
                content=content,
                start_line=0,
                end_line=len(lines),
                confidence=0.5,
                fields_found=self._find_fields_in_content(content)
            )
            return sections
        
        for i, (start_line, section_type, title) in enumerate(boundaries):
            # Determine end line
            if i + 1 < len(boundaries):
                end_line = boundaries[i + 1][0]
            else:
                end_line = len(lines)
            
            # Extract content
            content = '\n'.join(lines[start_line:end_line])
            
            # Find fields in this section
            fields_found = self._find_fields_in_content(content)
            
            # Calculate section confidence
            confidence = self._calculate_section_confidence(section_type, content, fields_found)
            
            # Use section type name as key
            section_key = section_type.name
            if section_key in sections:
                section_key = f"{section_key}_{i}"
            
            sections[section_key] = SectionInfo(
                section_type=section_type,
                title=title,
                content=content,
                start_line=start_line,
                end_line=end_line,
                confidence=confidence,
                fields_found=fields_found
            )
        
        return sections
    
    def _find_fields_in_content(self, content: str) -> List[str]:
        """Find all field labels in content"""
        fields = []
        
        # Common field patterns
        field_patterns = [
            r'([A-Za-z][A-Za-z\s]{2,30})[\s:]+(?=\S)',  # Label followed by value
            r'([A-Za-z][A-Za-z\s]{2,30})\s*\|',  # Table header
        ]
        
        for pattern in field_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                field = match.strip()
                if len(field) > 3 and not field.isupper():
                    fields.append(field)
        
        return list(set(fields))[:20]  # Limit to 20 unique fields
    
    def _guess_section_type(self, content: str) -> FormSection:
        """Guess section type from content"""
        scores = {}
        
        for section, patterns in self.compiled_field_patterns.items():
            score = sum(1 for p in patterns if p.search(content))
            scores[section] = score
        
        if scores:
            best = max(scores, key=scores.get)
            if scores[best] > 0:
                return best
        
        return FormSection.UNKNOWN
    
    def _calculate_section_confidence(self, section_type: FormSection, 
                                       content: str, fields_found: List[str]) -> float:
        """Calculate confidence score for section identification"""
        confidence = 0.5  # Base confidence
        
        # Check if expected fields are present
        if section_type in self.compiled_field_patterns:
            patterns = self.compiled_field_patterns[section_type]
            matches = sum(1 for p in patterns if p.search(content))
            confidence += min(0.4, matches * 0.1)
        
        # Bonus for having multiple fields
        if len(fields_found) > 5:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _identify_form_type(self, text: str) -> str:
        """Identify the type of claim form"""
        form_types = {
            'Cashless Request Form': [r'cashless', r'pre[-\s]?auth'],
            'Reimbursement Claim Form': [r'reimbursement', r'claim\s+form'],
            'Health Insurance Claim': [r'health\s+insurance', r'mediclaim'],
            'Hospital Bill': [r'hospital\s+bill', r'discharge\s+summary'],
            'Discharge Summary': [r'discharge\s+summary', r'final\s+bill'],
        }
        
        text_lower = text.lower()
        
        for form_type, patterns in form_types.items():
            if any(re.search(p, text_lower) for p in patterns):
                return form_type
        
        return "Insurance Claim Form"
    
    def _calculate_parse_confidence(self, sections: Dict[str, SectionInfo]) -> float:
        """Calculate overall parsing confidence"""
        if not sections:
            return 0.0
        
        # Average section confidences
        avg_confidence = sum(s.confidence for s in sections.values()) / len(sections)
        
        # Bonus for finding multiple standard sections
        standard_sections = [FormSection.PART_A, FormSection.PART_B, FormSection.PART_C]
        found_standard = sum(1 for s in sections.values() if s.section_type in standard_sections)
        bonus = min(0.2, found_standard * 0.05)
        
        return min(1.0, avg_confidence + bonus)
    
    def extract_section_data(self, section: SectionInfo) -> Dict[str, Any]:
        """
        Extract structured data from a section based on its type
        """
        extractors = {
            FormSection.PART_A: self._extract_insured_data,
            FormSection.PART_B: self._extract_hospital_data,
            FormSection.PART_C: self._extract_claim_data,
            FormSection.PART_D: self._extract_medical_data,
            FormSection.PART_E: self._extract_declaration_data,
        }
        
        extractor = extractors.get(section.section_type, self._extract_generic_data)
        return extractor(section.content)
    
    def _extract_insured_data(self, content: str) -> Dict[str, Any]:
        """Extract insured/patient details from Part A"""
        patterns = {
            'policy_number': [
                r'policy\s*(?:no\.?|number)[\s:]+([A-Z0-9/-]+)',
                r'policy[\s:]+([A-Z0-9/-]+)',
            ],
            'member_name': [
                r'name\s+of\s+(?:the\s+)?(?:insured|patient|member)[\s:]+([A-Za-z\s]+)',
                r'(?:patient|member|insured)\s+name[\s:]+([A-Za-z\s]+)',
                r'name[\s:]+([A-Za-z][A-Za-z\s]{2,40})',
            ],
            'date_of_birth': [
                r'(?:date\s+of\s+birth|d\.?o\.?b\.?)[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                r'(?:dob|birth\s+date)[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            ],
            'age': [
                r'age[\s:]+(\d{1,3})\s*(?:years?|yrs?)?',
            ],
            'gender': [
                r'(?:gender|sex)[\s:]+([MmFf](?:ale)?|[Oo]ther)',
            ],
            'member_id': [
                r'member\s*(?:id|no\.?)[\s:]+([A-Z0-9]+)',
                r'(?:uhid|mrn)[\s:]+([A-Z0-9]+)',
            ],
            'contact': [
                r'(?:contact|mobile|phone)\s*(?:no\.?|number)?[\s:]+(\d{10})',
            ],
            'email': [
                r'email[\s:]+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            ],
            'address': [
                r'address[\s:]+([A-Za-z0-9,.\s-]+?)(?=\n|$)',
            ],
        }
        
        return self._apply_patterns(content, patterns)
    
    def _extract_hospital_data(self, content: str) -> Dict[str, Any]:
        """Extract hospital/provider details from Part B"""
        patterns = {
            'hospital_name': [
                r'hospital\s+name[\s:]+([A-Za-z\s&.]+)',
                r'name\s+of\s+(?:the\s+)?hospital[\s:]+([A-Za-z\s&.]+)',
            ],
            'hospital_address': [
                r'(?:hospital\s+)?address[\s:]+([A-Za-z0-9,.\s-]+?)(?=\n|$)',
            ],
            'admission_date': [
                r'date\s+of\s+admission[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                r'(?:admitted|admission)\s+(?:on|date)[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            ],
            'discharge_date': [
                r'date\s+of\s+discharge[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
                r'(?:discharged?|discharge)\s+(?:on|date)[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            ],
            'room_type': [
                r'room\s+(?:type|category)[\s:]+([A-Za-z\s]+)',
                r'(?:type\s+of\s+)?(?:room|accommodation)[\s:]+([A-Za-z\s]+)',
            ],
            'treating_doctor': [
                r'(?:treating|attending)\s+(?:doctor|physician)[\s:]+(?:Dr\.?\s*)?([A-Za-z\s.]+)',
                r"doctor'?s?\s+name[\s:]+(?:Dr\.?\s*)?([A-Za-z\s.]+)",
            ],
            'registration_number': [
                r'(?:registration|rohini|hospital)\s*(?:no\.?|number)[\s:]+([A-Z0-9]+)',
            ],
            'ipd_number': [
                r'(?:ipd|opd)\s*(?:no\.?|number)[\s:]+([A-Z0-9]+)',
            ],
        }
        
        return self._apply_patterns(content, patterns)
    
    def _extract_claim_data(self, content: str) -> Dict[str, Any]:
        """Extract claim/billing details from Part C"""
        patterns = {
            'total_amount': [
                r'(?:total|gross|net)\s+(?:amount|bill)[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)',
                r'(?:grand\s+)?total[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            'room_charges': [
                r'room\s+(?:rent|charges?)[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            'icu_charges': [
                r'(?:icu|iccu)\s+charges?[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            'surgery_charges': [
                r'(?:ot|operation|surgery)\s+charges?[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            'medicine_charges': [
                r'(?:medicine|pharmacy|drug)\s+(?:charges?|cost)[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            'lab_charges': [
                r'(?:lab|laboratory|pathology)\s+charges?[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            'consultation_fee': [
                r'(?:consultation|doctor)\s+(?:fee|charges?)[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
            'discount': [
                r'discount[\s:]+(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)',
            ],
        }
        
        return self._apply_patterns(content, patterns)
    
    def _extract_medical_data(self, content: str) -> Dict[str, Any]:
        """Extract medical/treatment details from Part D"""
        patterns = {
            'diagnosis': [
                r'diagnosis[\s:]+([A-Za-z\s,.-]+?)(?=\n|icd|$)',
                r'nature\s+of\s+illness[\s:]+([A-Za-z\s,.-]+)',
            ],
            'icd_codes': [
                r'(?:icd|icd-?10)[\s:]*([A-Z]\d{2}(?:\.\d+)?(?:\s*,\s*[A-Z]\d{2}(?:\.\d+)?)*)',
            ],
            'procedure_code': [
                r'(?:procedure|cpt)\s*(?:code)?[\s:]+(\d{5})',
            ],
            'treatment': [
                r'treatment\s+(?:given|details?)[\s:]+([A-Za-z\s,.-]+)',
            ],
            'surgery_performed': [
                r'(?:surgery|operation)\s+(?:performed|done)[\s:]+([A-Za-z\s,.-]+)',
            ],
            'duration_of_illness': [
                r'(?:duration|period)\s+of\s+(?:illness|symptoms?)[\s:]+([A-Za-z0-9\s]+)',
            ],
        }
        
        return self._apply_patterns(content, patterns)
    
    def _extract_declaration_data(self, content: str) -> Dict[str, Any]:
        """Extract declaration/signature details from Part E"""
        patterns = {
            'declaration_date': [
                r'date[\s:]+(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            ],
            'place': [
                r'place[\s:]+([A-Za-z\s]+)',
            ],
            'has_signature': [
                r'(signature)',
            ],
        }
        
        result = self._apply_patterns(content, patterns)
        result['has_declaration'] = bool(re.search(r'declare|authorize|consent', content, re.IGNORECASE))
        return result
    
    def _extract_generic_data(self, content: str) -> Dict[str, Any]:
        """Generic extraction for unknown sections"""
        # Extract any amounts found
        amounts = re.findall(r'(?:Rs\.?|INR)\s*([\d,]+\.?\d*)', content)
        
        # Extract any dates found
        dates = re.findall(r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', content)
        
        return {
            'amounts_found': [float(a.replace(',', '')) for a in amounts if a],
            'dates_found': dates,
        }
    
    def _apply_patterns(self, content: str, patterns: Dict[str, List[str]]) -> Dict[str, Any]:
        """Apply extraction patterns to content"""
        result = {}
        
        for field_name, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    value = re.sub(r'\s+', ' ', value)
                    value = value.strip(' .,:-')
                    if value:
                        result[field_name] = value
                        break
        
        return result


def parse_claim_form(text: str) -> Dict[str, Any]:
    """
    Convenience function to parse a claim form
    
    Args:
        text: OCR text from claim document
        
    Returns:
        Dictionary with parsed form data
    """
    parser = FormSectionParser()
    parsed = parser.parse(text)
    
    # Extract data from each section
    extracted_data = {}
    for section_key, section_info in parsed.sections.items():
        section_data = parser.extract_section_data(section_info)
        extracted_data[section_key] = {
            'type': section_info.section_type.value,
            'confidence': section_info.confidence,
            'data': section_data
        }
    
    return {
        'form_type': parsed.form_type,
        'parse_confidence': parsed.parse_confidence,
        'sections': extracted_data,
        'section_count': len(parsed.sections)
    }


if __name__ == "__main__":
    # Test with sample form
    sample_text = """
    HEALTH INSURANCE CLAIM FORM
    
    PART - A: DETAILS OF PRIMARY INSURED
    
    Policy No.: HDFC/2024/12345
    Name of Insured: RAHUL SHARMA
    Date of Birth: 15/06/1985
    Age: 39 Years
    Gender: Male
    Member ID: MEM123456
    Contact No.: 9876543210
    Address: 123, ABC Colony, Mumbai - 400001
    
    PART - B: DETAILS OF HOSPITALIZATION
    
    Hospital Name: Apollo Hospitals
    Hospital Address: Navi Mumbai
    Date of Admission: 20/12/2024
    Date of Discharge: 25/12/2024
    Room Type: Private
    Treating Doctor: Dr. Amit Kumar
    IPD No.: IPD/2024/5678
    
    PART - C: DETAILS OF CLAIM
    
    Room Charges: Rs. 25,000
    ICU Charges: Rs. 40,000
    Surgery Charges: Rs. 75,000
    Medicine: Rs. 15,000
    Lab Tests: Rs. 8,000
    Consultation Fee: Rs. 5,000
    
    Total Amount: Rs. 168,000
    Discount: Rs. 8,000
    Net Payable: Rs. 160,000
    
    PART - D: MEDICAL DETAILS
    
    Diagnosis: Acute Appendicitis
    ICD Code: K35.80
    Procedure: Appendectomy
    Duration of Illness: 3 days
    
    PART - E: DECLARATION
    
    I hereby declare that the information given above is true.
    
    Date: 26/12/2024
    Place: Mumbai
    Signature: _____________
    """
    
    result = parse_claim_form(sample_text)
    
    import json
    print(json.dumps(result, indent=2))
