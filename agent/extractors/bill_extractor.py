# agent/extractors/bill_extractor.py
"""
Smart Bill Extraction Module
Extracts actual billing items from medical claim documents.
Filters out false positives like addresses, phone numbers, pin codes, etc.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum


class BillCategory(Enum):
    ROOM = "Room Charges"
    ICU = "ICU Charges"
    SURGERY = "Surgery"
    MEDICINE = "Medicine/Pharmacy"
    LAB = "Laboratory"
    RADIOLOGY = "Radiology/Imaging"
    CONSULTATION = "Consultation"
    NURSING = "Nursing Charges"
    PROCEDURE = "Procedure"
    CONSUMABLES = "Consumables"
    EQUIPMENT = "Equipment"
    AMBULANCE = "Ambulance"
    OTHER = "Other"


@dataclass
class BillItem:
    """Single billing line item"""
    description: str
    amount: float
    quantity: int = 1
    unit_price: float = 0.0
    code: str = ""
    category: BillCategory = BillCategory.OTHER
    line_number: int = 0
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'description': self.description,
            'amount': self.amount,
            'quantity': self.quantity,
            'unit_price': self.unit_price,
            'code': self.code,
            'category': self.category.value,
            'confidence': self.confidence
        }


@dataclass
class BillingSummary:
    """Summary of all billing information"""
    items: List[BillItem] = field(default_factory=list)
    subtotal: float = 0.0
    discount: float = 0.0
    tax: float = 0.0
    total: float = 0.0
    currency: str = "INR"
    extraction_confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'items': [item.to_dict() for item in self.items],
            'subtotal': self.subtotal,
            'discount': self.discount,
            'tax': self.tax,
            'total': self.total,
            'currency': self.currency,
            'item_count': len(self.items),
            'extraction_confidence': self.extraction_confidence
        }


class BillExtractor:
    """
    Smart bill extraction that identifies actual billing tables
    and filters out false positives like addresses, codes, etc.
    """
    
    # Keywords that indicate billing section
    BILLING_SECTION_KEYWORDS = [
        r'bill\s*(?:of\s*)?particulars',
        r'itemized\s*bill',
        r'hospital\s*bill',
        r'final\s*bill',
        r'invoice',
        r'statement\s*of\s*charges',
        r'billing\s*details',
        r'charges\s*details',
        r'particulars\s*of\s*charges',
        r'expense\s*details',
        r'cost\s*summary',
        r'payment\s*details',
        r'amount\s*payable',
    ]
    
    # Keywords that indicate billing line items
    BILLING_ITEM_KEYWORDS = [
        # Room charges
        r'room\s*(?:rent|charges?|tariff)',
        r'bed\s*charges?',
        r'accommodation',
        r'general\s*ward',
        r'private\s*room',
        r'semi[\-\s]?private',
        r'deluxe\s*room',
        r'suite',
        r'icu\s*(?:charges?)?',
        r'iccu',
        r'nicu',
        r'cccu',
        r'hdu',
        
        # Medical services
        r'consultation',
        r'doctor[\'\s]?s?\s*(?:fee|charges?|visit)',
        r'surgeon[\'\s]?s?\s*(?:fee|charges?)',
        r'anesthesia',
        r'anesthetist',
        r'operation\s*(?:theatre|charges?)',
        r'ot\s*charges?',
        r'surgery',
        r'procedure',
        
        # Medicines and consumables
        r'medicine',
        r'pharmacy',
        r'drugs?',
        r'injection',
        r'iv\s*fluids?',
        r'consumables?',
        r'disposables?',
        r'implant',
        r'stent',
        r'dressing',
        r'bandage',
        r'syringes?',
        
        # Diagnostics
        r'lab(?:oratory)?\s*(?:charges?|test)?',
        r'pathology',
        r'blood\s*test',
        r'urine\s*test',
        r'x[\-\s]?ray',
        r'ct\s*scan',
        r'mri',
        r'ultrasound',
        r'usg',
        r'sonography',
        r'ecg',
        r'ekg',
        r'echo(?:cardiogram)?',
        r'radiology',
        r'imaging',
        r'scan',
        
        # Nursing and care
        r'nursing',
        r'patient\s*care',
        r'monitoring',
        r'oxygen',
        r'ventilator',
        r'dialysis',
        r'physiotherapy',
        r'diet',
        r'food',
        r'meals?',
        
        # Other charges
        r'ambulance',
        r'registration',
        r'admission',
        r'discharge',
        r'documentation',
        r'certificate',
        r'equipment',
        r'service\s*charge',
        r'misc(?:ellaneous)?',
        r'other\s*charges?',
        r'total',
        r'sub[\-\s]?total',
        r'grand\s*total',
        r'net\s*(?:amount|payable)',
        r'gross\s*(?:amount|total)',
    ]
    
    # Patterns that are NOT billing items (false positives)
    FALSE_POSITIVE_PATTERNS = [
        # Address components
        r'shop\s*no\.?\s*\d+',
        r'plot\s*no\.?\s*\d+',
        r'flat\s*no\.?\s*\d+',
        r'block\s*[\w]?\s*\d+',
        r'sector\s*[-]?\s*\d+',
        r'floor\s*\d+',
        r'building\s*(?:no\.?)?\s*\d+',
        r'house\s*no\.?\s*\d+',
        
        # Pin codes (6 digits, 4xxxxx or 5xxxxx for India)
        r'\b[4-8]\d{5}\b',
        
        # Phone numbers (10+ digits)
        r'\b\d{10,}\b',
        r'\+91\s*\d{10}',
        
        # Bank account numbers
        r'a/?c\s*(?:no\.?)?\s*\d+',
        r'account\s*(?:no\.?|number)\s*\d+',
        
        # IFSC codes
        r'[A-Z]{4}0[A-Z0-9]{6}',
        
        # Policy/ID numbers context
        r'policy\s*(?:no\.?|number)',
        r'certificate\s*no\.?',
        r'member\s*(?:id|no\.?)',
        r'uhid',
        r'mrn',
        r'ipd\s*no\.?',
        r'opd\s*no\.?',
        
        # Date patterns that look like amounts
        r'\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',
        
        # Version numbers, page numbers
        r'page\s*\d+',
        r'v\d+\.\d+',
        r'version\s*\d+',
    ]
    
    # Amount patterns
    AMOUNT_PATTERNS = [
        # Indian format with Rs/₹
        r'(?:Rs\.?|₹|INR)\s*([\d,]+\.?\d*)',
        r'([\d,]+\.?\d*)\s*(?:Rs\.?|₹|INR)',
        
        # With commas (Indian: 1,00,000 or Western: 100,000)
        r'((?:\d{1,3},)*\d{1,3}(?:\.\d{2})?)',
        
        # Decimal amounts
        r'(\d+\.\d{2})',
        
        # Plain numbers (5+ digits likely amounts)
        r'(\d{5,})',
    ]
    
    def __init__(self):
        self.billing_section_re = re.compile(
            '|'.join(self.BILLING_SECTION_KEYWORDS), 
            re.IGNORECASE
        )
        self.billing_item_re = re.compile(
            '|'.join(self.BILLING_ITEM_KEYWORDS), 
            re.IGNORECASE
        )
        self.false_positive_re = re.compile(
            '|'.join(self.FALSE_POSITIVE_PATTERNS), 
            re.IGNORECASE
        )
    
    def extract_bills(self, text: str) -> BillingSummary:
        """
        Main extraction method - extracts all billing information from text
        
        Args:
            text: OCR text from claim document
            
        Returns:
            BillingSummary with all extracted billing items
        """
        # Step 1: Find billing section
        billing_text = self._find_billing_section(text)
        
        # Step 2: Extract line items
        items = self._extract_line_items(billing_text or text)
        
        # Step 3: Filter false positives
        items = self._filter_false_positives(items, text)
        
        # Step 4: Categorize items
        items = self._categorize_items(items)
        
        # Step 5: Extract totals
        totals = self._extract_totals(billing_text or text)
        
        # Step 6: Validate and calculate confidence
        items, confidence = self._validate_items(items, totals)
        
        # Build summary
        subtotal = sum(item.amount for item in items)
        
        return BillingSummary(
            items=items,
            subtotal=subtotal,
            discount=totals.get('discount', 0.0),
            tax=totals.get('tax', 0.0),
            total=totals.get('total', subtotal),
            extraction_confidence=confidence
        )
    
    def _find_billing_section(self, text: str) -> Optional[str]:
        """Find the billing/invoice section in the document"""
        lines = text.split('\n')
        
        # Look for billing section header
        billing_start = -1
        billing_end = len(lines)
        
        for i, line in enumerate(lines):
            if self.billing_section_re.search(line):
                billing_start = i
                break
        
        if billing_start == -1:
            return None
        
        # Find end of billing section (next major section or end)
        section_end_patterns = [
            r'part\s*[-–—]?\s*[b-e]',
            r'declaration',
            r'signature',
            r'authorization',
            r'terms\s*(?:and|&)\s*conditions',
        ]
        section_end_re = re.compile('|'.join(section_end_patterns), re.IGNORECASE)
        
        for i in range(billing_start + 1, len(lines)):
            if section_end_re.search(lines[i]):
                billing_end = i
                break
        
        return '\n'.join(lines[billing_start:billing_end])
    
    def _extract_line_items(self, text: str) -> List[BillItem]:
        """Extract individual billing line items from text"""
        items = []
        lines = text.split('\n')
        
        # Keywords that indicate a total/summary line (not an individual item)
        summary_keywords = [
            'total', 'sub-total', 'subtotal', 'grand total', 'net amount',
            'net payable', 'amount payable', 'gross amount', 'final bill',
            'sum total', 'bill amount', 'discount', 'tax', 'gst', 'cgst', 'sgst',
            'balance', 'paid', 'payment', 'advance', 'deposit'
        ]
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line or len(line) < 3:
                continue
            
            # Skip if line is a known false positive
            if self.false_positive_re.search(line):
                continue
            
            # Skip if this is a summary line (not an individual item)
            line_lower = line.lower()
            if any(kw in line_lower for kw in summary_keywords):
                continue
            
            # Check if line contains billing keywords
            has_billing_keyword = bool(self.billing_item_re.search(line))
            
            # Extract amounts from line
            amounts = self._extract_amounts_from_line(line)
            
            if amounts and (has_billing_keyword or self._looks_like_billing_line(line)):
                # Get description (text before the amount)
                description = self._extract_description(line, amounts[0])
                
                if description and len(description) > 2:
                    # Get quantity if present
                    qty, unit_price = self._extract_quantity(line, amounts[0])
                    
                    items.append(BillItem(
                        description=description,
                        amount=amounts[0],
                        quantity=qty,
                        unit_price=unit_price,
                        line_number=line_num,
                        confidence=0.8 if has_billing_keyword else 0.5
                    ))
        
        return items
    
    def _extract_amounts_from_line(self, line: str) -> List[float]:
        """Extract numeric amounts from a line"""
        amounts = []
        
        # First check for Rs/₹ prefixed amounts
        for pattern in self.AMOUNT_PATTERNS[:2]:
            matches = re.findall(pattern, line)
            for match in matches:
                try:
                    # Remove commas and convert
                    clean = match.replace(',', '')
                    amount = float(clean)
                    if 1 <= amount <= 50000000:  # Reasonable medical bill range
                        amounts.append(amount)
                except ValueError:
                    continue
        
        # If no prefixed amounts, look for standalone amounts
        if not amounts:
            # Look for amounts with 2 decimal places (common in bills)
            decimal_matches = re.findall(r'(\d{1,3}(?:,\d{2,3})*(?:,\d{3})*\.\d{2})\b', line)
            for match in decimal_matches:
                try:
                    clean = match.replace(',', '')
                    amount = float(clean)
                    if 10 <= amount <= 50000000:
                        amounts.append(amount)
                except ValueError:
                    continue
            
            # Look for large round numbers (likely amounts, not IDs)
            if not amounts:
                round_matches = re.findall(r'\b(\d{4,})\b', line)
                for match in round_matches:
                    try:
                        amount = float(match)
                        # Only consider if it's in a typical medical billing range
                        # and the line has billing context
                        if 1000 <= amount <= 50000000:
                            # Verify it's not a PIN code (6 digits starting with 4-8)
                            if not (len(match) == 6 and match[0] in '45678'):
                                amounts.append(amount)
                    except ValueError:
                        continue
        
        return amounts
    
    def _looks_like_billing_line(self, line: str) -> bool:
        """Check if a line looks like a billing entry based on structure"""
        # Should have both text and numbers
        has_text = bool(re.search(r'[a-zA-Z]{2,}', line))
        has_number = bool(re.search(r'\d{3,}', line))
        
        if not (has_text and has_number):
            return False
        
        # Should not be an address line
        address_keywords = ['road', 'street', 'lane', 'nagar', 'colony', 
                          'sector', 'plot', 'flat', 'floor', 'mumbai', 
                          'delhi', 'bangalore', 'chennai', 'kolkata']
        if any(kw in line.lower() for kw in address_keywords):
            return False
        
        # Should not be a header/title line
        if line.isupper() and len(line.split()) <= 3:
            return False
        
        return True
    
    def _extract_description(self, line: str, amount: float) -> str:
        """Extract item description from line"""
        # Remove the amount from line
        amount_str = str(amount)
        amount_patterns = [
            f'Rs\\.?\\s*{re.escape(amount_str)}',
            f'₹\\s*{re.escape(amount_str)}',
            re.escape(f'{amount:,.2f}'),
            re.escape(f'{amount:.2f}'),
            re.escape(f'{int(amount):,}'),
            re.escape(str(int(amount))),
        ]
        
        desc = line
        for pattern in amount_patterns:
            desc = re.sub(pattern, '', desc, flags=re.IGNORECASE)
        
        # Clean up
        desc = re.sub(r'[\|\[\]\(\)\{\}]', '', desc)
        desc = re.sub(r'\s+', ' ', desc)
        desc = desc.strip(' :-–—.')
        
        # Remove quantity patterns from end
        desc = re.sub(r'\s*\d+\s*(?:nos?\.?|units?|pcs?\.?|qty\.?)?\s*$', '', desc, flags=re.IGNORECASE)
        
        return desc.strip()
    
    def _extract_quantity(self, line: str, amount: float) -> Tuple[int, float]:
        """Extract quantity and calculate unit price"""
        # Look for quantity patterns
        qty_patterns = [
            r'(\d+)\s*(?:nos?\.?|units?|pcs?\.?|qty\.?)',
            r'qty[:\s]*(\d+)',
            r'x\s*(\d+)',
            r'(\d+)\s*@',
        ]
        
        for pattern in qty_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                qty = int(match.group(1))
                if qty > 0 and qty < 1000:  # Reasonable quantity
                    return qty, amount / qty
        
        return 1, amount
    
    def _filter_false_positives(self, items: List[BillItem], full_text: str) -> List[BillItem]:
        """Remove items that are likely false positives"""
        filtered = []
        
        # Collect known false values from address/header section
        false_values = set()
        
        # Extract pin codes
        pin_codes = re.findall(r'\b[4-8]\d{5}\b', full_text[:2000])
        for pin in pin_codes:
            false_values.add(float(pin))
        
        # Extract phone numbers
        phones = re.findall(r'\b\d{10}\b', full_text)
        for phone in phones:
            try:
                false_values.add(float(phone))
            except:
                pass
        
        # Extract address numbers
        addr_nums = re.findall(r'(?:shop|plot|flat|block|sector|floor)\s*(?:no\.?)?\s*(\d+)', 
                               full_text[:2000], re.IGNORECASE)
        for num in addr_nums:
            try:
                false_values.add(float(num))
            except:
                pass
        
        for item in items:
            # Skip if amount is a known false value
            if item.amount in false_values:
                continue
            
            # Skip very small amounts (likely not medical bills)
            if item.amount < 50:
                continue
            
            # Skip if description contains false positive indicators
            desc_lower = item.description.lower()
            if any(kw in desc_lower for kw in ['shop', 'plot', 'flat', 'sector', 'road', 'street']):
                continue
            
            # Skip if it's a policy/ID number context
            if any(kw in desc_lower for kw in ['policy', 'member', 'certificate', 'uhid', 'mrn']):
                continue
            
            filtered.append(item)
        
        return filtered
    
    def _categorize_items(self, items: List[BillItem]) -> List[BillItem]:
        """Assign categories to billing items"""
        category_keywords = {
            BillCategory.ROOM: ['room', 'bed', 'accommodation', 'ward', 'suite', 'deluxe'],
            BillCategory.ICU: ['icu', 'iccu', 'nicu', 'cccu', 'hdu', 'intensive'],
            BillCategory.SURGERY: ['surgery', 'operation', 'ot ', 'surgeon', 'procedure'],
            BillCategory.MEDICINE: ['medicine', 'pharmacy', 'drug', 'injection', 'tablet', 'syrup'],
            BillCategory.LAB: ['lab', 'pathology', 'blood', 'urine', 'test', 'culture'],
            BillCategory.RADIOLOGY: ['x-ray', 'xray', 'ct', 'mri', 'scan', 'ultrasound', 'usg', 'imaging'],
            BillCategory.CONSULTATION: ['consultation', 'doctor', 'visit', 'opinion'],
            BillCategory.NURSING: ['nursing', 'care', 'monitoring'],
            BillCategory.CONSUMABLES: ['consumable', 'disposable', 'dressing', 'bandage', 'syringe'],
            BillCategory.EQUIPMENT: ['equipment', 'ventilator', 'oxygen', 'dialysis'],
            BillCategory.AMBULANCE: ['ambulance', 'transport'],
        }
        
        for item in items:
            desc_lower = item.description.lower()
            for category, keywords in category_keywords.items():
                if any(kw in desc_lower for kw in keywords):
                    item.category = category
                    break
        
        return items
    
    def _extract_totals(self, text: str) -> Dict[str, float]:
        """Extract total, subtotal, discount, tax from text"""
        totals = {}
        
        # Total patterns
        total_patterns = [
            r'(?:grand\s*)?total[\s:]*(?:Rs\.?|₹|INR)?\s*([\d,]+\.?\d*)',
            r'(?:net|gross)\s*(?:amount|payable|total)[\s:]*(?:Rs\.?|₹|INR)?\s*([\d,]+\.?\d*)',
            r'amount\s*payable[\s:]*(?:Rs\.?|₹|INR)?\s*([\d,]+\.?\d*)',
            r'final\s*bill[\s:]*(?:Rs\.?|₹|INR)?\s*([\d,]+\.?\d*)',
        ]
        
        for pattern in total_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    totals['total'] = float(match.group(1).replace(',', ''))
                    break
                except ValueError:
                    continue
        
        # Discount
        discount_pattern = r'discount[\s:]*(?:Rs\.?|₹|INR)?\s*-?\s*([\d,]+\.?\d*)'
        match = re.search(discount_pattern, text, re.IGNORECASE)
        if match:
            try:
                totals['discount'] = float(match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        # Tax/GST
        tax_pattern = r'(?:tax|gst|cgst|sgst)[\s:]*(?:Rs\.?|₹|INR)?\s*([\d,]+\.?\d*)'
        match = re.search(tax_pattern, text, re.IGNORECASE)
        if match:
            try:
                totals['tax'] = float(match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        return totals
    
    def _validate_items(self, items: List[BillItem], totals: Dict[str, float]) -> Tuple[List[BillItem], float]:
        """Validate extracted items and calculate confidence"""
        if not items:
            return items, 0.0
        
        # Calculate item total
        item_total = sum(item.amount for item in items)
        
        # If we have a document total, validate against it
        doc_total = totals.get('total', 0)
        
        if doc_total > 0:
            # Check if itemized total is close to document total
            diff_pct = abs(item_total - doc_total) / doc_total if doc_total else 1.0
            
            if diff_pct < 0.05:  # Within 5%
                confidence = 0.95
            elif diff_pct < 0.15:  # Within 15%
                confidence = 0.80
            elif diff_pct < 0.30:  # Within 30%
                confidence = 0.60
            else:
                confidence = 0.40
        else:
            # No total to validate against
            # Base confidence on number of items with billing keywords
            keyword_items = sum(1 for item in items if item.confidence >= 0.8)
            confidence = min(0.7, 0.3 + (keyword_items / len(items)) * 0.4)
        
        return items, confidence


# Standalone function for easy use
def extract_bills_from_text(text: str) -> Dict[str, Any]:
    """
    Extract billing information from OCR text
    
    Args:
        text: OCR text from claim document
        
    Returns:
        Dictionary with billing summary
    """
    extractor = BillExtractor()
    summary = extractor.extract_bills(text)
    return summary.to_dict()


if __name__ == "__main__":
    # Test with sample text
    sample = """
    HOSPITAL BILL
    Patient: John Doe
    
    Room Charges (5 days)          Rs. 25,000.00
    ICU Charges (2 days)           Rs. 40,000.00
    Surgery - Appendectomy         Rs. 75,000.00
    Medicines                      Rs. 15,500.50
    Lab Tests                      Rs. 8,004.00
    X-Ray                          Rs. 1,775.51
    Consultation Fee               Rs. 6,566.16
    Nursing Charges                Rs. 5,000.00
    
    Sub Total                      Rs. 176,846.17
    Discount                       Rs. 10,000.00
    --------------------------------
    Grand Total                    Rs. 166,846.17
    """
    
    result = extract_bills_from_text(sample)
    import json
    print(json.dumps(result, indent=2))
