"""
Form Layout Detection Module
Detects form structure: boxes, lines, labels, and field regions.
Crops each field region for specialized OCR processing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FieldType(Enum):
    """Types of form fields"""
    TEXT_PRINTED = "text_printed"
    TEXT_HANDWRITTEN = "text_handwritten"
    CHECKBOX = "checkbox"
    SIGNATURE = "signature"
    TABLE_CELL = "table_cell"
    LABEL = "label"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Represents a bounding box region"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def contains(self, other: 'BoundingBox') -> bool:
        return (self.x <= other.x and self.y <= other.y and 
                self.x2 >= other.x2 and self.y2 >= other.y2)
    
    def overlaps(self, other: 'BoundingBox', threshold: float = 0.5) -> bool:
        """Check if boxes overlap by more than threshold"""
        x_overlap = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        y_overlap = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        intersection = x_overlap * y_overlap
        min_area = min(self.area, other.area)
        return intersection / min_area > threshold if min_area > 0 else False
    
    def crop(self, image: np.ndarray, padding: int = 2) -> np.ndarray:
        """Crop this region from an image"""
        h, w = image.shape[:2]
        x1 = max(0, self.x - padding)
        y1 = max(0, self.y - padding)
        x2 = min(w, self.x2 + padding)
        y2 = min(h, self.y2 + padding)
        return image[y1:y2, x1:x2]


@dataclass
class FormField:
    """Represents a detected form field"""
    bbox: BoundingBox
    field_type: FieldType
    label: Optional[str] = None
    confidence: float = 0.0
    content_image: Optional[np.ndarray] = None
    extracted_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormLayout:
    """Represents the complete form layout"""
    fields: List[FormField]
    horizontal_lines: List[Tuple[int, int, int, int]]  # (x1, y1, x2, y2)
    vertical_lines: List[Tuple[int, int, int, int]]
    table_cells: List[BoundingBox]
    checkboxes: List[FormField]
    text_regions: List[FormField]
    image_shape: Tuple[int, int]


class FormLayoutDetector:
    """
    Detects form structure including:
    - Horizontal and vertical lines
    - Table cells and grid structures
    - Checkboxes (filled and empty)
    - Text regions (printed vs handwritten)
    - Labels and field areas
    """
    
    def __init__(self, 
                 min_line_length: int = 50,
                 checkbox_size_range: Tuple[int, int] = (15, 50),
                 min_text_height: int = 10,
                 max_text_height: int = 100):
        self.min_line_length = min_line_length
        self.checkbox_size_range = checkbox_size_range
        self.min_text_height = min_text_height
        self.max_text_height = max_text_height
    
    def detect_lines(self, image: np.ndarray) -> Tuple[List, List]:
        """Detect horizontal and vertical lines in the image"""
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarize
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        h, w = binary.shape
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Extract line coordinates using HoughLinesP
        horizontal_lines = []
        vertical_lines = []
        
        # Horizontal lines
        h_lines = cv2.HoughLinesP(horizontal, 1, np.pi/180, 50, 
                                   minLineLength=self.min_line_length, maxLineGap=10)
        if h_lines is not None:
            for line in h_lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) < 10:  # Nearly horizontal
                    horizontal_lines.append((x1, y1, x2, y2))
        
        # Vertical lines
        v_lines = cv2.HoughLinesP(vertical, 1, np.pi/180, 50,
                                   minLineLength=self.min_line_length, maxLineGap=10)
        if v_lines is not None:
            for line in v_lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 10:  # Nearly vertical
                    vertical_lines.append((x1, y1, x2, y2))
        
        logger.debug(f"Detected {len(horizontal_lines)} horizontal, {len(vertical_lines)} vertical lines")
        return horizontal_lines, vertical_lines
    
    def detect_table_cells(self, image: np.ndarray, 
                           horizontal_lines: List, 
                           vertical_lines: List) -> List[BoundingBox]:
        """Detect table cells from line intersections"""
        if not horizontal_lines or not vertical_lines:
            return []
        
        # Sort lines
        h_lines = sorted(horizontal_lines, key=lambda l: l[1])
        v_lines = sorted(vertical_lines, key=lambda l: l[0])
        
        cells = []
        
        # Find cells from line intersections
        for i in range(len(h_lines) - 1):
            for j in range(len(v_lines) - 1):
                y1 = h_lines[i][1]
                y2 = h_lines[i + 1][1]
                x1 = v_lines[j][0]
                x2 = v_lines[j + 1][0]
                
                # Filter reasonable cell sizes
                if 20 < (x2 - x1) < image.shape[1] // 2 and 10 < (y2 - y1) < image.shape[0] // 4:
                    cells.append(BoundingBox(x1, y1, x2 - x1, y2 - y1))
        
        logger.debug(f"Detected {len(cells)} table cells")
        return cells
    
    def detect_checkboxes(self, image: np.ndarray) -> List[FormField]:
        """Detect checkboxes (squares) in the image"""
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarize
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        checkboxes = []
        min_size, max_size = self.checkbox_size_range
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if approximately square and in size range
            aspect_ratio = w / h if h > 0 else 0
            if (0.7 < aspect_ratio < 1.3 and 
                min_size < w < max_size and 
                min_size < h < max_size):
                
                # Approximate contour to check if it's rectangular
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
                
                if len(approx) == 4:  # Rectangle/square
                    bbox = BoundingBox(x, y, w, h)
                    
                    # Determine if checked (filled)
                    roi = gray[y:y+h, x:x+w]
                    fill_ratio = np.sum(roi < 127) / roi.size if roi.size > 0 else 0
                    
                    is_checked = fill_ratio > 0.3
                    
                    field = FormField(
                        bbox=bbox,
                        field_type=FieldType.CHECKBOX,
                        confidence=0.8,
                        metadata={'is_checked': is_checked, 'fill_ratio': fill_ratio}
                    )
                    checkboxes.append(field)
        
        logger.debug(f"Detected {len(checkboxes)} checkboxes")
        return checkboxes
    
    def detect_text_regions(self, image: np.ndarray) -> List[FormField]:
        """Detect text regions using MSER or connected components"""
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Binarize
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Dilate to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if (self.min_text_height < h < self.max_text_height and 
                w > 20 and w / h > 0.5):  # Text is usually wider than tall
                
                bbox = BoundingBox(x, y, w, h)
                
                # Analyze region to determine if printed or handwritten
                roi = gray[y:y+h, x:x+w]
                field_type = self._classify_text_type(roi)
                
                field = FormField(
                    bbox=bbox,
                    field_type=field_type,
                    confidence=0.7
                )
                text_regions.append(field)
        
        # Sort by position (top to bottom, left to right)
        text_regions.sort(key=lambda f: (f.bbox.y // 20, f.bbox.x))
        
        logger.debug(f"Detected {len(text_regions)} text regions")
        return text_regions
    
    def _classify_text_type(self, roi: np.ndarray) -> FieldType:
        """Classify text region as printed or handwritten"""
        if roi.size == 0:
            return FieldType.UNKNOWN
        
        # Calculate edge density and regularity
        edges = cv2.Canny(roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Calculate horizontal projection variance (printed text is more regular)
        h_projection = np.sum(roi < 127, axis=1)
        h_variance = np.var(h_projection) if len(h_projection) > 0 else 0
        
        # Normalize variance by region height
        normalized_variance = h_variance / (roi.shape[0] ** 2) if roi.shape[0] > 0 else 0
        
        # Handwritten text typically has higher variance and more irregular edges
        if edge_density > 0.15 and normalized_variance > 0.1:
            return FieldType.TEXT_HANDWRITTEN
        else:
            return FieldType.TEXT_PRINTED
    
    def detect_labels(self, image: np.ndarray, text_regions: List[FormField]) -> List[FormField]:
        """Identify label regions (typically printed text before fields)"""
        labels = []
        
        for region in text_regions:
            # Labels are typically:
            # 1. Printed text
            # 2. Followed by a colon or located at the start of a line
            # 3. Relatively short
            
            if region.field_type == FieldType.TEXT_PRINTED:
                # Check if this could be a label based on position and size
                if region.bbox.width < image.shape[1] // 3:  # Labels are usually short
                    region.field_type = FieldType.LABEL
                    labels.append(region)
        
        return labels
    
    def analyze_form(self, image: np.ndarray) -> FormLayout:
        """
        Complete form layout analysis.
        
        Returns FormLayout with all detected elements.
        """
        logger.info(f"Analyzing form layout: {image.shape}")
        
        # Detect lines
        h_lines, v_lines = self.detect_lines(image)
        
        # Detect table cells
        table_cells = self.detect_table_cells(image, h_lines, v_lines)
        
        # Detect checkboxes
        checkboxes = self.detect_checkboxes(image)
        
        # Detect text regions
        text_regions = self.detect_text_regions(image)
        
        # Filter out regions that overlap with checkboxes
        filtered_text_regions = []
        for region in text_regions:
            is_checkbox = any(region.bbox.overlaps(cb.bbox, 0.7) for cb in checkboxes)
            if not is_checkbox:
                filtered_text_regions.append(region)
        
        # Identify labels
        labels = self.detect_labels(image, filtered_text_regions)
        
        # Combine all fields
        all_fields = checkboxes + filtered_text_regions
        
        layout = FormLayout(
            fields=all_fields,
            horizontal_lines=h_lines,
            vertical_lines=v_lines,
            table_cells=table_cells,
            checkboxes=checkboxes,
            text_regions=filtered_text_regions,
            image_shape=image.shape[:2]
        )
        
        logger.info(f"Form analysis complete: {len(checkboxes)} checkboxes, "
                   f"{len(filtered_text_regions)} text regions, "
                   f"{len(table_cells)} table cells")
        
        return layout
    
    def crop_field_regions(self, image: np.ndarray, layout: FormLayout, 
                           padding: int = 5) -> List[FormField]:
        """Crop each field region from the image"""
        for field in layout.fields:
            field.content_image = field.bbox.crop(image, padding)
        
        return layout.fields
    
    def visualize_layout(self, image: np.ndarray, layout: FormLayout) -> np.ndarray:
        """Draw detected layout on image for visualization"""
        # Convert to color if grayscale
        if len(image.shape) == 2:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis_image = image.copy()
        
        # Draw horizontal lines (blue)
        for x1, y1, x2, y2 in layout.horizontal_lines:
            cv2.line(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Draw vertical lines (green)
        for x1, y1, x2, y2 in layout.vertical_lines:
            cv2.line(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw checkboxes (red)
        for checkbox in layout.checkboxes:
            color = (0, 0, 255) if checkbox.metadata.get('is_checked') else (0, 128, 255)
            cv2.rectangle(vis_image, 
                         (checkbox.bbox.x, checkbox.bbox.y),
                         (checkbox.bbox.x2, checkbox.bbox.y2),
                         color, 2)
        
        # Draw text regions
        for region in layout.text_regions:
            if region.field_type == FieldType.TEXT_PRINTED:
                color = (255, 255, 0)  # Cyan for printed
            elif region.field_type == FieldType.TEXT_HANDWRITTEN:
                color = (255, 0, 255)  # Magenta for handwritten
            elif region.field_type == FieldType.LABEL:
                color = (0, 255, 255)  # Yellow for labels
            else:
                color = (128, 128, 128)  # Gray for unknown
            
            cv2.rectangle(vis_image,
                         (region.bbox.x, region.bbox.y),
                         (region.bbox.x2, region.bbox.y2),
                         color, 1)
        
        return vis_image


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python layout_detection.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        sys.exit(1)
    
    detector = FormLayoutDetector()
    layout = detector.analyze_form(image)
    
    # Visualize
    vis = detector.visualize_layout(image, layout)
    output_path = image_path.rsplit('.', 1)[0] + "_layout.png"
    cv2.imwrite(output_path, vis)
    
    print(f"Layout visualization saved to: {output_path}")
    print(f"Detected: {len(layout.checkboxes)} checkboxes, "
          f"{len(layout.text_regions)} text regions")
