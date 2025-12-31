"""
Advanced OCR Preprocessing Pipeline
PDF → images at 300–400 dpi → grayscale → blur + adaptive threshold → deskew → border removal → small-blob noise removal
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
from dataclasses import dataclass
from pdf2image import convert_from_path
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing"""
    dpi: int = 350  # 300-400 dpi range
    blur_kernel: Tuple[int, int] = (3, 3)
    adaptive_block_size: int = 31  # Must be odd
    adaptive_c: int = 10
    min_blob_area: int = 50  # Minimum blob area to keep
    border_margin: int = 10  # Pixels to remove from borders
    deskew_angle_threshold: float = 45.0  # Max angle for deskew


class ImagePreprocessor:
    """
    Advanced image preprocessing for OCR optimization.
    Implements: grayscale → blur → adaptive threshold → deskew → border removal → noise removal
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
    
    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """Convert PDF to high-resolution images"""
        logger.info(f"Converting PDF to images at {self.config.dpi} DPI: {pdf_path}")
        
        try:
            # Convert PDF pages to PIL images
            pil_images = convert_from_path(
                pdf_path,
                dpi=self.config.dpi,
                fmt='png'
            )
            
            # Convert to OpenCV format (numpy arrays)
            images = []
            for i, pil_img in enumerate(pil_images):
                # Convert PIL to numpy array (RGB)
                img_array = np.array(pil_img)
                # Convert RGB to BGR for OpenCV
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                images.append(img_array)
                logger.debug(f"Page {i+1}: {img_array.shape}")
            
            logger.info(f"Converted {len(images)} pages")
            return images
            
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise
    
    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def apply_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to reduce noise"""
        return cv2.GaussianBlur(image, self.config.blur_kernel, 0)
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better text extraction"""
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_block_size,
            self.config.adaptive_c
        )
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew using Hough transform"""
        # Find lines in the image
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 
            threshold=100, 
            minLineLength=100, 
            maxLineGap=10
        )
        
        if lines is None:
            logger.debug("No lines detected for deskew")
            return image
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:  # Avoid division by zero
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines
                if abs(angle) < self.config.deskew_angle_threshold:
                    angles.append(angle)
        
        if not angles:
            logger.debug("No suitable angles found for deskew")
            return image
        
        # Use median angle to avoid outliers
        median_angle = np.median(angles)
        
        if abs(median_angle) < 0.5:  # Skip if nearly straight
            return image
        
        logger.debug(f"Deskewing by {median_angle:.2f} degrees")
        
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new image size to avoid cropping
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255 if len(image.shape) == 2 else (255, 255, 255)
        )
        
        return rotated
    
    def remove_borders(self, image: np.ndarray) -> np.ndarray:
        """Remove dark borders from scanned documents"""
        # Find contours
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Invert for contour detection
        inverted = cv2.bitwise_not(gray)
        
        # Find the largest contour (document boundary)
        contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add margin
        margin = self.config.border_margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Crop the image
        if len(image.shape) == 3:
            return image[y:y+h, x:x+w]
        else:
            return image[y:y+h, x:x+w]
    
    def remove_noise_blobs(self, image: np.ndarray) -> np.ndarray:
        """Remove small noise blobs using connected components"""
        # Ensure binary image
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        else:
            binary = image.copy()
        
        # Invert for blob detection (text is black)
        inverted = cv2.bitwise_not(binary)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )
        
        # Create output image
        output = np.ones_like(binary) * 255
        
        # Keep only components larger than minimum area
        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.config.min_blob_area:
                output[labels == i] = 0
        
        return output
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def preprocess_full(self, image: np.ndarray, 
                        apply_threshold: bool = True,
                        return_intermediate: bool = False) -> Dict[str, np.ndarray]:
        """
        Apply full preprocessing pipeline.
        
        Steps:
        1. Grayscale conversion
        2. Contrast enhancement
        3. Gaussian blur
        4. Adaptive thresholding (optional)
        5. Deskew
        6. Border removal
        7. Noise blob removal
        
        Returns dict with 'final' and optionally intermediate results.
        """
        results = {'original': image.copy()}
        
        # Step 1: Grayscale
        gray = self.to_grayscale(image)
        if return_intermediate:
            results['grayscale'] = gray.copy()
        
        # Step 2: Contrast enhancement
        enhanced = self.enhance_contrast(gray)
        if return_intermediate:
            results['enhanced'] = enhanced.copy()
        
        # Step 3: Blur
        blurred = self.apply_blur(enhanced)
        if return_intermediate:
            results['blurred'] = blurred.copy()
        
        # Step 4: Adaptive threshold (optional)
        if apply_threshold:
            thresholded = self.adaptive_threshold(blurred)
            if return_intermediate:
                results['thresholded'] = thresholded.copy()
            current = thresholded
        else:
            current = blurred
        
        # Step 5: Deskew
        deskewed = self.deskew(current)
        if return_intermediate:
            results['deskewed'] = deskewed.copy()
        
        # Step 6: Border removal
        cropped = self.remove_borders(deskewed)
        if return_intermediate:
            results['cropped'] = cropped.copy()
        
        # Step 7: Noise removal (only for thresholded images)
        if apply_threshold:
            cleaned = self.remove_noise_blobs(cropped)
            results['final'] = cleaned
        else:
            results['final'] = cropped
        
        return results
    
    def preprocess_pdf(self, pdf_path: str, 
                       output_dir: Optional[str] = None,
                       save_intermediate: bool = False) -> List[Dict[str, np.ndarray]]:
        """
        Preprocess entire PDF document.
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save processed images (optional)
            save_intermediate: Whether to save intermediate processing steps
        
        Returns:
            List of preprocessing results for each page
        """
        logger.info(f"Preprocessing PDF: {pdf_path}")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        
        results = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")
            
            # Apply full preprocessing
            page_results = self.preprocess_full(
                image, 
                apply_threshold=True,
                return_intermediate=save_intermediate
            )
            results.append(page_results)
            
            # Save if output directory specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save final result
                cv2.imwrite(
                    str(output_path / f"page_{i+1:03d}_processed.png"),
                    page_results['final']
                )
                
                # Save intermediate results if requested
                if save_intermediate:
                    for step_name, step_image in page_results.items():
                        if step_name != 'final':
                            cv2.imwrite(
                                str(output_path / f"page_{i+1:03d}_{step_name}.png"),
                                step_image
                            )
        
        logger.info(f"Preprocessing complete: {len(results)} pages processed")
        return results


def preprocess_for_ocr(image: np.ndarray, config: Optional[PreprocessingConfig] = None) -> np.ndarray:
    """
    Convenience function to preprocess a single image for OCR.
    Returns the final preprocessed image.
    """
    preprocessor = ImagePreprocessor(config)
    results = preprocessor.preprocess_full(image, apply_threshold=True)
    return results['final']


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <pdf_path> [output_dir]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "preprocessed_output"
    
    preprocessor = ImagePreprocessor()
    results = preprocessor.preprocess_pdf(pdf_path, output_dir, save_intermediate=True)
    
    print(f"Processed {len(results)} pages. Output saved to: {output_dir}")
