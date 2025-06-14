"""
OCR and Document Parsing Script
Handles PDF text extraction, OCR for scanned images, and initial document processing
Uses configuration from .env file
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import logging
from datetime import datetime
import cv2
import numpy as np

# Add project root to path and fix directory paths
project_root = Path(__file__).parent.parent  # Go up from src to project root
sys.path.append(str(project_root))

# Import settings (with fallback if not available)
try:
    from config.settings import settings
except ImportError:
    # Fallback configuration if settings not available
    class Settings:
        PDF_DIRECTORY = str(project_root / "docs")
        PROCESSED_DOCUMENTS_DIR = str(project_root / "data" / "processed_documents")
        DEBUG = True
        USE_OCR = True
        OCR_ENGINE = "tesseract"
        OCR_DPI = 300
        OCR_LANGUAGE = "eng"
        ENHANCE_IMAGE = True
        MIN_PARAGRAPH_LENGTH = 50
        MAX_PARAGRAPH_LENGTH = 2000
        MAX_RETRIES = 3
        
        def validate_config(self):
            return True
        
        def create_directories(self):
            Path(self.PROCESSED_DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
    
    settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        """Initialize the document processor with configuration"""
        # Fix paths to be relative to project root
        self.pdf_directory = Path(settings.PDF_DIRECTORY)
        if not self.pdf_directory.is_absolute():
            self.pdf_directory = project_root / self.pdf_directory
            
        self.output_dir = Path(settings.PROCESSED_DOCUMENTS_DIR)
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # OCR Configuration
        self.use_ocr = settings.USE_OCR
        self.ocr_engine = settings.OCR_ENGINE
        self.ocr_dpi = settings.OCR_DPI
        self.ocr_language = settings.OCR_LANGUAGE
        self.enhance_image = settings.ENHANCE_IMAGE
        self.min_paragraph_length = settings.MIN_PARAGRAPH_LENGTH
        self.max_paragraph_length = settings.MAX_PARAGRAPH_LENGTH
        
        # Supported image formats
        self.image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.gif'}
        
        # Configure Tesseract
        if os.name == 'nt':  # Windows
            tesseract_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
            ]
            for path in tesseract_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
        logger.info(f"DocumentProcessor initialized")
        logger.info(f"Project root: {project_root}")
        logger.info(f"PDF Directory: {self.pdf_directory}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info(f"OCR Enabled: {self.use_ocr}")
    
    def is_image_file(self, file_path: str) -> bool:
        """
        Check if the file is an image format
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is an image format
        """
        return Path(file_path).suffix.lower() in self.image_extensions
    
    def enhance_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            Enhanced PIL Image
        """
        if not self.enhance_image:
            return image
        
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            
            # Apply denoising using OpenCV
            cv_image = np.array(image)
            cv_image = cv2.fastNlMeansDenoising(cv_image)
            image = Image.fromarray(cv_image)
            
            # Apply slight blur to smooth out noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def extract_text_with_ocr(self, page) -> str:
        """
        Perform OCR on a PDF page with enhanced image processing
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            Extracted text from OCR
        """
        try:
            # Convert page to image with high DPI
            mat = fitz.Matrix(self.ocr_dpi / 72.0, self.ocr_dpi / 72.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(img_data))
            
            # Enhance image if enabled
            if self.enhance_image:
                image = self.enhance_image_for_ocr(image)
            
            # Configure OCR
            ocr_config = f'--oem 3 --psm 6 -l {self.ocr_language}'
            
            # Perform OCR
            if self.ocr_engine == "tesseract":
                text = pytesseract.image_to_string(image, config=ocr_config)
            else:
                # Fallback to basic OCR
                text = pytesseract.image_to_string(image, lang=self.ocr_language)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Perform OCR directly on image files
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text from OCR
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Enhance image if enabled
            if self.enhance_image:
                image = self.enhance_image_for_ocr(image)
            
            # Configure OCR
            ocr_config = f'--oem 3 --psm 6 -l {self.ocr_language}'
            
            # Perform OCR
            if self.ocr_engine == "tesseract":
                text = pytesseract.image_to_string(image, config=ocr_config)
            else:
                # Fallback to basic OCR
                text = pytesseract.image_to_string(image, lang=self.ocr_language)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Image OCR failed for {image_path}: {str(e)}")
            return ""
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs with length filtering
        
        Args:
            text: Raw text content
            
        Returns:
            List of filtered paragraphs
        """
        # Split by double newlines first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If no double newlines, split by single newlines and group
        if len(paragraphs) <= 1:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            paragraphs = []
            current_paragraph = ""
            
            for line in lines:
                if len(line) < 50 and current_paragraph:
                    current_paragraph += " " + line
                else:
                    if current_paragraph:
                        paragraphs.append(current_paragraph)
                    current_paragraph = line
            
            if current_paragraph:
                paragraphs.append(current_paragraph)
        
        # Filter paragraphs by length
        filtered_paragraphs = []
        for para in paragraphs:
            if len(para) >= self.min_paragraph_length:
                if len(para) <= self.max_paragraph_length:
                    filtered_paragraphs.append(para)
                else:
                    # Split long paragraphs at sentence boundaries
                    sentences = para.split('. ')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) <= self.max_paragraph_length:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                filtered_paragraphs.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                    
                    if current_chunk:
                        filtered_paragraphs.append(current_chunk.strip())
        
        return filtered_paragraphs
    
    def process_image_file(self, image_path: str) -> Optional[Dict[str, Any]]:
        """
        Process image files directly with OCR
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        doc_id = str(uuid.uuid4())
        doc_name = Path(image_path).stem
        
        try:
            logger.info(f"Processing image file with OCR: {doc_name}")
            
            # Extract text using OCR
            text = self.extract_text_from_image(image_path)
            
            # Split text into paragraphs
            paragraphs = self.split_into_paragraphs(text)
            
            # Create page data structure (treating image as single page)
            page_data = {
                "page_number": 1,
                "paragraphs": paragraphs,
                "paragraph_count": len(paragraphs),
                "raw_text": text,
                "text_length": len(text),
                "ocr_used": True  # Always true for image files
            }
            
            document_data = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "file_path": str(image_path),
                "file_size": os.path.getsize(image_path),
                "file_type": "image",
                "total_pages": 1,
                "total_paragraphs": len(paragraphs),
                "total_text_length": len(text),
                "pages": [page_data],
                "processing_config": {
                    "use_ocr": self.use_ocr,
                    "ocr_engine": self.ocr_engine,
                    "min_paragraph_length": self.min_paragraph_length,
                    "max_paragraph_length": self.max_paragraph_length,
                    "enhance_image": self.enhance_image
                },
                "processed_at": datetime.now().isoformat()
            }
            
            # Save to JSON file
            output_file = self.output_dir / f"{doc_name}_{doc_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed image {doc_name}: {len(paragraphs)} paragraphs extracted")
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract text from PDF using PyMuPDF with OCR fallback
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        doc_id = str(uuid.uuid4())
        doc_name = Path(pdf_path).stem
        
        try:
            pdf_document = fitz.open(pdf_path)
            pages_data = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Try to extract text directly
                text = page.get_text()
                
                # If no text found or very little text, use OCR
                if self.use_ocr and len(text.strip()) < 50:
                    logger.info(f"Using OCR for page {page_num + 1} of {doc_name}")
                    ocr_text = self.extract_text_with_ocr(page)
                    if len(ocr_text) > len(text):
                        text = ocr_text
                
                # Split text into paragraphs
                paragraphs = self.split_into_paragraphs(text)
                
                page_data = {
                    "page_number": page_num + 1,
                    "paragraphs": paragraphs,
                    "paragraph_count": len(paragraphs),
                    "raw_text": text,
                    "text_length": len(text),
                    "ocr_used": self.use_ocr and len(text.strip()) < 50
                }
                pages_data.append(page_data)
            
            pdf_document.close()
            
            total_paragraphs = sum(page["paragraph_count"] for page in pages_data)
            total_text_length = sum(page["text_length"] for page in pages_data)
            
            document_data = {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "file_path": str(pdf_path),
                "file_size": os.path.getsize(pdf_path),
                "file_type": "pdf",
                "total_pages": len(pages_data),
                "total_paragraphs": total_paragraphs,
                "total_text_length": total_text_length,
                "pages": pages_data,
                "processing_config": {
                    "use_ocr": self.use_ocr,
                    "ocr_engine": self.ocr_engine,
                    "min_paragraph_length": self.min_paragraph_length,
                    "max_paragraph_length": self.max_paragraph_length,
                    "enhance_image": self.enhance_image
                },
                "processed_at": datetime.now().isoformat()
            }
            
            # Save to JSON file
            output_file = self.output_dir / f"{doc_name}_{doc_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Processed {doc_name}: {len(pages_data)} pages, {total_paragraphs} paragraphs")
            return document_data
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return None
    
    def process_document_folder(self) -> List[Dict[str, Any]]:
        """
        Process all PDF and image documents in the configured folder
        
        Returns:
            List of processed document data
        """
        if not self.pdf_directory.exists():
            logger.error(f"Document directory not found: {self.pdf_directory}")
            return []
        
        # Get both PDF and image files
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        image_files = []
        for ext in self.image_extensions:
            image_files.extend(self.pdf_directory.glob(f"*{ext}"))
        
        all_files = pdf_files + image_files
        
        if not all_files:
            logger.warning(f"No PDF or image files found in {self.pdf_directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files and {len(image_files)} image files to process")
        
        processed_documents = []
        failed_documents = []
        
        for i, file_path in enumerate(all_files, 1):
            logger.info(f"Processing ({i}/{len(all_files)}): {file_path.name}")
            
            # Retry logic
            for attempt in range(settings.MAX_RETRIES):
                try:
                    # Process based on file type
                    if self.is_image_file(str(file_path)):
                        doc_data = self.process_image_file(str(file_path))
                    else:
                        doc_data = self.extract_text_from_pdf(str(file_path))
                    
                    if doc_data:
                        processed_documents.append(doc_data)
                        break
                    else:
                        if attempt == settings.MAX_RETRIES - 1:
                            failed_documents.append(str(file_path))
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {file_path.name}: {e}")
                    if attempt == settings.MAX_RETRIES - 1:
                        failed_documents.append(str(file_path))
        
        # Create comprehensive summary
        summary = {
            "total_files_found": len(all_files),
            "pdf_files": len(pdf_files),
            "image_files": len(image_files),
            "successfully_processed": len(processed_documents),
            "failed_processing": len(failed_documents),
            "success_rate": len(processed_documents) / len(all_files) * 100 if all_files else 0,
            "failed_files": failed_documents,
            "total_pages": sum(doc["total_pages"] for doc in processed_documents),
            "total_paragraphs": sum(doc["total_paragraphs"] for doc in processed_documents),
            "processing_config": {
                "use_ocr": self.use_ocr,
                "ocr_engine": self.ocr_engine,
                "min_paragraph_length": self.min_paragraph_length,
                "max_paragraph_length": self.max_paragraph_length
            },
            "processed_at": datetime.now().isoformat(),
            "documents": [
                {
                    "doc_id": doc["doc_id"],
                    "doc_name": doc["doc_name"],
                    "file_type": doc.get("file_type", "pdf"),
                    "pages": doc["total_pages"],
                    "paragraphs": doc["total_paragraphs"],
                    "file_size": doc["file_size"]
                }
                for doc in processed_documents
            ]
        }
        
        with open(self.output_dir / "processing_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully processed {len(processed_documents)}/{len(all_files)} documents")
        if failed_documents:
            logger.warning(f"Failed to process: {failed_documents}")
        
        return processed_documents

def main():
    """Main function to run the OCR and parsing process"""
    print(f"🔍 Project root: {project_root}")
    print(f"📁 Looking for documents in: {project_root / 'docs'}")
    
    # Validate configuration
    if hasattr(settings, 'validate_config') and not settings.validate_config():
        return
    
    # Create directories
    if hasattr(settings, 'create_directories'):
        settings.create_directories()
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Check if directory exists and has files
    if not processor.pdf_directory.exists():
        logger.error(f"Document directory not found: {processor.pdf_directory}")
        print(f"❌ Please create the directory and add PDF/image files: {processor.pdf_directory}")
        return
    
    # Get both PDF and image files
    pdf_files = list(processor.pdf_directory.glob("*.pdf"))
    image_files = []
    for ext in processor.image_extensions:
        image_files.extend(processor.pdf_directory.glob(f"*{ext}"))
    
    all_files = pdf_files + image_files
    
    if not all_files:
        logger.error(f"No PDF or image files found in {processor.pdf_directory}")
        print(f"❌ No PDF or image files found in: {processor.pdf_directory}")
        print(f"💡 Please add some PDF or image files to: {processor.pdf_directory}")
        return
    
    print(f"📁 Found {len(pdf_files)} PDF files and {len(image_files)} image files in {processor.pdf_directory}")
    
    # Process all documents
    processed_docs = processor.process_document_folder()
    
    if processed_docs:
        total_pages = sum(doc['total_pages'] for doc in processed_docs)
        total_paragraphs = sum(doc['total_paragraphs'] for doc in processed_docs)
        
        print(f"\n✅ Successfully processed {len(processed_docs)} documents")
        print(f"📄 Total pages: {total_pages}")
        print(f"📝 Total paragraphs: {total_paragraphs}")
        print(f"📁 Output saved to: {processor.output_dir}")
        print(f"📋 Summary saved to: {processor.output_dir}/processing_summary.json")
    else:
        print("❌ No documents were processed successfully")

if __name__ == "__main__":
    main()