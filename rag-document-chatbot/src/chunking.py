"""
Smart Document Chunking Script
Creates meaningful, complete paragraphs from fragmented OCR text
Focuses on semantic coherence and readability
"""

import json
import os
import sys
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

# Add project root to path and fix directory paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import settings (with fallback if not available)
try:
    from config.settings import settings
except ImportError:
    # Fallback configuration if settings not available
    class Settings:
        PROCESSED_DOCUMENTS_DIR = str(project_root / "data" / "processed_documents")
        CHUNKED_DOCUMENTS_DIR = str(project_root / "data" / "chunked_documents")
        DEBUG = True
        MIN_PARAGRAPH_LENGTH = 150  # Increased for meaningful chunks
        MAX_PARAGRAPH_LENGTH = 2000  # Reasonable upper limit
        MAX_RETRIES = 3
        
        def validate_config(self):
            return True
        
        def create_directories(self):
            Path(self.CHUNKED_DOCUMENTS_DIR).mkdir(parents=True, exist_ok=True)
    
    settings = Settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SmartDocumentChunker:
    def __init__(self):
        """Initialize the smart document chunker"""
        # Fix paths to be relative to project root
        self.input_dir = Path(settings.PROCESSED_DOCUMENTS_DIR)
        if not self.input_dir.is_absolute():
            self.input_dir = project_root / self.input_dir
            
        self.output_dir = Path(settings.CHUNKED_DOCUMENTS_DIR)
        if not self.output_dir.is_absolute():
            self.output_dir = project_root / self.output_dir
            
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.min_paragraph_length = settings.MIN_PARAGRAPH_LENGTH
        self.max_paragraph_length = settings.MAX_PARAGRAPH_LENGTH
        
        logger.info(f"SmartDocumentChunker initialized")
        logger.info(f"Min paragraph length: {self.min_paragraph_length}")
        logger.info(f"Max paragraph length: {self.max_paragraph_length}")
    
    def is_sentence_fragment(self, text: str) -> bool:
        """Check if text is likely a sentence fragment"""
        text = text.strip()
        
        # Too short
        if len(text) < 30:
            return True
        
        # Doesn't start with capital or ends mid-sentence
        if not text[0].isupper():
            return True
        
        # Ends with incomplete markers
        if text.endswith(('and', 'or', 'but', 'as', 'the', 'of', 'in', 'on', 'at', 'by', 'for', 'with', '--')):
            return True
        
        # Starts with continuation words
        if text.lower().startswith(('and', 'or', 'but', 'however', 'therefore', 'thus', 'hence')):
            return True
        
        return False
    
    def should_merge_with_next(self, current_text: str, next_text: str) -> bool:
        """Determine if current text should be merged with next"""
        current = current_text.strip()
        next_text = next_text.strip()
        
        # Current is incomplete
        if self.is_sentence_fragment(current):
            return True
        
        # Current doesn't end with sentence punctuation
        if not current.endswith(('.', '!', '?', ':', ';')):
            return True
        
        # Next starts with continuation
        if next_text.lower().startswith(('and', 'or', 'but', 'however', 'therefore', 'thus', 'hence', 'moreover', 'furthermore')):
            return True
        
        # Current ends with specific patterns that suggest continuation
        if current.endswith(('--', '-', 'as', 'and', 'or', 'but', 'the', 'of', 'in', 'on', 'at')):
            return True
        
        return False
    
    def clean_and_normalize_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common OCR issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between cases
        text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)  # Add space between digit and letter
        text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)  # Add space between letter and digit
        
        # Fix punctuation spacing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)  # Space after sentence end
        text = re.sub(r'([,;:])([A-Za-z])', r'\1 \2', text)  # Space after punctuation
        
        # Fix currency and symbols
        text = re.sub(r'Rs(\d)', r'Rs \1', text)
        text = re.sub(r'₹(\d)', r'₹\1', text)
        text = re.sub(r'(\d)%', r'\1%', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def reconstruct_paragraphs_from_page(self, page_data: Dict[str, Any]) -> List[str]:
        """Reconstruct complete paragraphs from page data"""
        
        # Get all available text sources
        existing_paragraphs = page_data.get('paragraphs', [])
        raw_text = page_data.get('raw_text', '')
        
        logger.debug(f"Page {page_data.get('page_number', '?')}: {len(existing_paragraphs)} existing paragraphs, raw_text: {len(raw_text)} chars")
        
        # Strategy 1: Try to merge existing paragraph fragments
        if existing_paragraphs:
            merged_paragraphs = self.merge_paragraph_fragments(existing_paragraphs)
            if merged_paragraphs:
                logger.debug(f"Merged {len(existing_paragraphs)} fragments into {len(merged_paragraphs)} paragraphs")
                return merged_paragraphs
        
        # Strategy 2: Extract from raw text if paragraphs are poor
        if raw_text:
            raw_paragraphs = self.extract_paragraphs_from_raw_text(raw_text)
            if raw_paragraphs:
                logger.debug(f"Extracted {len(raw_paragraphs)} paragraphs from raw text")
                return raw_paragraphs
        
        # Strategy 3: Fall back to cleaned existing paragraphs
        cleaned_paragraphs = []
        for para in existing_paragraphs:
            cleaned = self.clean_and_normalize_text(para)
            if len(cleaned) >= 50:  # Lower threshold for fallback
                cleaned_paragraphs.append(cleaned)
        
        return cleaned_paragraphs
    
    def merge_paragraph_fragments(self, paragraphs: List[str]) -> List[str]:
        """Merge fragmented paragraphs into complete ones"""
        if not paragraphs:
            return []
        
        merged = []
        current_paragraph = ""
        
        for i, para in enumerate(paragraphs):
            cleaned_para = self.clean_and_normalize_text(para)
            
            if not cleaned_para:
                continue
            
            # If this is the first paragraph or we should start fresh
            if not current_paragraph:
                current_paragraph = cleaned_para
            else:
                # Check if we should merge with current
                should_merge = False
                
                # Get next paragraph for context
                next_para = ""
                if i + 1 < len(paragraphs):
                    next_para = self.clean_and_normalize_text(paragraphs[i + 1])
                
                should_merge = self.should_merge_with_next(current_paragraph, cleaned_para)
                
                if should_merge:
                    # Merge with current
                    if current_paragraph.endswith(('--', '-')):
                        current_paragraph = current_paragraph.rstrip('- ') + " " + cleaned_para
                    elif current_paragraph.endswith(('.', '!', '?')):
                        current_paragraph += " " + cleaned_para
                    else:
                        current_paragraph += " " + cleaned_para
                else:
                    # Finish current paragraph and start new one
                    if len(current_paragraph) >= self.min_paragraph_length:
                        merged.append(current_paragraph)
                    current_paragraph = cleaned_para
        
        # Add the last paragraph
        if current_paragraph and len(current_paragraph) >= self.min_paragraph_length:
            merged.append(current_paragraph)
        
        return merged
    
    def extract_paragraphs_from_raw_text(self, raw_text: str) -> List[str]:
        """Extract paragraphs from raw text using intelligent splitting"""
        if not raw_text or not raw_text.strip():
            return []
        
        # Clean the text
        cleaned_text = self.clean_and_normalize_text(raw_text)
        
        # Split by natural paragraph breaks
        paragraphs = []
        
        # Method 1: Split by double newlines
        potential_paragraphs = re.split(r'\n\s*\n', cleaned_text)
        
        if len(potential_paragraphs) > 1:
            # We have natural paragraph breaks
            for para in potential_paragraphs:
                cleaned = self.clean_and_normalize_text(para)
                if len(cleaned) >= self.min_paragraph_length:
                    paragraphs.append(cleaned)
        else:
            # Method 2: Use sentence-based splitting
            paragraphs = self.split_text_by_sentences(cleaned_text)
        
        return paragraphs
    
    def split_text_by_sentences(self, text: str) -> List[str]:
        """Split text into paragraphs using sentence boundaries"""
        
        # Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        paragraphs = []
        current_paragraph = []
        current_length = 0
        
        for sentence in sentences:
            # Clean the sentence
            sentence = self.clean_and_normalize_text(sentence)
            
            if not sentence:
                continue
            
            # Decide if this sentence should start a new paragraph
            should_break = self.should_start_new_paragraph(sentence, current_paragraph, current_length)
            
            if should_break and current_paragraph and current_length >= self.min_paragraph_length:
                # Finish current paragraph
                para_text = ' '.join(current_paragraph)
                paragraphs.append(para_text)
                
                # Start new paragraph
                current_paragraph = [sentence]
                current_length = len(sentence)
            else:
                # Add to current paragraph
                current_paragraph.append(sentence)
                current_length += len(sentence) + 1
                
                # Check if paragraph is getting too long
                if current_length > self.max_paragraph_length:
                    para_text = ' '.join(current_paragraph)
                    paragraphs.append(para_text)
                    current_paragraph = []
                    current_length = 0
        
        # Add final paragraph
        if current_paragraph and current_length >= self.min_paragraph_length:
            para_text = ' '.join(current_paragraph)
            paragraphs.append(para_text)
        
        return paragraphs
    
    def should_start_new_paragraph(self, sentence: str, current_paragraph: List[str], current_length: int) -> bool:
        """Determine if sentence should start a new paragraph"""
        
        if not current_paragraph:
            return False
        
        # Paragraph break indicators
        break_indicators = [
            r'^(However|Moreover|Furthermore|Additionally|Meanwhile|Subsequently|Therefore|Thus|Hence|Consequently)',
            r'^(The company|The fund|The investment|The round|The order|The regulator)',
            r'^(According to|As per|In its order|In the order)',
            r'^(Sebi|The Securities and Exchange Board)',
            r'^(In \w+ \d{4})',  # Date references
        ]
        
        for pattern in break_indicators:
            if re.match(pattern, sentence, re.IGNORECASE):
                return True
        
        # Break if getting too long
        if current_length > 600:  # Reasonable paragraph size
            return True
        
        return False
    
    def create_chunks(self, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from document data with proper paragraphs"""
        chunks = []
        doc_id = document_data['doc_id']
        doc_name = document_data['doc_name']
        
        logger.info(f"Processing document: {doc_name}")
        
        chunk_id = 1
        
        for page in document_data['pages']:
            page_number = page['page_number']
            
            # Reconstruct proper paragraphs for this page
            complete_paragraphs = self.reconstruct_paragraphs_from_page(page)
            
            logger.debug(f"Page {page_number}: Created {len(complete_paragraphs)} complete paragraphs")
            
            for para_text in complete_paragraphs:
                if len(para_text) < self.min_paragraph_length:
                    continue
                
                # Ensure paragraph doesn't exceed max length
                if len(para_text) > self.max_paragraph_length:
                    # Split at sentence boundary
                    split_point = para_text.rfind('.', 0, self.max_paragraph_length)
                    if split_point > self.max_paragraph_length // 2:
                        para_text = para_text[:split_point + 1]
                    else:
                        para_text = para_text[:self.max_paragraph_length] + "..."
                
                # Create chunk
                chunk = {
                    "doc_id": doc_id,
                    "page": page_number,
                    "paragraph_number": chunk_id,
                    "text": para_text,
                    "keywords": []
                }
                
                chunks.append(chunk)
                chunk_id += 1
                
                # Log sample chunks
                if len(chunks) <= 3:
                    logger.info(f"Sample chunk {len(chunks)}: length={len(para_text)}")
                    logger.info(f"Text: '{para_text[:200]}...'")
        
        logger.info(f"Created {len(chunks)} complete paragraph chunks for {doc_name}")
        return chunks
    
    def process_document(self, doc_file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single document and create proper chunks"""
        try:
            with open(doc_file_path, 'r', encoding='utf-8') as f:
                document_data = json.load(f)
            
            logger.info(f"Processing: {Path(doc_file_path).name}")
            
            # Create chunks
            chunks = self.create_chunks(document_data)
            
            if not chunks:
                logger.warning(f"No chunks created for {document_data['doc_name']}")
                return None
            
            # Calculate statistics
            total_chars = sum(len(chunk['text']) for chunk in chunks)
            chunk_lengths = [len(chunk['text']) for chunk in chunks]
            
            # Prepare output
            chunked_data = {
                "doc_id": document_data['doc_id'],
                "doc_name": document_data['doc_name'],
                "file_path": document_data.get('file_path', ''),
                "file_type": document_data.get('file_type', 'pdf'),
                "total_chunks": len(chunks),
                "chunks": chunks,
                "metadata": {
                    "original_document": {
                        "total_pages": document_data['total_pages'],
                        "total_paragraphs": document_data.get('total_paragraphs', 0),
                        "file_size": document_data.get('file_size', 0)
                    },
                    "chunking_stats": {
                        "chunks_created": len(chunks),
                        "chunks_per_page": round(len(chunks) / document_data['total_pages'], 2),
                        "avg_chunk_length": round(total_chars / len(chunks), 2),
                        "total_characters": total_chars,
                        "min_chunk_length": min(chunk_lengths),
                        "max_chunk_length": max(chunk_lengths)
                    },
                    "processing_config": {
                        "min_paragraph_length": self.min_paragraph_length,
                        "max_paragraph_length": self.max_paragraph_length,
                        "smart_paragraph_merging": True,
                        "fragment_detection": True,
                        "sentence_boundary_preservation": True,
                        "keywords_extracted": False
                    },
                    "processed_at": datetime.now().isoformat()
                }
            }
            
            # Save
            output_file = self.output_dir / f"{document_data['doc_name']}_chunks_{document_data['doc_id']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunked_data, f, indent=2, ensure_ascii=False)
            
            avg_length = total_chars / len(chunks)
            logger.info(f"Saved {len(chunks)} chunks, avg length: {avg_length:.0f} characters")
            
            return chunked_data
            
        except Exception as e:
            logger.error(f"Error processing {doc_file_path}: {e}")
            return None
    
    def process_all_documents(self) -> List[Dict[str, Any]]:
        """Process all documents"""
        doc_files = list(self.input_dir.glob("*.json"))
        doc_files = [f for f in doc_files if not f.name.startswith(("processing_summary", "chunking_summary", "all_chunks"))]
        
        if not doc_files:
            logger.warning(f"No documents found in {self.input_dir}")
            return []
        
        logger.info(f"Found {len(doc_files)} documents to process")
        
        all_chunked_data = []
        all_chunks = []
        failed_documents = []
        
        for i, doc_file in enumerate(doc_files, 1):
            logger.info(f"Processing ({i}/{len(doc_files)}): {doc_file.name}")
            
            chunked_data = self.process_document(str(doc_file))
            if chunked_data:
                all_chunked_data.append(chunked_data)
                all_chunks.extend(chunked_data['chunks'])
            else:
                failed_documents.append(str(doc_file))
        
        # Save consolidated chunks
        consolidated_chunks = {
            "total_documents": len(all_chunked_data),
            "total_chunks": len(all_chunks),
            "chunks": all_chunks,
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "documents_included": [
                    {
                        "doc_id": doc['doc_id'],
                        "doc_name": doc['doc_name'],
                        "chunks_count": doc['total_chunks']
                    }
                    for doc in all_chunked_data
                ],
                "failed_documents": failed_documents
            }
        }
        
        consolidated_file = self.output_dir / "all_chunks.json"
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_chunks, f, indent=2, ensure_ascii=False)
        
        # Create summary
        if all_chunks:
            chunk_lengths = [len(chunk['text']) for chunk in all_chunks]
            
            summary = {
                "processing_summary": {
                    "total_documents_found": len(doc_files),
                    "successfully_processed": len(all_chunked_data),
                    "failed_processing": len(failed_documents),
                    "success_rate": round((len(all_chunked_data) / len(doc_files) * 100), 2)
                },
                "chunk_statistics": {
                    "total_chunks_created": len(all_chunks),
                    "average_chunks_per_document": round(len(all_chunks) / len(all_chunked_data), 2),
                    "chunk_length_stats": {
                        "min_length": min(chunk_lengths),
                        "max_length": max(chunk_lengths),
                        "avg_length": round(sum(chunk_lengths) / len(chunk_lengths), 2),
                        "total_characters": sum(chunk_lengths)
                    }
                },
                "quality_metrics": {
                    "chunks_under_200_chars": sum(1 for length in chunk_lengths if length < 200),
                    "chunks_over_1000_chars": sum(1 for length in chunk_lengths if length > 1000),
                    "optimal_chunks_200_1000": sum(1 for length in chunk_lengths if 200 <= length <= 1000)
                },
                "processed_at": datetime.now().isoformat()
            }
        else:
            summary = {"error": "No chunks created"}
        
        summary_file = self.output_dir / "chunking_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return all_chunked_data

def main():
    """Main function"""
    print(f"🔍 Smart Document Chunking")
    print(f"=" * 50)
    
    chunker = SmartDocumentChunker()
    
    if not chunker.input_dir.exists():
        print("❌ No processed documents found")
        return
    
    doc_files = list(chunker.input_dir.glob("*.json"))
    doc_files = [f for f in doc_files if not f.name.startswith(("processing_summary", "chunking_summary", "all_chunks"))]
    
    if not doc_files:
        print("❌ No documents to process")
        return
    
    print(f"📁 Found {len(doc_files)} documents")
    
    chunked_docs = chunker.process_all_documents()
    
    if chunked_docs:
        total_chunks = sum(doc['total_chunks'] for doc in chunked_docs)
        all_lengths = []
        for doc in chunked_docs:
            all_lengths.extend([len(chunk['text']) for chunk in doc['chunks']])
        
        print(f"\n✅ Successfully processed {len(chunked_docs)} documents")
        print(f"📊 Total chunks: {total_chunks}")
        print(f"📏 Average chunk length: {sum(all_lengths)/len(all_lengths):.0f} characters")
        print(f"📐 Length range: {min(all_lengths)}-{max(all_lengths)} characters")
        
        quality_chunks = sum(1 for length in all_lengths if 200 <= length <= 1000)
        print(f"✅ Quality chunks (200-1000 chars): {quality_chunks}/{len(all_lengths)} ({quality_chunks/len(all_lengths)*100:.1f}%)")
        
    else:
        print("❌ No documents processed successfully")

if __name__ == "__main__":
    main()