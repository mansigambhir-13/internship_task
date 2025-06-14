"""
Main Pipeline Coordinator
Runs all three scripts in sequence: OCR -> Chunking -> Embeddings
Updated to use configuration from .env file with OpenAI and Gemini support
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(project_root / 'pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('fitz', 'PyMuPDF'),
        ('pytesseract', 'pytesseract'),
        ('PIL', 'Pillow'),
        ('qdrant_client', 'qdrant-client'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn')
    ]
    
    # Optional packages for advanced features
    optional_packages = [
        ('openai', 'openai'),
        ('google.generativeai', 'google-generativeai'),
        ('spacy', 'spacy')
    ]
    
    missing_required = []
    missing_optional = []
    
    # Check required packages
    for package, pip_name in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_required.append(pip_name)
    
    # Check optional packages
    for package, pip_name in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(pip_name)
    
    if missing_required:
        print("❌ Missing required packages:")
        for package in missing_required:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print("⚠️  Missing optional packages (some features may be limited):")
        for package in missing_optional:
            print(f"   - {package}")
        print("\nFor full functionality, install with:")
        print("pip install -r requirements.txt")
    
    return True

def validate_configuration():
    """Validate the configuration from .env file"""
    print("🔧 Validating configuration...")
    
    # Check critical configuration
    if not settings.validate_config():
        print("❌ Configuration validation failed")
        return False
    
    # Check directories
    if not Path(settings.PDF_DIRECTORY).exists():
        print(f"❌ PDF directory not found: {settings.PDF_DIRECTORY}")
        print("Please create the directory and add your PDF documents")
        return False
    
    # Check for PDF files
    pdf_files = list(Path(settings.PDF_DIRECTORY).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ No PDF files found in {settings.PDF_DIRECTORY}")
        print("Please add your PDF documents to process")
        return False
    
    # Check embedding service configuration
    embedding_service = None
    if settings.OPENAI_API_KEY and settings.EMBEDDING_MODEL.startswith("text-embedding"):
        embedding_service = "OpenAI"
    elif settings.GEMINI_API_KEY:
        embedding_service = "Gemini"
    else:
        print("❌ No valid embedding service configured")
        print("Please set either OPENAI_API_KEY or GEMINI_API_KEY in .env file")
        return False
    
    print(f"✅ Configuration valid")
    print(f"   📁 PDF Directory: {settings.PDF_DIRECTORY} ({len(pdf_files)} files)")
    print(f"   🗄️ Qdrant URL: {settings.QDRANT_URL}")
    print(f"   🤖 Embedding Service: {embedding_service}")
    print(f"   📊 Batch Size: {settings.BATCH_SIZE}")
    
    return True

def run_pipeline(skip_confirmation: bool = False):
    """
    Run the complete RAG pipeline
    
    Args:
        skip_confirmation: Skip user confirmations between steps
    """
    start_time = time.time()
    
    print("🚀 Starting RAG Pipeline for Document Research & Theme Identification")
    print("   Wasserstoff AI Internship Project")
    print("=" * 70)
    
    # Validate configuration
    if not validate_configuration():
        return False
    
    # Get document count
    pdf_files = list(Path(settings.PDF_DIRECTORY).glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF documents to process")
    
    if not skip_confirmation:
        proceed = input(f"Proceed with processing {len(pdf_files)} documents? (y/N): ")
        if proceed.lower() != 'y':
            print("❌ Pipeline cancelled")
            return False
    
    # Create necessary directories
    settings.create_directories()
    
    # Step 1: OCR and Parsing
    print("\n" + "="*50)
    print("📄 STEP 1: OCR and Document Parsing")
    print("="*50)
    
    try:
        from ocr_parsing import DocumentProcessor
        
        processor = DocumentProcessor()
        processed_docs = processor.process_document_folder()
        
        if not processed_docs:
            print("❌ OCR processing failed")
            return False
        
        total_pages = sum(doc['total_pages'] for doc in processed_docs)
        total_paragraphs = sum(doc['total_paragraphs'] for doc in processed_docs)
        
        print(f"✅ Successfully processed {len(processed_docs)} documents")
        print(f"   📄 Total pages: {total_pages}")
        print(f"   📝 Total paragraphs: {total_paragraphs}")
        
        time.sleep(1)
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        print(f"❌ OCR processing failed: {e}")
        return False
    
    # Step 2: Chunking
    print("\n" + "="*50)
    print("🔪 STEP 2: Document Chunking")
    print("="*50)
    
    try:
        from chunking import SmartDocumentChunker  # FIXED: Changed from DocumentChunker
        
        chunker = SmartDocumentChunker()  # FIXED: Changed from DocumentChunker
        chunked_docs = chunker.process_all_documents()
        
        if not chunked_docs:
            print("❌ Chunking failed")
            return False
        
        total_chunks = sum(doc['total_chunks'] for doc in chunked_docs)
        # Note: SmartDocumentChunker doesn't create clusters, so removing cluster logic
        
        print(f"✅ Successfully created {total_chunks} chunks from {len(chunked_docs)} documents")
        print(f"   🔑 Keywords extracted: {getattr(settings, 'EXTRACT_KEYWORDS', False)}")
        print(f"   🧠 Advanced NLP: {getattr(settings, 'USE_ADVANCED_NLP', False)}")
        print(f"   🔄 Deduplication: {getattr(settings, 'DEDUPLICATE', False)}")
        
        time.sleep(1)
        
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        print(f"❌ Chunking failed: {e}")
        return False
    
    # Step 3: Embeddings and Vector Storage
    print("\n" + "="*50)
    print("🧮 STEP 3: Creating Embeddings and Vector Storage")
    print("="*50)
    
    try:
        from embeddings import EmbeddingProcessor
        
        embedding_processor = EmbeddingProcessor()
        summary = embedding_processor.process_chunks()
        
        if not summary:
            print("❌ Embeddings processing failed")
            return False
        
        print(f"✅ Successfully created embeddings for {summary['total_chunks_processed']} chunks")
        print(f"   🤖 Service: {summary['embedding_service']}")
        print(f"   📏 Dimension: {summary['embedding_dimension']}")
        print(f"   🗄️ Collection: {summary['qdrant_collection']}")
        
    except Exception as e:
        logger.error(f"Embeddings processing failed: {e}")
        print(f"❌ Embeddings processing failed: {e}")
        return False
    
    # Pipeline completion
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n" + "="*70)
    print("🎉 RAG PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    # Generate comprehensive summary
    pipeline_summary = {
        "pipeline_completed_at": datetime.now().isoformat(),
        "processing_time_seconds": processing_time,
        "documents": {
            "total_processed": len(processed_docs),
            "total_pages": total_pages,
            "total_paragraphs": total_paragraphs
        },
        "chunks": {
            "total_created": total_chunks,
            "avg_per_document": total_chunks / len(processed_docs) if processed_docs else 0
        },
        "embeddings": {
            "service": summary['embedding_service'],
            "model": summary['model_used'],
            "dimension": summary['embedding_dimension'],
            "collection": summary['qdrant_collection']
        },
        "configuration": {
            "pdf_directory": settings.PDF_DIRECTORY,
            "use_ocr": getattr(settings, 'USE_OCR', True),
            "extract_keywords": getattr(settings, 'EXTRACT_KEYWORDS', False),
            "use_advanced_nlp": getattr(settings, 'USE_ADVANCED_NLP', False),
            "deduplicate": getattr(settings, 'DEDUPLICATE', False),
            "batch_size": settings.BATCH_SIZE
        }
    }
    
    # Save pipeline summary
    summary_file = project_root / "pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(pipeline_summary, f, indent=2, ensure_ascii=False)
    
    print(f"📊 Pipeline Summary:")
    print(f"   • Documents processed: {len(processed_docs)}")
    print(f"   • Total pages: {total_pages}")
    print(f"   • Total chunks created: {total_chunks}")
    print(f"   • Embedding service: {summary['embedding_service']}")
    print(f"   • Embeddings dimension: {summary['embedding_dimension']}")
    print(f"   • Processing time: {processing_time:.2f} seconds")
    
    print(f"\n📁 Output Files:")
    print(f"   • Processed documents: {settings.PROCESSED_DOCUMENTS_DIR}/")
    print(f"   • Chunked documents: {settings.CHUNKED_DOCUMENTS_DIR}/")
    print(f"   • Embeddings backup: {getattr(settings, 'EMBEDDING_STORAGE_DIR', 'data/embeddings_storage')}/")
    print(f"   • Processing logs: pipeline.log")
    print(f"   • Pipeline summary: pipeline_summary.json")
    
    print(f"\n🗄️ Vector Database:")
    print(f"   • URL: {settings.QDRANT_URL}")
    print(f"   • Collection: {summary['qdrant_collection']}")
    print(f"   • Total points: {summary['total_chunks_processed']}")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Your knowledge base is ready in Qdrant")
    print(f"   2. You can now build the chatbot interface")
    print(f"   3. Use the search functionality to test queries")
    print(f"   4. Implement theme identification and synthesis")
    print(f"   5. Deploy the complete system")
    
    return True

def test_search_functionality():
    """Test the search functionality with sample queries"""
    try:
        from embeddings import EmbeddingProcessor
        
        processor = EmbeddingProcessor()
        
        print("\n🔍 Testing Search Functionality")
        print("-" * 40)
        
        # Sample queries based on common document types
        sample_queries = [
            "regulatory compliance",
            "penalty and fine",
            "financial disclosure",
            "tribunal decision",
            "corporate governance",
            "market manipulation"
        ]
        
        for query in sample_queries:
            print(f"\nQuery: '{query}'")
            results = processor.search_similar_chunks(query, limit=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. [{result['doc_name']}] Score: {result['score']:.3f}")
                    print(f"     Page {result['page']}, Para {result['para_id']}")
                    print(f"     {result['text'][:100]}...")
                    if result.get('keywords'):
                        print(f"     Keywords: {', '.join(result['keywords'][:3])}")
            else:
                print("  No results found")
        
    except Exception as e:
        print(f"❌ Search test failed: {e}")

def run_individual_step():
    """Run individual pipeline steps"""
    print("\n🔧 Individual Step Execution")
    print("Choose which step to run:")
    print("1. OCR and Document Parsing")
    print("2. Document Chunking")
    print("3. Embeddings and Vector Storage")
    print("4. Test Search Functionality")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    try:
        if choice == "1":
            print("\n📄 Running OCR and Document Parsing...")
            from ocr_parsing import DocumentProcessor
            processor = DocumentProcessor()
            result = processor.process_document_folder()
            if result:
                print(f"✅ Processed {len(result)} documents")
            else:
                print("❌ OCR processing failed")
        
        elif choice == "2":
            print("\n🔪 Running Document Chunking...")
            from chunking import SmartDocumentChunker  # FIXED: Changed from DocumentChunker
            chunker = SmartDocumentChunker()  # FIXED: Changed from DocumentChunker
            result = chunker.process_all_documents()
            if result:
                total_chunks = sum(doc['total_chunks'] for doc in result)
                print(f"✅ Created {total_chunks} chunks from {len(result)} documents")
            else:
                print("❌ Chunking failed")
        
        elif choice == "3":
            print("\n🧮 Running Embeddings and Vector Storage...")
            from embeddings import EmbeddingProcessor
            processor = EmbeddingProcessor()
            result = processor.process_chunks()
            if result:
                print(f"✅ Created embeddings for {result['total_chunks_processed']} chunks")
            else:
                print("❌ Embeddings processing failed")
        
        elif choice == "4":
            print("\n🔍 Testing Search Functionality...")
            test_search_functionality()
        
        else:
            print("❌ Invalid choice")
    
    except Exception as e:
        print(f"❌ Step execution failed: {e}")

def main():
    """Main function"""
    print("🤖 RAG Pipeline for Document Research & Theme Identification")
    print("   Wasserstoff AI Internship Project")
    print("   Configuration loaded from .env file")
    print()
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Please create .env file with your configuration.")
        print("Use .env.example as a template.")
        return
    
    # Main menu
    print("Choose an option:")
    print("1. Run complete pipeline")
    print("2. Run individual steps")
    print("3. Test connections only")
    print("4. Show configuration")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        # Run complete pipeline
        success = run_pipeline()
        
        if success:
            # Optionally test search
            test_search = input("\nWould you like to test the search functionality? (y/N): ")
            if test_search.lower() == 'y':
                test_search_functionality()
    
    elif choice == "2":
        # Run individual steps
        run_individual_step()
    
    elif choice == "3":
        # Test connections only
        try:
            from connection_tester import ConnectionTester
            tester = ConnectionTester()
            tester.run_all_tests()
        except ImportError:
            print("❌ Connection tester not available")
            print("You can manually test by running: python connection_tester.py")
    
    elif choice == "4":
        # Show configuration
        print("\n🔧 Current Configuration:")
        print(f"   PDF Directory: {settings.PDF_DIRECTORY}")
        print(f"   Use OCR: {getattr(settings, 'USE_OCR', True)}")
        print(f"   Extract Keywords: {getattr(settings, 'EXTRACT_KEYWORDS', False)}")
        print(f"   Advanced NLP: {getattr(settings, 'USE_ADVANCED_NLP', False)}")
        print(f"   Deduplication: {getattr(settings, 'DEDUPLICATE', False)}")
        print(f"   Embedding Model: {settings.EMBEDDING_MODEL}")
        print(f"   Batch Size: {settings.BATCH_SIZE}")
        print(f"   Qdrant URL: {settings.QDRANT_URL}")
        print(f"   Collection Name: {settings.COLLECTION_NAME}")
        print(f"   Debug Mode: {settings.DEBUG}")
        print(f"   OpenAI API Key: {'✅ Configured' if settings.OPENAI_API_KEY else '❌ Not set'}")
        print(f"   Gemini API Key: {'✅ Configured' if settings.GEMINI_API_KEY else '❌ Not set'}")
    
    else:
        print("❌ Invalid choice")
    
    print("\n📝 Check pipeline.log for detailed processing logs")

if __name__ == "__main__":
    main()