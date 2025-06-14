"""
Complete RAG Pipeline Main Controller
Orchestrates the entire workflow: OCR -> Chunking -> Embeddings -> Search/Retrieval -> Theme Analysis
Wasserstoff AI Internship Project - Advanced Document Research & Analysis System
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse
from dataclasses import dataclass
import subprocess
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

class PipelineStage(Enum):
    """Pipeline execution stages"""
    SETUP = "setup"
    OCR = "ocr"
    CHUNKING = "chunking"
    EMBEDDINGS = "embeddings"
    SEARCH = "search"
    THEMES = "themes"
    COMPLETE = "complete"

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    
    # Environment and setup
    project_root: Path = PROJECT_ROOT
    env_file: Path = PROJECT_ROOT / ".env"
    logs_dir: Path = PROJECT_ROOT / "logs"
    
    # Data directories - Updated to match actual folder structure
    pdf_directory: str = "docs"  # Changed from "documents/pdfs" to "docs"
    processed_docs_dir: str = "data/processed_documents"
    chunked_docs_dir: str = "data/chunked_documents" 
    embeddings_dir: str = "data/embeddings"
    themes_dir: str = "data/themes"
    
    # Pipeline settings
    skip_existing: bool = True
    auto_proceed: bool = False
    enable_theme_analysis: bool = True
    enable_interactive_search: bool = True
    
    # Processing settings
    batch_size: int = 10
    max_retries: int = 3
    
    def __post_init__(self):
        """Initialize configuration"""
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.logs_dir,
            self.project_root / self.processed_docs_dir,
            self.project_root / self.chunked_docs_dir,
            self.project_root / self.embeddings_dir,
            self.project_root / self.themes_dir
        ]
        
        # Note: We don't create the PDF directory since it already exists as "docs"
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

def setup_logging(config: PipelineConfig) -> logging.Logger:
    """Setup comprehensive logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatters
    file_formatter = logging.Formatter(log_format)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Setup logger
    logger = logging.getLogger('RAG_Pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler
    log_file = config.logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def load_environment(config: PipelineConfig) -> bool:
    """Load environment variables"""
    if not config.env_file.exists():
        print(f"❌ Environment file not found: {config.env_file}")
        print("Please create .env file with your configuration.")
        return False
    
    try:
        # Try to load with python-dotenv
        try:
            from dotenv import load_dotenv
            load_dotenv(config.env_file)
            print("✅ Loaded environment with python-dotenv")
            return True
        except ImportError:
            pass
        
        # Manual loading
        with open(config.env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")
        
        print("✅ Loaded environment variables manually")
        return True
        
    except Exception as e:
        print(f"❌ Failed to load environment: {e}")
        return False

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available"""
    dependencies = {
        'core': ['json', 'os', 'sys', 'time', 'pathlib'],
        'ocr': ['fitz', 'pytesseract', 'PIL'],
        'chunking': ['sklearn', 'numpy'],
        'embeddings': ['qdrant_client'],
        'ai_services': ['openai', 'google.generativeai'],
        'optional': ['spacy', 'transformers', 'dotenv']
    }
    
    results = {}
    
    for category, packages in dependencies.items():
        category_results = {}
        for package in packages:
            try:
                if package == 'google.generativeai':
                    import google.generativeai
                elif package == 'PIL':
                    from PIL import Image
                elif package == 'fitz':
                    import fitz  # PyMuPDF
                else:
                    __import__(package)
                category_results[package] = True
            except ImportError:
                category_results[package] = False
        results[category] = category_results
    
    return results

def validate_environment() -> Dict[str, Any]:
    """Validate environment configuration"""
    validation = {
        'pdf_directory': False,
        'api_keys': False,
        'qdrant_config': False,
        'pdf_files': 0,
        'issues': []
    }
    
    # Check PDF directory - use default "docs" if not specified
    pdf_dir = Path(os.getenv('PDF_DIRECTORY', 'docs'))
    if pdf_dir.exists():
        validation['pdf_directory'] = True
        pdf_files = list(pdf_dir.glob("*.pdf"))
        validation['pdf_files'] = len(pdf_files)
    else:
        validation['issues'].append(f"PDF directory not found: {pdf_dir}")
    
    # Check API keys
    openai_key = os.getenv('OPENAI_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    if openai_key or gemini_key:
        validation['api_keys'] = True
    else:
        validation['issues'].append("No AI service API keys configured")
    
    # Check Qdrant configuration
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_host = os.getenv('QDRANT_HOST', 'localhost')
    
    if qdrant_url or qdrant_host:
        validation['qdrant_config'] = True
    else:
        validation['issues'].append("No Qdrant configuration found")
    
    return validation

class RAGPipelineController:
    """Main RAG Pipeline Controller"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = setup_logging(config)
        self.current_stage = PipelineStage.SETUP
        self.pipeline_state = {}
        self.start_time = time.time()
        
        # Initialize pipeline state
        self.pipeline_state = {
            'start_time': self.start_time,
            'stages_completed': [],
            'stages_failed': [],
            'current_stage': None,
            'documents_processed': 0,
            'chunks_created': 0,
            'embeddings_created': 0,
            'errors': []
        }
    
    def print_banner(self):
        """Print pipeline banner"""
        print("=" * 80)
        print("🤖 ADVANCED RAG PIPELINE FOR DOCUMENT RESEARCH & ANALYSIS")
        print("   Wasserstoff AI Internship Project")
        print("   Complete Workflow: OCR → Chunking → Embeddings → Search → Themes")
        print("=" * 80)
    
    def check_system_readiness(self) -> bool:
        """Check if system is ready for pipeline execution"""
        self.logger.info("Checking system readiness...")
        
        # Load environment
        if not load_environment(self.config):
            return False
        
        # Check dependencies
        deps = check_dependencies()
        missing_critical = []
        
        for category, packages in deps.items():
            if category in ['ocr', 'chunking', 'embeddings']:
                for package, available in packages.items():
                    if not available:
                        missing_critical.append(package)
        
        if missing_critical:
            print(f"❌ Missing critical dependencies: {missing_critical}")
            print("\nTo install missing packages:")
            
            # Provide specific installation commands
            install_commands = {
                'fitz': 'pip install PyMuPDF',
                'pytesseract': 'pip install pytesseract',
                'PIL': 'pip install Pillow',
                'qdrant_client': 'pip install qdrant-client',
                'sklearn': 'pip install scikit-learn',
                'numpy': 'pip install numpy',
                'openai': 'pip install openai',
                'google.generativeai': 'pip install google-generativeai'
            }
            
            print("\nInstall commands:")
            for package in missing_critical:
                if package in install_commands:
                    print(f"   {install_commands[package]}")
                else:
                    print(f"   pip install {package}")
            
            print("\nOr install all at once:")
            print("   pip install PyMuPDF pytesseract Pillow qdrant-client scikit-learn numpy openai google-generativeai")
            return False
        
        # Validate environment
        validation = validate_environment()
        
        if validation['issues']:
            print("⚠️  Configuration issues found:")
            for issue in validation['issues']:
                print(f"   - {issue}")
            
            if not validation['api_keys'] or not validation['qdrant_config']:
                print("❌ Critical configuration missing")
                return False
        
        # Print system status
        print("✅ System Status:")
        print(f"   📁 PDF Directory: {validation['pdf_directory']} ({validation['pdf_files']} files)")
        print(f"   🔑 API Keys: {validation['api_keys']}")
        print(f"   🗄️ Qdrant Config: {validation['qdrant_config']}")
        
        return True
    
    def execute_stage(self, stage: PipelineStage, **kwargs) -> bool:
        """Execute a specific pipeline stage"""
        self.current_stage = stage
        self.pipeline_state['current_stage'] = stage.value
        
        stage_start = time.time()
        
        try:
            self.logger.info(f"Starting stage: {stage.value}")
            
            if stage == PipelineStage.OCR:
                success = self._execute_ocr_stage(**kwargs)
            elif stage == PipelineStage.CHUNKING:
                success = self._execute_chunking_stage(**kwargs)
            elif stage == PipelineStage.EMBEDDINGS:
                success = self._execute_embeddings_stage(**kwargs)
            elif stage == PipelineStage.SEARCH:
                success = self._execute_search_stage(**kwargs)
            elif stage == PipelineStage.THEMES:
                success = self._execute_themes_stage(**kwargs)
            else:
                self.logger.error(f"Unknown stage: {stage}")
                return False
            
            stage_time = time.time() - stage_start
            
            if success:
                self.pipeline_state['stages_completed'].append(stage.value)
                self.logger.info(f"Stage {stage.value} completed in {stage_time:.2f}s")
                return True
            else:
                self.pipeline_state['stages_failed'].append(stage.value)
                self.logger.error(f"Stage {stage.value} failed after {stage_time:.2f}s")
                return False
                
        except Exception as e:
            self.logger.error(f"Stage {stage.value} crashed: {e}")
            self.pipeline_state['stages_failed'].append(stage.value)
            self.pipeline_state['errors'].append(str(e))
            return False
    
    def _execute_ocr_stage(self, **kwargs) -> bool:
        """Execute OCR and document parsing stage"""
        print("\n" + "="*60)
        print("📄 STAGE 1: OCR AND DOCUMENT PARSING")
        print("="*60)
        
        try:
            # Check PDF directory first
            pdf_directory = os.getenv('PDF_DIRECTORY', self.config.pdf_directory)
            pdf_path = Path(pdf_directory)
            
            if not pdf_path.exists():
                self.logger.error(f"PDF directory not found: {pdf_path}")
                print(f"❌ PDF directory not found: {pdf_path}")
                return False
            
            # Check for PDF files
            pdf_files = list(pdf_path.glob("*.pdf"))
            if not pdf_files:
                self.logger.error(f"No PDF files found in {pdf_path}")
                print(f"❌ No PDF files found in {pdf_path}")
                print(f"   Please add PDF files to: {pdf_path.absolute()}")
                return False
            
            print(f"📁 Found {len(pdf_files)} PDF files in {pdf_path}")
            for pdf_file in pdf_files:
                print(f"   • {pdf_file.name}")
            
            # Check if OCR script exists
            ocr_script_paths = [
                self.config.project_root / "ocr_parsing.py",
                self.config.project_root / "scripts" / "ocr_parsing.py", 
                self.config.project_root / "src" / "ocr_parsing.py",
            ]
            
            ocr_script = None
            for script_path in ocr_script_paths:
                if script_path.exists():
                    ocr_script = script_path
                    break
            
            if not ocr_script:
                self.logger.error("OCR script not found in any expected location")
                print("❌ OCR script (ocr_parsing.py) not found")
                print("   Expected locations:")
                for path in ocr_script_paths:
                    print(f"   • {path}")
                print("\n💡 Please ensure ocr_parsing.py is in the project root directory")
                return False
            
            print(f"✅ Found OCR script: {ocr_script}")
            
            # Try to import OCR module
            try:
                # Add the script directory to Python path temporarily
                script_dir = str(ocr_script.parent)
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)
                
                # Try different import methods
                try:
                    from ocr_parsing import DocumentProcessor
                    print("✅ Successfully imported DocumentProcessor")
                except ImportError as e:
                    print(f"❌ Failed to import DocumentProcessor: {e}")
                    
                    # Try alternative import
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("ocr_parsing", ocr_script)
                    ocr_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(ocr_module)
                    DocumentProcessor = ocr_module.DocumentProcessor
                    print("✅ Successfully imported DocumentProcessor via importlib")
                
                processor = DocumentProcessor()
                
                # Check for existing processed documents
                if self.config.skip_existing:
                    processed_dir = Path(self.config.processed_docs_dir)
                    existing_files = list(processed_dir.glob("*.json")) if processed_dir.exists() else []
                    
                    if existing_files:
                        print(f"✅ Found {len(existing_files)} existing processed documents, skipping OCR")
                        # Update pipeline state with existing data
                        try:
                            # Count documents from existing files
                            doc_count = 0
                            for file in existing_files:
                                try:
                                    with open(file, 'r', encoding='utf-8') as f:
                                        data = json.load(f)
                                        if isinstance(data, list):
                                            doc_count += len(data)
                                        else:
                                            doc_count += 1
                                except:
                                    doc_count += 1
                            
                            self.pipeline_state['documents_processed'] = doc_count
                            print(f"   📄 Existing documents: {doc_count}")
                            return True
                        except Exception as e:
                            print(f"⚠️  Could not count existing documents: {e}")
                            return True  # Still consider it successful
                
                print("🔄 Starting OCR processing...")
                
                # Execute OCR processing
                results = processor.process_document_folder()
                
                if results:
                    total_pages = sum(doc.get('total_pages', 0) for doc in results)
                    total_paragraphs = sum(doc.get('total_paragraphs', 0) for doc in results)
                    
                    self.pipeline_state['documents_processed'] = len(results)
                    
                    print(f"✅ OCR Processing Complete:")
                    print(f"   📄 Documents processed: {len(results)}")
                    print(f"   📝 Total pages: {total_pages}")
                    print(f"   📋 Total paragraphs: {total_paragraphs}")
                    
                    # Verify output files were created
                    processed_dir = Path(self.config.processed_docs_dir)
                    if processed_dir.exists():
                        output_files = list(processed_dir.glob("*.json"))
                        print(f"   💾 Output files created: {len(output_files)}")
                    
                    return True
                else:
                    print("❌ OCR processing failed - no documents processed")
                    print("   Check if PDF files are readable and not corrupted")
                    return False
                    
            except ImportError as e:
                self.logger.error(f"Failed to import OCR module: {e}")
                print(f"❌ Failed to import OCR module: {e}")
                print("\n💡 Troubleshooting tips:")
                print("   1. Make sure ocr_parsing.py exists in the project root")
                print("   2. Check that the DocumentProcessor class is defined in ocr_parsing.py")
                print("   3. Ensure all OCR dependencies are installed:")
                print("      pip install PyMuPDF pytesseract Pillow")
                return False
            except Exception as e:
                self.logger.error(f"OCR processing error: {e}")
                print(f"❌ OCR processing error: {e}")
                print("\n💡 Common issues:")
                print("   1. Tesseract not installed on system")
                print("   2. PDF files are password protected or corrupted")
                print("   3. Insufficient disk space for output")
                print("   4. Permission issues with output directory")
                return False
                
        except Exception as e:
            self.logger.error(f"OCR stage failed: {e}")
            print(f"❌ OCR stage failed: {e}")
            return False
    
    def _execute_chunking_stage(self, **kwargs) -> bool:
        """Execute document chunking stage"""
        print("\n" + "="*60)
        print("🔪 STAGE 2: DOCUMENT CHUNKING")
        print("="*60)
        
        try:
            # Import chunking module
            try:
                from chunking import DocumentChunker
                
                chunker = DocumentChunker()
                
                # Check for existing chunks
                if self.config.skip_existing:
                    chunked_dir = Path(self.config.chunked_docs_dir)
                    if chunked_dir.exists() and list(chunked_dir.glob("*.json")):
                        print("✅ Found existing chunks, skipping chunking")
                        return True
                
                results = chunker.process_all_documents()
                
                if results:
                    total_chunks = sum(doc.get('total_chunks', 0) for doc in results)
                    self.pipeline_state['chunks_created'] = total_chunks
                    
                    print(f"✅ Chunking Complete:")
                    print(f"   🧩 Total chunks created: {total_chunks}")
                    print(f"   📚 Documents chunked: {len(results)}")
                    
                    return True
                else:
                    print("❌ Chunking failed - no chunks created")
                    return False
                    
            except ImportError as e:
                self.logger.error(f"Failed to import chunking module: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Chunking stage failed: {e}")
            return False
    
    def _execute_embeddings_stage(self, **kwargs) -> bool:
        """Execute embeddings creation stage"""
        print("\n" + "="*60)
        print("🧮 STAGE 3: EMBEDDINGS AND VECTOR STORAGE")
        print("="*60)
        
        try:
            # Import embeddings module
            try:
                from embeddings import EmbeddingProcessor
                
                processor = EmbeddingProcessor()
                
                # Check for existing embeddings
                if self.config.skip_existing:
                    # Check Qdrant collection
                    try:
                        collection_name = os.getenv('COLLECTION_NAME', 'document_chunks')
                        collection_info = processor.qdrant_client.get_collection(collection_name)
                        if collection_info.points_count > 0:
                            print(f"✅ Found existing embeddings in Qdrant ({collection_info.points_count:,} points)")
                            return True
                    except:
                        pass
                
                results = processor.process_chunks()
                
                if results:
                    self.pipeline_state['embeddings_created'] = results.get('total_chunks_processed', 0)
                    
                    print(f"✅ Embeddings Complete:")
                    print(f"   🤖 Service: {results.get('embedding_service', 'Unknown')}")
                    print(f"   📏 Dimension: {results.get('embedding_dimension', 0)}")
                    print(f"   🗄️ Collection: {results.get('qdrant_collection', 'Unknown')}")
                    print(f"   📊 Chunks processed: {results.get('total_chunks_processed', 0):,}")
                    
                    return True
                else:
                    print("❌ Embeddings creation failed")
                    return False
                    
            except ImportError as e:
                self.logger.error(f"Failed to import embeddings module: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Embeddings stage failed: {e}")
            return False
    
    def _execute_search_stage(self, **kwargs) -> bool:
        """Execute search functionality setup and testing"""
        print("\n" + "="*60)
        print("🔍 STAGE 4: SEARCH AND RETRIEVAL SETUP")
        print("="*60)
        
        try:
            # Import search module
            try:
                from search import SemanticSearchEngine, SearchConfig
                
                search_config = SearchConfig()
                search_engine = SemanticSearchEngine(search_config)
                
                # Test search functionality
                test_queries = [
                    "main topic",
                    "key findings", 
                    "recommendations"
                ]
                
                print("🧪 Testing search functionality...")
                
                successful_searches = 0
                for query in test_queries:
                    try:
                        result = search_engine.search_and_answer(query)
                        if result and result.get('confidence') != 'error':
                            successful_searches += 1
                            print(f"   ✅ Query '{query}': {result.get('confidence', 'unknown')} confidence")
                        else:
                            print(f"   ❌ Query '{query}': Failed")
                    except Exception as e:
                        print(f"   ❌ Query '{query}': Error - {e}")
                
                if successful_searches > 0:
                    print(f"✅ Search Setup Complete:")
                    print(f"   🎯 Successful tests: {successful_searches}/{len(test_queries)}")
                    print(f"   🗄️ Vector database ready")
                    print(f"   🤖 AI services connected")
                    return True
                else:
                    print("❌ Search functionality failed all tests")
                    return False
                    
            except ImportError as e:
                self.logger.error(f"Failed to import search module: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Search stage failed: {e}")
            return False
    
    def _execute_themes_stage(self, **kwargs) -> bool:
        """Execute theme identification stage"""
        print("\n" + "="*60)
        print("🎯 STAGE 5: THEME IDENTIFICATION & ANALYSIS")
        print("="*60)
        
        try:
            # Import theme analysis module
            try:
                from theme_identifier import ConciseThemeIdentifier
                
                identifier = ConciseThemeIdentifier()
                
                if not identifier.check_setup():
                    print("❌ Theme identifier setup failed")
                    return False
                
                # Test theme identification with sample queries
                test_queries = [
                    "regulatory compliance requirements",
                    "financial penalties and violations",
                    "corporate governance issues"
                ]
                
                print("🧪 Testing theme identification...")
                
                successful_analyses = 0
                for query in test_queries:
                    try:
                        themes = identifier.identify_themes_concise(query, max_themes=3)
                        if themes:
                            successful_analyses += 1
                            print(f"   ✅ Query '{query}': {len(themes)} themes identified")
                        else:
                            print(f"   ❌ Query '{query}': No themes found")
                    except Exception as e:
                        print(f"   ❌ Query '{query}': Error - {e}")
                
                if successful_analyses > 0:
                    print(f"✅ Theme Analysis Setup Complete:")
                    print(f"   🎯 Successful analyses: {successful_analyses}/{len(test_queries)}")
                    print(f"   🧩 Theme identification ready")
                    print(f"   📊 Comprehensive analysis available")
                    return True
                else:
                    print("❌ Theme identification failed all tests")
                    return False
                    
            except ImportError as e:
                self.logger.error(f"Failed to import theme analysis module: {e}")
                # Theme analysis is optional, so don't fail pipeline
                print("⚠️  Theme analysis not available - continuing without it")
                return True
                
        except Exception as e:
            self.logger.error(f"Theme analysis stage failed: {e}")
            # Theme analysis is optional
            return True
    
    def run_complete_pipeline(self, stages: List[PipelineStage] = None) -> bool:
        """Run the complete pipeline or specified stages"""
        
        if stages is None:
            stages = [
                PipelineStage.OCR,
                PipelineStage.CHUNKING, 
                PipelineStage.EMBEDDINGS,
                PipelineStage.SEARCH,
                PipelineStage.THEMES
            ]
        
        self.logger.info("Starting complete RAG pipeline execution")
        
        # Execute each stage
        for stage in stages:
            print(f"\n⏭️  Proceeding to {stage.value.upper()} stage...")
            
            if not self.config.auto_proceed:
                user_input = input(f"Continue with {stage.value}? (y/N/s=skip): ").strip().lower()
                if user_input == 's':
                    print(f"⏭️  Skipping {stage.value} stage")
                    continue
                elif user_input != 'y':
                    print("❌ Pipeline stopped by user")
                    return False
            
            success = self.execute_stage(stage)
            
            if not success:
                print(f"❌ Pipeline failed at {stage.value} stage")
                return False
            
            time.sleep(1)  # Brief pause between stages
        
        self.current_stage = PipelineStage.COMPLETE
        return True
    
    def generate_pipeline_report(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline report"""
        total_time = time.time() - self.start_time
        
        report = {
            "pipeline_execution": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_duration_seconds": round(total_time, 2),
                "status": "completed" if self.current_stage == PipelineStage.COMPLETE else "partial"
            },
            "stages": {
                "completed": self.pipeline_state['stages_completed'],
                "failed": self.pipeline_state['stages_failed'],
                "current_stage": self.pipeline_state.get('current_stage')
            },
            "processing_summary": {
                "documents_processed": self.pipeline_state.get('documents_processed', 0),
                "chunks_created": self.pipeline_state.get('chunks_created', 0),
                "embeddings_created": self.pipeline_state.get('embeddings_created', 0)
            },
            "configuration": {
                "pdf_directory": self.config.pdf_directory,
                "skip_existing": self.config.skip_existing,
                "auto_proceed": self.config.auto_proceed,
                "enable_theme_analysis": self.config.enable_theme_analysis
            },
            "errors": self.pipeline_state.get('errors', [])
        }
        
        return report
    
    def save_pipeline_report(self, report: Dict[str, Any]):
        """Save pipeline report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.config.project_root / f"pipeline_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Pipeline report saved: {report_file}")
    
    def print_final_summary(self, report: Dict[str, Any]):
        """Print final pipeline summary"""
        print("\n" + "="*80)
        print("🎉 RAG PIPELINE EXECUTION SUMMARY")
        print("="*80)
        
        # Status
        status = report['pipeline_execution']['status']
        duration = report['pipeline_execution']['total_duration_seconds']
        
        print(f"📊 Status: {status.upper()}")
        print(f"⏱️  Total Duration: {duration:.2f} seconds ({duration/60:.1f} minutes)")
        
        # Stages
        completed = report['stages']['completed']
        failed = report['stages']['failed']
        
        print(f"\n✅ Completed Stages ({len(completed)}):")
        for stage in completed:
            print(f"   • {stage}")
        
        if failed:
            print(f"\n❌ Failed Stages ({len(failed)}):")
            for stage in failed:
                print(f"   • {stage}")
        
        # Processing summary
        summary = report['processing_summary']
        print(f"\n📈 Processing Summary:")
        print(f"   📄 Documents processed: {summary['documents_processed']}")
        print(f"   🧩 Chunks created: {summary['chunks_created']:,}")
        print(f"   🧮 Embeddings created: {summary['embeddings_created']:,}")
        
        # Next steps
        if status == "completed":
            print(f"\n🚀 Your RAG system is ready! Next steps:")
            print(f"   1. 🔍 Test search functionality")
            print(f"   2. 🎯 Run theme analysis")
            print(f"   3. 💬 Build chatbot interface")
            print(f"   4. 📊 Monitor and optimize performance")
        else:
            print(f"\n⚠️  Pipeline incomplete. Check logs for issues.")
        
        print("="*80)

def interactive_mode(controller: RAGPipelineController):
    """Interactive pipeline execution mode"""
    print("\n🎮 INTERACTIVE MODE")
    print("Choose execution option:")
    print("1. Run complete pipeline")
    print("2. Run individual stages")
    print("3. Resume from specific stage")
    print("4. Test existing system")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        print("\n🚀 Running complete pipeline...")
        success = controller.run_complete_pipeline()
        
    elif choice == "2":
        print("\n📋 Available stages:")
        stages = [PipelineStage.OCR, PipelineStage.CHUNKING, PipelineStage.EMBEDDINGS, PipelineStage.SEARCH, PipelineStage.THEMES]
        
        for i, stage in enumerate(stages, 1):
            print(f"{i}. {stage.value}")
        
        stage_choices = input("Enter stage numbers (comma-separated): ").strip()
        
        try:
            selected_indices = [int(x.strip()) - 1 for x in stage_choices.split(',')]
            selected_stages = [stages[i] for i in selected_indices]
            success = controller.run_complete_pipeline(selected_stages)
        except (ValueError, IndexError):
            print("❌ Invalid stage selection")
            return
    
    elif choice == "3":
        print("\n📋 Resume from stage:")
        stages = [PipelineStage.OCR, PipelineStage.CHUNKING, PipelineStage.EMBEDDINGS, PipelineStage.SEARCH, PipelineStage.THEMES]
        
        for i, stage in enumerate(stages, 1):
            print(f"{i}. {stage.value}")
        
        stage_choice = input("Enter stage number: ").strip()
        
        try:
            start_index = int(stage_choice) - 1
            resume_stages = stages[start_index:]
            success = controller.run_complete_pipeline(resume_stages)
        except (ValueError, IndexError):
            print("❌ Invalid stage selection")
            return
    
    elif choice == "4":
        print("\n🧪 Testing existing system...")
        # Test search functionality
        success = controller.execute_stage(PipelineStage.SEARCH)
        if success and controller.config.enable_theme_analysis:
            success = controller.execute_stage(PipelineStage.THEMES)
    
    elif choice == "5":
        print("👋 Goodbye!")
        return
    
    else:
        print("❌ Invalid choice")
        return
    
    # Generate and display report
    report = controller.generate_pipeline_report()
    controller.save_pipeline_report(report)
    controller.print_final_summary(report)

def diagnose_ocr_issues():
    """Diagnose OCR-related issues"""
    print("\n🔍 OCR DIAGNOSTIC CHECK")
    print("=" * 40)
    
    issues = []
    
    # 1. Check PDF directory
    pdf_directory = os.getenv('PDF_DIRECTORY', 'docs')
    pdf_path = Path(pdf_directory)
    
    print(f"📁 PDF Directory: {pdf_path.absolute()}")
    
    if not pdf_path.exists():
        issues.append(f"PDF directory does not exist: {pdf_path}")
        print(f"❌ Directory not found")
    else:
        pdf_files = list(pdf_path.glob("*.pdf"))
        print(f"✅ Directory exists")
        print(f"📄 PDF files found: {len(pdf_files)}")
        
        if pdf_files:
            for i, pdf_file in enumerate(pdf_files[:5], 1):
                file_size = pdf_file.stat().st_size / (1024*1024)  # MB
                print(f"   {i}. {pdf_file.name} ({file_size:.1f} MB)")
            if len(pdf_files) > 5:
                print(f"   ... and {len(pdf_files) - 5} more files")
        else:
            issues.append("No PDF files found in directory")
            print(f"❌ No PDF files found")
    
    # 2. Check OCR script
    print(f"\n📜 OCR Script Check:")
    
    ocr_script_paths = [
        Path("ocr_parsing.py"),
        Path("scripts/ocr_parsing.py"),
        Path("src/ocr_parsing.py"),
    ]
    
    ocr_script_found = False
    for script_path in ocr_script_paths:
        print(f"   Checking: {script_path.absolute()}")
        if script_path.exists():
            print(f"   ✅ Found: {script_path}")
            ocr_script_found = True
            
            # Check if it contains DocumentProcessor class
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'class DocumentProcessor' in content:
                        print(f"   ✅ Contains DocumentProcessor class")
                    else:
                        issues.append(f"OCR script missing DocumentProcessor class")
                        print(f"   ❌ Missing DocumentProcessor class")
            except Exception as e:
                issues.append(f"Cannot read OCR script: {e}")
                print(f"   ❌ Cannot read file: {e}")
            break
    
    if not ocr_script_found:
        issues.append("OCR script (ocr_parsing.py) not found")
        print(f"   ❌ OCR script not found in any expected location")
    
    # 3. Check OCR dependencies
    print(f"\n🔧 OCR Dependencies Check:")
    
    ocr_deps = {
        'PyMuPDF (fitz)': 'fitz',
        'Tesseract (pytesseract)': 'pytesseract', 
        'Pillow (PIL)': 'PIL'
    }
    
    for name, module in ocr_deps.items():
        try:
            if module == 'PIL':
                from PIL import Image
            else:
                __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            issues.append(f"Missing dependency: {name}")
            print(f"   ❌ {name} - Install with: pip install {name.split()[0]}")
    
    # 4. Check Tesseract system installation
    print(f"\n🖼️  Tesseract System Check:")
    try:
        import pytesseract
        import subprocess
        
        # Try to get tesseract version
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"   ✅ Tesseract installed: {version_line}")
        else:
            issues.append("Tesseract command not working")
            print(f"   ❌ Tesseract command failed")
    except subprocess.TimeoutExpired:
        issues.append("Tesseract command timeout")
        print(f"   ❌ Tesseract command timeout")
    except FileNotFoundError:
        issues.append("Tesseract not installed on system")
        print(f"   ❌ Tesseract not installed on system")
        print(f"      Install instructions:")
        print(f"      - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print(f"      - macOS: brew install tesseract")
        print(f"      - Linux: sudo apt-get install tesseract-ocr")
    except Exception as e:
        issues.append(f"Tesseract check failed: {e}")
        print(f"   ❌ Tesseract check failed: {e}")
    
    # 5. Check output directories
    print(f"\n📂 Output Directories Check:")
    
    output_dirs = [
        ('Processed Documents', os.getenv('PROCESSED_DOCUMENTS_DIR', 'data/processed_documents')),
        ('Chunked Documents', os.getenv('CHUNKED_DOCUMENTS_DIR', 'data/chunked_documents')),
        ('Embeddings', os.getenv('EMBEDDING_STORAGE_DIR', 'data/embeddings')),
    ]
    
    for name, dir_path in output_dirs:
        dir_path = Path(dir_path)
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {name}: {dir_path}")
        except Exception as e:
            issues.append(f"Cannot create {name} directory: {e}")
            print(f"   ❌ {name}: {e}")
    
    # Summary
    print(f"\n📋 DIAGNOSTIC SUMMARY:")
    if not issues:
        print(f"   ✅ All checks passed! OCR should work.")
        return True
    else:
        print(f"   ❌ Found {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            print(f"      {i}. {issue}")
        
        print(f"\n💡 RECOMMENDED ACTIONS:")
        if any("PDF" in issue for issue in issues):
            print(f"   1. Add PDF files to: {pdf_path.absolute()}")
        if any("OCR script" in issue for issue in issues):
            print(f"   2. Ensure ocr_parsing.py is in the project root")
        if any("dependency" in issue.lower() for issue in issues):
            print(f"   3. Install missing dependencies:")
            print(f"      pip install PyMuPDF pytesseract Pillow")
        if any("Tesseract" in issue for issue in issues):
            print(f"   4. Install Tesseract OCR on your system")
        
        return False

def launch_search_interface():
    """Launch interactive search interface"""
    print("\n🔍 Launching search interface...")
    
    try:
        from search import interactive_search_session
        interactive_search_session()
    except ImportError:
        print("❌ Search module not available")
    except Exception as e:
        print(f"❌ Search interface failed: {e}")

def launch_theme_analysis():
    """Launch theme analysis interface"""
    print("\n🎯 Launching theme analysis...")
    
    try:
        # Import and run theme analysis
        import subprocess
        import sys
        
        theme_script = PROJECT_ROOT / "theme_identifier.py"
        if theme_script.exists():
            subprocess.run([sys.executable, str(theme_script)], cwd=PROJECT_ROOT)
        else:
            print("❌ Theme analysis script not found")
    except Exception as e:
        print(f"❌ Theme analysis failed: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="RAG Pipeline Controller")
    parser.add_argument("--auto", action="store_true", help="Run pipeline automatically without prompts")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip stages with existing data")
    parser.add_argument("--force", action="store_true", help="Force reprocessing of all stages")
    parser.add_argument("--stages", nargs="+", choices=["ocr", "chunking", "embeddings", "search", "themes"], 
                       help="Run specific stages only")
    parser.add_argument("--search-only", action="store_true", help="Launch search interface only")
    parser.add_argument("--themes-only", action="store_true", help="Launch theme analysis only")
    parser.add_argument("--diagnose-ocr", action="store_true", help="Diagnose OCR setup issues")
    parser.add_argument("--config-check", action="store_true", help="Check configuration and exit")
    parser.add_argument("--batch-mode", action="store_true", help="Run in batch mode (no interactive prompts)")
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = PipelineConfig()
    config.auto_proceed = args.auto or args.batch_mode
    config.skip_existing = args.skip_existing and not args.force
    
    # Initialize controller
    controller = RAGPipelineController(config)
    controller.print_banner()
    
    # OCR diagnostic mode
    if args.diagnose_ocr:
        print("\n🔍 DIAGNOSING OCR ISSUES")
        print("-" * 40)
        
        if diagnose_ocr_issues():
            print("\n✅ OCR setup looks good! Try running the pipeline.")
        else:
            print("\n❌ OCR setup has issues. Fix the problems above and try again.")
        return
    
    # Configuration check mode
    if args.config_check:
        print("\n🔧 CONFIGURATION CHECK")
        print("-" * 40)
        
        if controller.check_system_readiness():
            print("✅ System configuration is valid")
            
            # Additional checks
            deps = check_dependencies()
            validation = validate_environment()
            
            print(f"\n📊 System Details:")
            print(f"   📁 PDF files found: {validation['pdf_files']}")
            print(f"   🔑 API keys configured: {validation['api_keys']}")
            print(f"   🗄️ Qdrant ready: {validation['qdrant_config']}")
            
            print(f"\n🧩 Dependencies:")
            for category, packages in deps.items():
                available = sum(packages.values())
                total = len(packages)
                print(f"   {category}: {available}/{total} available")
        else:
            print("❌ System configuration has issues")
        return
    
    # Search-only mode
    if args.search_only:
        if controller.check_system_readiness():
            launch_search_interface()
        return
    
    # Theme analysis only mode
    if args.themes_only:
        if controller.check_system_readiness():
            launch_theme_analysis()
        return
    
    # Check system readiness
    if not controller.check_system_readiness():
        print("\n❌ System not ready. Use --config-check to diagnose issues.")
        return
    
    # Determine stages to run
    if args.stages:
        stage_map = {
            "ocr": PipelineStage.OCR,
            "chunking": PipelineStage.CHUNKING, 
            "embeddings": PipelineStage.EMBEDDINGS,
            "search": PipelineStage.SEARCH,
            "themes": PipelineStage.THEMES
        }
        stages_to_run = [stage_map[stage] for stage in args.stages]
    else:
        stages_to_run = None  # Run all stages
    
    # Execute pipeline
    try:
        if args.batch_mode or args.auto:
            print("\n🤖 Running in automatic mode...")
            success = controller.run_complete_pipeline(stages_to_run)
            
            # Generate final report
            report = controller.generate_pipeline_report()
            controller.save_pipeline_report(report)
            controller.print_final_summary(report)
            
            if success:
                print("\n🎯 Pipeline completed successfully!")
                
                # Offer to launch interfaces
                if not args.batch_mode:
                    launch_search = input("\nLaunch search interface? (y/N): ").strip().lower()
                    if launch_search == 'y':
                        launch_search_interface()
                    
                    if config.enable_theme_analysis:
                        launch_themes = input("Launch theme analysis? (y/N): ").strip().lower()
                        if launch_themes == 'y':
                            launch_theme_analysis()
            else:
                print("\n❌ Pipeline failed. Check logs for details.")
        
        else:
            # Interactive mode
            interactive_mode(controller)
    
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        
        # Save partial report
        report = controller.generate_pipeline_report()
        controller.save_pipeline_report(report)
        print("📊 Partial report saved")
    
    except Exception as e:
        print(f"\n❌ Pipeline crashed: {e}")
        controller.logger.error(f"Pipeline crashed: {e}")
        
        # Save error report
        report = controller.generate_pipeline_report()
        report['crash_error'] = str(e)
        controller.save_pipeline_report(report)
        print("📊 Error report saved")

def quick_start_wizard():
    """Quick start wizard for first-time users"""
    print("\n🧙‍♂️ RAG PIPELINE QUICK START WIZARD")
    print("=" * 50)
    
    # Check if .env exists
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("❌ No .env file found!")
        print("\n📝 Creating .env template...")
        
        # Create .env template
        env_template = """# RAG Pipeline Configuration
# Copy this to .env and fill in your values

# PDF Documents Directory (matches your folder structure)
PDF_DIRECTORY=docs

# AI Service API Keys (choose one or both)
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Qdrant Vector Database Configuration
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=document_chunks

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
BATCH_SIZE=10

# Processing Options
USE_OCR=true
EXTRACT_KEYWORDS=true
USE_ADVANCED_NLP=true
DEDUPLICATE=true

# Search Configuration
TOP_K_CHUNKS=20
FINAL_CHUNKS=3
SIMILARITY_THRESHOLD=0.3

# Answer Generation
ANSWER_MODEL=gpt-3.5-turbo
MAX_CONTEXT_LENGTH=4000

# System Settings
DEBUG=false
"""
        
        with open(env_file, 'w') as f:
            f.write(env_template)
        
        print(f"✅ Created .env template at: {env_file}")
        print("\n📋 Please edit .env file and add your API keys, then run the pipeline again.")
        return False
    
    # Check PDF directory
    pdf_dir = Path("documents/pdfs")
    if not pdf_dir.exists():
        print(f"📁 Creating PDF directory: {pdf_dir}")
        pdf_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"📄 No PDF files found in {pdf_dir}")
        print("Please add your PDF documents to process.")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files")
    
    # Check dependencies
    print("\n🔍 Checking dependencies...")
    deps = check_dependencies()
    
    missing = []
    for category, packages in deps.items():
        for package, available in packages.items():
            if not available and category in ['ocr', 'chunking', 'embeddings']:
                missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {missing}")
        print("Please install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    
    # Final check
    print("\n🎯 Ready to start RAG pipeline!")
    print("Run with: python main.py --auto")
    
    return True

if __name__ == "__main__":
    try:
        # Check if this is first run
        env_file = PROJECT_ROOT / ".env"
        if not env_file.exists():
            if quick_start_wizard():
                print("\n🚀 Run 'python main.py' to start the pipeline!")
        else:
            main()
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logging.getLogger().error(f"Fatal error: {e}")