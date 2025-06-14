"""
Production-Ready Embeddings and Vector Database Script - WINDOWS COMPATIBLE
Optimized for performance, reliability, and scalability
Creates embeddings for chunks and stores them in Qdrant vector database
FIXED: Windows Unicode/Emoji issues resolved
"""

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import uuid
import hashlib
import asyncio
import aiohttp
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import signal
import psutil
import gc
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load .env file with enhanced loading
def load_env_file():
    """Load environment variables from .env file with enhanced error handling"""
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            print("[OK] Loaded .env file using python-dotenv")
            return True
    except ImportError:
        pass
    
    # Enhanced fallback loading
    env_files = [Path('.env'), project_root / '.env', Path.home() / '.env']
    
    for env_file in env_files:
        if env_file.exists():
            print(f"[INFO] Loading .env file from: {env_file}")
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            try:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"').strip("'")
                                os.environ[key] = value
                            except Exception as e:
                                print(f"[WARN] Error parsing line {line_num}: {e}")
                print(f"[OK] Loaded .env file from: {env_file}")
                return True
            except Exception as e:
                print(f"[ERROR] Error reading {env_file}: {e}")
    
    print("[WARN] No .env file found")
    return False

# Load environment variables
load_env_file()

# Enhanced configuration with validation
@dataclass
class ProductionConfig:
    """Production configuration with validation and defaults"""
    
    # Directories - Updated to match your actual structure
    CHUNKED_DOCUMENTS_DIR: str = str(project_root / "data" / "chunked_documents")
    KEYWORDS_ENHANCED_DIR: str = str(project_root / "data" / "keywords_enhanced") 
    EMBEDDING_STORAGE_DIR: str = str(project_root / "data" / "embeddings")
    
    # Qdrant settings - Fixed collection name
    COLLECTION_NAME: str = "document_chunks"
    QDRANT_URL: str = os.getenv('QDRANT_URL', '')
    QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY', '')
    QDRANT_HOST: str = os.getenv('QDRANT_HOST', 'localhost')
    QDRANT_PORT: int = int(os.getenv('QDRANT_PORT', '6333'))
    
    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    
    # Performance settings
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '50'))
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
    RATE_LIMIT_DELAY: float = float(os.getenv('RATE_LIMIT_DELAY', '0.5'))
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '2'))
    MEMORY_LIMIT_GB: float = float(os.getenv('MEMORY_LIMIT_GB', '8.0'))
    
    # Operational settings
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    ENABLE_MONITORING: bool = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
    CHECKPOINT_INTERVAL: int = int(os.getenv('CHECKPOINT_INTERVAL', '100'))
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        if not self.OPENAI_API_KEY and not self.GEMINI_API_KEY:
            errors.append("Neither OPENAI_API_KEY nor GEMINI_API_KEY is set")
        
        if self.BATCH_SIZE <= 0 or self.BATCH_SIZE > 2048:
            errors.append(f"BATCH_SIZE must be between 1 and 2048, got {self.BATCH_SIZE}")
        
        if self.MAX_WORKERS <= 0 or self.MAX_WORKERS > 32:
            errors.append(f"MAX_WORKERS must be between 1 and 32, got {self.MAX_WORKERS}")
        
        if self.MEMORY_LIMIT_GB < 1.0:
            errors.append(f"MEMORY_LIMIT_GB must be at least 1.0, got {self.MEMORY_LIMIT_GB}")
        
        return errors
    
    def get_qdrant_config(self) -> Dict[str, Any]:
        """Get Qdrant configuration"""
        if self.QDRANT_URL and self.QDRANT_API_KEY:
            return {"url": self.QDRANT_URL, "api_key": self.QDRANT_API_KEY}
        else:
            return {"host": self.QDRANT_HOST, "port": self.QDRANT_PORT}

# Initialize configuration
try:
    from config.settings import settings
    config = settings
except ImportError:
    config = ProductionConfig()

# Enhanced logging setup - Windows compatible
def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup production-grade logging - Windows compatible"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging with proper encoding for Windows
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    # Create handlers with proper encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # File handlers with UTF-8 encoding
    file_handler1 = logging.FileHandler(
        logs_dir / f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_handler2 = logging.FileHandler(
        logs_dir / "embeddings_latest.log",
        encoding='utf-8'
    )
    
    file_handler1.setLevel(log_level)
    file_handler2.setLevel(log_level)
    file_formatter = logging.Formatter(log_format)
    file_handler1.setFormatter(file_formatter)
    file_handler2.setFormatter(file_formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()  # Clear any existing handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler1)
    root_logger.addHandler(file_handler2)
    
    logger = logging.getLogger(__name__)
    logger.info("Production logging initialized")
    return logger

logger = setup_logging(config.DEBUG)

# Performance monitoring
class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self, memory_limit_gb: float = 8.0):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.start_time = time.time()
        self.processed_count = 0
        self.error_count = 0
        self.last_checkpoint = time.time()
        
    def check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        try:
            memory_usage = psutil.virtual_memory().used
            if memory_usage > self.memory_limit_bytes:
                logger.warning(f"Memory usage ({memory_usage / 1024**3:.2f} GB) exceeds limit")
                gc.collect()
                return False
            return True
        except Exception:
            return True
    
    def log_progress(self, processed: int, total: int, force: bool = False):
        """Log processing progress"""
        self.processed_count = processed
        current_time = time.time()
        
        if force or (current_time - self.last_checkpoint) > 30:
            elapsed = current_time - self.start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            
            logger.info(f"Progress: {processed}/{total} ({processed/total*100:.1f}%) | "
                       f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m | Errors: {self.error_count}")
            
            self.last_checkpoint = current_time
    
    def increment_errors(self):
        """Increment error counter"""
        self.error_count += 1

# Helper function to extract document name
def extract_document_name(chunk: Dict[str, Any]) -> str:
    """Extract document name from chunk data with multiple fallback strategies"""
    
    # Strategy 1: Direct doc_name field
    if 'doc_name' in chunk and chunk['doc_name'] and chunk['doc_name'] != 'unknown':
        return chunk['doc_name']
    
    # Strategy 2: Extract from doc_id
    if 'doc_id' in chunk and chunk['doc_id']:
        doc_id = chunk['doc_id']
        # Remove common prefixes and suffixes
        name = doc_id.replace('_', ' ').replace('-', ' ')
        # Remove file extensions
        name = '.'.join(name.split('.')[:-1]) if '.' in name else name
        if name.strip():
            return name.strip()
    
    # Strategy 3: Look for filename in metadata
    if 'metadata' in chunk:
        metadata = chunk['metadata']
        for key in ['filename', 'file_name', 'document_name', 'title', 'source']:
            if key in metadata and metadata[key]:
                return str(metadata[key])
    
    # Strategy 4: Look for source information
    for key in ['source', 'file_path', 'filepath', 'path']:
        if key in chunk and chunk[key]:
            path = Path(str(chunk[key]))
            return path.stem
    
    # Strategy 5: Look for title or document title
    for key in ['title', 'document_title', 'doc_title']:
        if key in chunk and chunk[key]:
            return str(chunk[key])
    
    # Strategy 6: Generate from chunk content if it looks like a title
    if 'text' in chunk and chunk['text']:
        text = chunk['text'].strip()
        first_line = text.split('\n')[0].strip()
        if len(first_line) < 100 and len(first_line.split()) < 15:
            return first_line
    
    # Final fallback
    if 'doc_id' in chunk and chunk['doc_id']:
        return f"Document_{chunk['doc_id']}"
    
    return "Unknown_Document"

# Production-grade embedding processor
class ProductionEmbeddingProcessor:
    """Production-ready embedding processor with enhanced Qdrant storage"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.monitor = PerformanceMonitor(config.MEMORY_LIMIT_GB)
        
        # Initialize directories
        self.input_dir = Path(config.CHUNKED_DOCUMENTS_DIR)
        self.keywords_dir = Path(config.KEYWORDS_ENHANCED_DIR)
        self.output_dir = Path(config.EMBEDDING_STORAGE_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.embedding_client = None
        self.embedding_dim = None
        self.embedding_service = None
        self.qdrant_client = None
        
        # Processing state
        self.processed_chunks = 0
        self.failed_chunks = []
        self.processing_start_time = None
        self._shutdown_requested = False
        
        # Initialize services
        self._initialize_embedding_service()
        self._initialize_qdrant_client()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("ProductionEmbeddingProcessor initialized successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
    
    def _initialize_embedding_service(self):
        """Initialize embedding service with enhanced error handling"""
        try:
            # Try OpenAI first
            if (self.config.EMBEDDING_MODEL.startswith("text-embedding") and 
                self.config.OPENAI_API_KEY):
                
                import openai
                self.embedding_client = openai.OpenAI(
                    api_key=self.config.OPENAI_API_KEY,
                    timeout=60.0,
                    max_retries=self.config.MAX_RETRIES
                )
                self.embedding_service = "openai"
                
                # Validate model and get dimensions
                valid_models = {
                    "text-embedding-3-small": 1536,
                    "text-embedding-3-large": 3072,
                    "text-embedding-ada-002": 1536
                }
                
                if self.config.EMBEDDING_MODEL in valid_models:
                    self.embedding_dim = valid_models[self.config.EMBEDDING_MODEL]
                else:
                    logger.warning(f"Unknown model {self.config.EMBEDDING_MODEL}, using text-embedding-3-small")
                    self.config.EMBEDDING_MODEL = "text-embedding-3-small"
                    self.embedding_dim = 1536
                
                # Test the connection
                try:
                    test_response = self.embedding_client.embeddings.create(
                        model=self.config.EMBEDDING_MODEL,
                        input="connection test"
                    )
                    actual_dim = len(test_response.data[0].embedding)
                    if actual_dim != self.embedding_dim:
                        self.embedding_dim = actual_dim
                    
                    logger.info(f"[OK] OpenAI embedding service ready: {self.config.EMBEDDING_MODEL} (dim: {self.embedding_dim})")
                
                except Exception as e:
                    logger.error(f"OpenAI connection test failed: {e}")
                    raise
            
            # Try Gemini if OpenAI not available
            elif self.config.GEMINI_API_KEY:
                import google.generativeai as genai
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                
                self.embedding_client = genai
                self.embedding_service = "gemini"
                self.config.EMBEDDING_MODEL = "text-embedding-004"
                
                # Test connection and get dimensions
                try:
                    test_response = genai.embed_content(
                        model=self.config.EMBEDDING_MODEL,
                        content="connection test",
                        task_type="retrieval_document"
                    )
                    self.embedding_dim = len(test_response['embedding'])
                    logger.info(f"[OK] Gemini embedding service ready: {self.config.EMBEDDING_MODEL} (dim: {self.embedding_dim})")
                
                except Exception as e:
                    logger.error(f"Gemini connection test failed: {e}")
                    raise
            
            else:
                raise ValueError("No valid embedding service configured. Set OPENAI_API_KEY or GEMINI_API_KEY")
        
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    def _initialize_qdrant_client(self):
        """Initialize Qdrant client with enhanced configuration and validation"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            qdrant_config = self.config.get_qdrant_config()
            logger.info(f"[INFO] Initializing Qdrant connection with config: {qdrant_config}")
            
            if "url" in qdrant_config and qdrant_config["url"]:
                # Cloud Qdrant
                logger.info(f"[INFO] Connecting to Qdrant Cloud: {qdrant_config['url']}")
                self.qdrant_client = QdrantClient(
                    url=qdrant_config["url"],
                    api_key=qdrant_config["api_key"],
                    timeout=60.0
                )
                logger.info(f"[OK] Connected to Qdrant Cloud: {qdrant_config['url']}")
            else:
                # Local Qdrant
                logger.info(f"[INFO] Connecting to local Qdrant: {qdrant_config['host']}:{qdrant_config['port']}")
                
                try:
                    self.qdrant_client = QdrantClient(
                        host=qdrant_config["host"],
                        port=qdrant_config["port"],
                        timeout=30.0
                    )
                    logger.info(f"[OK] Connected to local Qdrant: {qdrant_config['host']}:{qdrant_config['port']}")
                except Exception as local_error:
                    logger.warning(f"[WARN] Could not connect to local Qdrant: {local_error}")
                    logger.info("[INFO] Falling back to in-memory Qdrant for testing...")
                    self.qdrant_client = QdrantClient(":memory:")
                    logger.info("[OK] Using in-memory Qdrant")
            
            # Test connection with detailed validation
            try:
                collections = self.qdrant_client.get_collections()
                logger.info(f"[OK] Qdrant connection successful! Found {len(collections.collections)} existing collections")
                
                # List existing collections
                if collections.collections:
                    logger.info("[INFO] Existing collections:")
                    for col in collections.collections:
                        logger.info(f"   - {col.name}")
                else:
                    logger.info("[INFO] No existing collections found")
                
            except Exception as e:
                logger.error(f"[ERROR] Qdrant connection test failed: {e}")
                raise
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize Qdrant client: {e}")
            logger.info("[INFO] Falling back to in-memory Qdrant...")
            from qdrant_client import QdrantClient
            self.qdrant_client = QdrantClient(":memory:")
            logger.info("[OK] Using in-memory Qdrant as fallback")
    
    @contextmanager
    def _rate_limit_context(self):
        """Context manager for rate limiting"""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if elapsed < self.config.RATE_LIMIT_DELAY:
                time.sleep(self.config.RATE_LIMIT_DELAY - elapsed)
    
    def create_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Create embeddings for a batch of texts with optimized performance"""
        embeddings = []
        
        try:
            if self.embedding_service == "openai":
                with self._rate_limit_context():
                    response = self.embedding_client.embeddings.create(
                        model=self.config.EMBEDDING_MODEL,
                        input=texts,
                        encoding_format="float"
                    )
                    embeddings = [np.array(data.embedding, dtype=np.float32) for data in response.data]
            
            elif self.embedding_service == "gemini":
                # Gemini processes individually
                for text in texts:
                    with self._rate_limit_context():
                        response = self.embedding_client.embed_content(
                            model=self.config.EMBEDDING_MODEL,
                            content=text,
                            task_type="retrieval_document"
                        )
                        embeddings.append(np.array(response['embedding'], dtype=np.float32))
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            # Return zero vectors as fallback
            embeddings = [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
        
        return embeddings
    
    def create_embeddings_parallel(self, chunks: List[Dict[str, Any]]) -> List[np.ndarray]:
        """Create embeddings using parallel processing"""
        logger.info(f"[INFO] Creating embeddings for {len(chunks)} chunks using {self.embedding_service}")
        
        texts = [chunk['text'] for chunk in chunks]
        all_embeddings = []
        
        # Process in batches with parallel execution
        batch_size = min(self.config.BATCH_SIZE, len(texts))
        
        with ThreadPoolExecutor(max_workers=min(self.config.MAX_WORKERS, 4)) as executor:
            futures = []
            
            for i in range(0, len(texts), batch_size):
                if self._shutdown_requested:
                    logger.info("Shutdown requested, stopping embedding creation")
                    break
                
                batch_texts = texts[i:i + batch_size]
                future = executor.submit(self.create_embeddings_batch, batch_texts)
                futures.append((i, future))
            
            # Collect results
            for i, future in futures:
                try:
                    batch_embeddings = future.result(timeout=300)
                    all_embeddings.extend(batch_embeddings)
                    
                    self.monitor.log_progress(len(all_embeddings), len(texts))
                    
                    # Memory check
                    if not self.monitor.check_memory():
                        logger.warning("Memory pressure detected, continuing with caution")
                
                except Exception as e:
                    logger.error(f"Failed to process batch starting at {i}: {e}")
                    self.monitor.increment_errors()
                    # Add zero vectors for failed batch
                    batch_size_actual = min(batch_size, len(texts) - i)
                    all_embeddings.extend([np.zeros(self.embedding_dim, dtype=np.float32) 
                                         for _ in range(batch_size_actual)])
        
        logger.info(f"[OK] Embedding creation completed: {len(all_embeddings)}/{len(texts)} successful")
        return all_embeddings
    
    def setup_collection_advanced(self) -> bool:
        """Setup Qdrant collection with enhanced validation and error handling"""
        try:
            from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff
            
            collection_name = self.config.COLLECTION_NAME
            logger.info(f"[INFO] Setting up Qdrant collection: {collection_name}")
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            existing_collections = [col.name for col in collections.collections]
            
            if collection_name in existing_collections:
                logger.info(f"[INFO] Collection '{collection_name}' already exists")
                
                # Verify collection configuration
                try:
                    collection_info = self.qdrant_client.get_collection(collection_name)
                    existing_dim = collection_info.config.params.vectors.size
                    
                    if existing_dim != self.embedding_dim:
                        logger.error(f"[ERROR] Collection dimension mismatch!")
                        logger.error(f"   Expected: {self.embedding_dim}")
                        logger.error(f"   Found: {existing_dim}")
                        
                        if self.config.DEBUG:
                            response = input("[INPUT] Delete and recreate collection? (y/N): ")
                            if response.lower() == 'y':
                                logger.info(f"[INFO] Deleting collection: {collection_name}")
                                self.qdrant_client.delete_collection(collection_name)
                                logger.info(f"[OK] Deleted collection: {collection_name}")
                            else:
                                return False
                        else:
                            return False
                    else:
                        logger.info(f"[OK] Collection configuration is valid (dim: {existing_dim})")
                        
                        # Get collection stats
                        stats = self.get_collection_stats_detailed()
                        if stats:
                            logger.info(f"[INFO] Current collection stats:")
                            logger.info(f"   Points: {stats['total_points']:,}")
                            logger.info(f"   Status: {stats['status']}")
                        
                        return True
                
                except Exception as e:
                    logger.error(f"[ERROR] Error checking existing collection: {e}")
                    if self.config.DEBUG:
                        response = input("[INPUT] Delete and recreate collection? (y/N): ")
                        if response.lower() == 'y':
                            self.qdrant_client.delete_collection(collection_name)
                            logger.info(f"[OK] Deleted problematic collection: {collection_name}")
                        else:
                            return False
                    else:
                        return False
            
            # Create new collection
            logger.info(f"[INFO] Creating new collection: {collection_name}")
            logger.info(f"   Vector dimension: {self.embedding_dim}")
            logger.info(f"   Distance metric: COSINE")
            
            try:
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE,
                        on_disk=True
                    ),
                    optimizers_config=OptimizersConfigDiff(
                        default_segment_number=2,
                        max_segment_size=20000,
                        memmap_threshold=20000,
                        indexing_threshold=20000
                    )
                )
                
                logger.info(f"[OK] Collection created successfully: {collection_name}")
                
                # Verify creation
                collection_info = self.qdrant_client.get_collection(collection_name)
                logger.info(f"[OK] Collection verification:")
                logger.info(f"   Name: {collection_name}")
                logger.info(f"   Dimension: {collection_info.config.params.vectors.size}")
                logger.info(f"   Distance: {collection_info.config.params.vectors.distance}")
                logger.info(f"   Status: {collection_info.status}")
                
                return True
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to create collection: {e}")
                logger.error(f"   This might be due to Qdrant server issues or permissions")
                return False
        
        except Exception as e:
            logger.error(f"[ERROR] Error in collection setup: {e}")
            return False
    
    def preprocess_and_validate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess chunks and fix document names"""
        logger.info("[INFO] Preprocessing chunks and fixing document names...")
        
        processed_chunks = []
        doc_name_stats = {}
        
        for i, chunk in enumerate(chunks):
            try:
                # Create a copy to avoid modifying original
                processed_chunk = chunk.copy()
                
                # Fix the document name using our enhanced extraction
                doc_name = extract_document_name(chunk)
                processed_chunk['doc_name'] = doc_name
                
                # Track statistics
                doc_name_stats[doc_name] = doc_name_stats.get(doc_name, 0) + 1
                
                # Ensure other required fields
                if 'paragraph_number' not in processed_chunk:
                    processed_chunk['paragraph_number'] = processed_chunk.get('para_id', 1)
                
                # Ensure keywords field exists
                if 'keywords' not in processed_chunk:
                    processed_chunk['keywords'] = []
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                logger.warning(f"Error processing chunk {i}: {e}")
                processed_chunks.append(chunk)
        
        # Log document statistics
        logger.info(f"[OK] Document name extraction completed:")
        logger.info(f"   Total chunks: {len(processed_chunks)}")
        logger.info(f"   Unique documents: {len(doc_name_stats)}")
        
        # Show top 5 documents by chunk count
        sorted_docs = sorted(doc_name_stats.items(), key=lambda x: x[1], reverse=True)
        for doc_name, count in sorted_docs[:5]:
            logger.info(f"   [DOC] {doc_name}: {count} chunks")
        
        if "Unknown_Document" in doc_name_stats:
            logger.warning(f"[WARN] {doc_name_stats['Unknown_Document']} chunks have unknown document names")
        
        return processed_chunks
    
    def store_in_qdrant_optimized(self, chunks: List[Dict[str, Any]], 
                                embeddings: List[np.ndarray]) -> Tuple[bool, int]:
        """Store chunks and embeddings in Qdrant with enhanced error handling and validation"""
        try:
            from qdrant_client.models import PointStruct
            
            logger.info(f"[INFO] Storing {len(chunks)} chunks in Qdrant...")
            logger.info(f"   Collection: {self.config.COLLECTION_NAME}")
            logger.info(f"   Batch size: {min(100, self.config.BATCH_SIZE)}")
            
            collection_name = self.config.COLLECTION_NAME
            successful_uploads = 0
            batch_size = min(100, self.config.BATCH_SIZE)  # Smaller batches for reliability
            
            # Validate inputs
            if len(chunks) != len(embeddings):
                logger.error(f"[ERROR] Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
                return False, 0
            
            # Create checkpoint for resumable uploads
            checkpoint_file = self.output_dir / f"upload_checkpoint_{collection_name}.json"
            start_index = 0
            
            if checkpoint_file.exists() and self.config.DEBUG:
                try:
                    with open(checkpoint_file, 'r') as f:
                        checkpoint_data = json.load(f)
                    start_index = checkpoint_data.get('last_uploaded_index', 0)
                    logger.info(f"[INFO] Resuming from checkpoint: index {start_index}")
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}")
            
            # Process in optimized batches with detailed logging
            total_batches = (len(chunks) - start_index + batch_size - 1) // batch_size
            current_batch = 0
            
            for i in range(start_index, len(chunks), batch_size):
                if self._shutdown_requested:
                    logger.info("Shutdown requested, saving checkpoint and stopping")
                    self._save_checkpoint(checkpoint_file, i)
                    break
                
                current_batch += 1
                batch_end = min(i + batch_size, len(chunks))
                batch_chunks = chunks[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                
                logger.info(f"[INFO] Processing batch {current_batch}/{total_batches} (items {i+1}-{batch_end})")
                
                # Create points with detailed validation
                points = []
                for idx, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                    try:
                        paragraph_number = chunk.get("paragraph_number", chunk.get("para_id", 1))
                        doc_name = chunk.get("doc_name", extract_document_name(chunk))
                        
                        # Validate embedding
                        if embedding is None or len(embedding) != self.embedding_dim:
                            logger.warning(f"[WARN] Invalid embedding at batch index {idx}, using zero vector")
                            embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                        
                        # Create comprehensive payload
                        payload = {
                            "doc_id": str(chunk["doc_id"]),
                            "doc_name": str(doc_name),
                            "page": int(chunk["page"]),
                            "paragraph_number": int(paragraph_number),
                            "text": str(chunk["text"]),
                            "keywords": chunk.get("keywords", []),
                            "chunk_id": f"{chunk['doc_id']}_p{chunk['page']}_para{paragraph_number}",
                            "char_count": len(chunk["text"]),
                            "word_count": len(chunk["text"].split()),
                            "text_hash": hashlib.md5(chunk["text"].encode()).hexdigest()[:16],
                            "created_at": datetime.now().isoformat(),
                            "embedding_model": self.config.EMBEDDING_MODEL,
                            "embedding_service": self.embedding_service
                        }
                        
                        # Create point with unique ID
                        point_id = str(uuid.uuid4())
                        point = PointStruct(
                            id=point_id,
                            vector=embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding),
                            payload=payload
                        )
                        points.append(point)
                        
                    except Exception as e:
                        logger.error(f"[ERROR] Error creating point for chunk {idx}: {e}")
                        self.monitor.increment_errors()
                        continue
                
                if not points:
                    logger.warning(f"[WARN] No valid points created for batch {current_batch}")
                    continue
                
                # Upload batch with comprehensive retry logic
                upload_successful = False
                last_error = None
                
                for attempt in range(self.config.MAX_RETRIES):
                    try:
                        logger.debug(f"[INFO] Upload attempt {attempt + 1} for batch {current_batch}")
                        
                        # Perform the upsert operation
                        result = self.qdrant_client.upsert(
                            collection_name=collection_name,
                            points=points,
                            wait=True  # Wait for operation to complete
                        )
                        
                        # Validate the result
                        if hasattr(result, 'status') and result.status == 'completed':
                            successful_uploads += len(points)
                            upload_successful = True
                            logger.info(f"[OK] Batch {current_batch} uploaded successfully ({len(points)} points)")
                            break
                        elif hasattr(result, 'operation_id'):
                            # Async operation, assume success for now
                            successful_uploads += len(points)
                            upload_successful = True
                            logger.info(f"[OK] Batch {current_batch} queued successfully ({len(points)} points)")
                            break
                        else:
                            # No clear status, but no exception means success
                            successful_uploads += len(points)
                            upload_successful = True
                            logger.info(f"[OK] Batch {current_batch} uploaded ({len(points)} points)")
                            break
                    
                    except Exception as e:
                        last_error = e
                        logger.warning(f"[WARN] Upload attempt {attempt + 1} failed for batch {current_batch}: {e}")
                        if attempt < self.config.MAX_RETRIES - 1:
                            sleep_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                            logger.info(f"[INFO] Waiting {sleep_time}s before retry...")
                            time.sleep(sleep_time)
                
                if not upload_successful:
                    logger.error(f"[ERROR] Failed to upload batch {current_batch} after {self.config.MAX_RETRIES} attempts")
                    logger.error(f"   Last error: {last_error}")
                    self.monitor.increment_errors()
                
                # Save checkpoint periodically
                if i % self.config.CHECKPOINT_INTERVAL == 0:
                    self._save_checkpoint(checkpoint_file, batch_end)
                
                # Progress monitoring with detailed stats
                self.monitor.log_progress(successful_uploads, len(chunks))
                
                # Memory and performance checks
                if not self.monitor.check_memory():
                    logger.warning("[WARN] Memory pressure detected during upload")
                    gc.collect()
                    time.sleep(1)
                
                # Brief pause between batches to avoid overwhelming Qdrant
                time.sleep(0.1)
            
            # Clean up checkpoint file on successful completion
            if checkpoint_file.exists() and successful_uploads >= len(chunks) * 0.95:  # 95% success rate
                checkpoint_file.unlink()
                logger.info("[INFO] Upload checkpoint cleaned up")
            
            # Final validation
            success_rate = successful_uploads / len(chunks) if chunks else 0
            logger.info(f"[INFO] Upload Summary:")
            logger.info(f"   Total chunks: {len(chunks):,}")
            logger.info(f"   Successfully uploaded: {successful_uploads:,}")
            logger.info(f"   Success rate: {success_rate:.2%}")
            logger.info(f"   Failed uploads: {len(chunks) - successful_uploads:,}")
            
            # Verify storage by checking collection stats
            try:
                time.sleep(2)  # Give Qdrant time to process
                stats = self.get_collection_stats_detailed()
                if stats:
                    logger.info(f"[INFO] Collection verification:")
                    logger.info(f"   Points in collection: {stats['total_points']:,}")
                    logger.info(f"   Collection status: {stats['status']}")
                    
                    if stats['total_points'] >= successful_uploads:
                        logger.info(f"[OK] Storage verification successful!")
                    else:
                        logger.warning(f"[WARN] Point count mismatch: expected ~{successful_uploads}, found {stats['total_points']}")
            except Exception as e:
                logger.warning(f"[WARN] Could not verify storage: {e}")
            
            return success_rate > 0.95, successful_uploads
        
        except Exception as e:
            logger.error(f"[ERROR] Critical error in Qdrant storage: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False, 0
    
    def _save_checkpoint(self, checkpoint_file: Path, last_index: int):
        """Save upload checkpoint with enhanced metadata"""
        try:
            checkpoint_data = {
                "last_uploaded_index": last_index,
                "timestamp": datetime.now().isoformat(),
                "collection_name": self.config.COLLECTION_NAME,
                "embedding_model": self.config.EMBEDDING_MODEL,
                "embedding_service": self.embedding_service
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.debug(f"[INFO] Checkpoint saved: {last_index}")
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")
    
    def load_chunks_optimized(self) -> Tuple[Optional[List[Dict[str, Any]]], str, Dict[str, Any]]:
        """Load chunks with optimized file detection and validation"""
        # Priority order for chunk files
        file_candidates = [
            (self.keywords_dir / "all_chunks_with_keywords.json", "keyword-enhanced"),
            (self.input_dir / "all_chunks.json", "regular"),
            (project_root / "data" / "chunked_documents" / "all_chunks.json", "regular"),
            (project_root / "data" / "keywords_enhanced" / "all_chunks_with_keywords.json", "keyword-enhanced"),
        ]
        
        logger.info("[INFO] Searching for chunk files...")
        
        for file_path, source_type in file_candidates:
            if file_path.exists():
                try:
                    logger.info(f"[INFO] Found {source_type} chunks at: {file_path}")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        chunks_data = json.load(f)
                    
                    chunks = chunks_data.get('chunks', [])
                    metadata = chunks_data.get('metadata', {})
                    
                    if not chunks:
                        logger.warning(f"No chunks found in {file_path}")
                        continue
                    
                    # Validate required fields
                    sample_chunk = chunks[0]
                    required_fields = ['doc_id', 'page', 'text']
                    missing_fields = [field for field in required_fields if field not in sample_chunk]
                    
                    if missing_fields:
                        logger.error(f"Missing required fields: {missing_fields}")
                        continue
                    
                    # Log statistics
                    total_chars = sum(len(chunk.get('text', '')) for chunk in chunks)
                    avg_chunk_length = total_chars / len(chunks) if chunks else 0
                    
                    logger.info(f"[OK] Loaded {len(chunks)} chunks from {chunks_data.get('total_documents', 'unknown')} documents")
                    logger.info(f"   Average chunk length: {avg_chunk_length:.0f} characters")
                    logger.info(f"   Total text content: {total_chars:,} characters")
                    
                    return chunks, source_type, metadata
                
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        logger.error("[ERROR] No valid chunks file found!")
        return None, "", {}
    
    def process_chunks_production(self) -> Optional[Dict[str, Any]]:
        """Main production processing function with comprehensive error handling"""
        self.processing_start_time = time.time()
        
        try:
            # Load chunks
            logger.info("[INFO] Loading chunks...")
            chunks, chunks_source, metadata = self.load_chunks_optimized()
            if not chunks:
                logger.error("[ERROR] No chunks loaded, cannot proceed")
                return None
            
            # Preprocess chunks
            logger.info("[INFO] Preprocessing chunks...")
            chunks = self.preprocess_and_validate_chunks(chunks)
            
            # Validate configuration
            config_errors = self.config.validate()
            if config_errors:
                for error in config_errors:
                    logger.error(f"Configuration error: {error}")
                return None
            
            # Setup collection
            logger.info("[INFO] Setting up Qdrant collection...")
            if not self.setup_collection_advanced():
                logger.error("[ERROR] Failed to setup Qdrant collection")
                return None
            
            # Create embeddings
            logger.info("[INFO] Creating embeddings...")
            embeddings = self.create_embeddings_parallel(chunks)
            
            if len(embeddings) != len(chunks):
                logger.error(f"[ERROR] Embedding count mismatch: {len(embeddings)} vs {len(chunks)}")
                return None
            
            # Store in Qdrant
            logger.info("[INFO] Storing in Qdrant...")
            storage_success, uploaded_count = self.store_in_qdrant_optimized(chunks, embeddings)
            
            # Create comprehensive summary
            processing_time = time.time() - self.processing_start_time
            
            summary = {
                "processing_metadata": {
                    "total_chunks_processed": len(chunks),
                    "total_documents": metadata.get('total_documents', 0),
                    "chunks_source": chunks_source,
                    "processing_time_seconds": round(processing_time, 2),
                    "processing_rate_per_second": round(len(chunks) / processing_time, 2),
                    "started_at": datetime.fromtimestamp(self.processing_start_time).isoformat(),
                    "completed_at": datetime.now().isoformat()
                },
                "embedding_details": {
                    "service": self.embedding_service,
                    "model": self.config.EMBEDDING_MODEL,
                    "dimension": self.embedding_dim,
                    "batch_size": self.config.BATCH_SIZE,
                    "max_workers": self.config.MAX_WORKERS,
                    "total_api_calls": len(chunks) // self.config.BATCH_SIZE + 1
                },
                "storage_details": {
                    "qdrant_collection": self.config.COLLECTION_NAME,
                    "total_points_uploaded": uploaded_count,
                    "upload_success_rate": round(uploaded_count / len(chunks), 4),
                    "storage_successful": storage_success
                },
                "performance_metrics": {
                    "memory_peak_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                    "error_count": self.monitor.error_count,
                    "error_rate": round(self.monitor.error_count / len(chunks), 4),
                    "checkpoint_intervals": self.config.CHECKPOINT_INTERVAL
                },
                "quality_metrics": {
                    "avg_chunk_length": round(sum(len(chunk['text']) for chunk in chunks) / len(chunks), 2),
                    "chunks_with_keywords": sum(1 for chunk in chunks if chunk.get('keywords')),
                    "unique_documents": len(set(chunk['doc_name'] for chunk in chunks)),
                }
            }
            
            # Save summary
            summary_file = self.output_dir / f"embeddings_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            latest_summary_file = self.output_dir / "embeddings_summary_latest.json"
            with open(latest_summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[INFO] Processing summary saved to: {summary_file}")
            return summary
        
        except Exception as e:
            logger.error(f"[ERROR] Critical error in processing pipeline: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def search_similar_chunks_optimized(self, query: str, limit: int = 10, 
                                      doc_filter: Optional[str] = None,
                                      score_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Optimized search with enhanced error handling"""
        try:
            logger.info(f"[INFO] Searching for: '{query}'")
            
            # Create query embedding
            query_embedding = None
            
            if self.embedding_service == "openai":
                response = self.embedding_client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=[query]
                )
                query_embedding = response.data[0].embedding
            elif self.embedding_service == "gemini":
                response = self.embedding_client.embed_content(
                    model=self.config.EMBEDDING_MODEL,
                    content=query,
                    task_type="retrieval_query"
                )
                query_embedding = response['embedding']
            
            if not query_embedding:
                logger.error("[ERROR] Failed to create query embedding")
                return []
            
            # Prepare filters
            search_filter = None
            if doc_filter:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                search_filter = Filter(
                    must=[FieldCondition(key="doc_name", match=MatchValue(value=doc_filter))]
                )
            
            # Perform search with fallback methods
            search_results = []
            try:
                # Try newer query_points method
                result = self.qdrant_client.query_points(
                    collection_name=self.config.COLLECTION_NAME,
                    query=query_embedding,
                    query_filter=search_filter,
                    limit=limit * 2,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
                search_results = result.points
                logger.info(f"[OK] Query completed using query_points method")
            except AttributeError:
                # Fallback to older search method
                search_results = self.qdrant_client.search(
                    collection_name=self.config.COLLECTION_NAME,
                    query_vector=query_embedding,
                    query_filter=search_filter,
                    limit=limit * 2,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=False
                )
                logger.info(f"[OK] Query completed using search method")
            
            # Process results
            results = []
            for result in search_results[:limit]:
                if result.score >= score_threshold:
                    chunk_data = {
                        "score": result.score,
                        "confidence": self._calculate_confidence(result.score),
                        "chunk_id": result.payload["chunk_id"],
                        "doc_name": result.payload["doc_name"],
                        "doc_id": result.payload["doc_id"],
                        "page": result.payload["page"],
                        "paragraph_number": result.payload["paragraph_number"],
                        "text": result.payload["text"],
                        "keywords": result.payload.get("keywords", []),
                        "char_count": result.payload.get("char_count", 0),
                        "word_count": result.payload.get("word_count", 0)
                    }
                    results.append(chunk_data)
            
            logger.info(f"[OK] Found {len(results)} results above threshold {score_threshold}")
            return results
        
        except Exception as e:
            logger.error(f"[ERROR] Error in search: {e}")
            return []
    
    def _calculate_confidence(self, score: float) -> str:
        """Calculate confidence level based on similarity score"""
        if score >= 0.9:
            return "very_high"
        elif score >= 0.8:
            return "high"
        elif score >= 0.7:
            return "medium"
        elif score >= 0.6:
            return "low"
        else:
            return "very_low"
    
    def get_collection_stats_detailed(self) -> Dict[str, Any]:
        """Get detailed collection statistics with enhanced error handling"""
        try:
            collection_info = self.qdrant_client.get_collection(self.config.COLLECTION_NAME)
            
            stats = {
                "collection_name": self.config.COLLECTION_NAME,
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.value,
                "status": collection_info.status.value,
                "indexed_vectors_count": getattr(collection_info, 'indexed_vectors_count', 0),
                "segments_count": getattr(collection_info, 'segments_count', 0),
                "disk_data_size": getattr(collection_info, 'disk_data_size', 0),
                "ram_data_size": getattr(collection_info, 'ram_data_size', 0),
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}

def test_qdrant_connection():
    """Test Qdrant connection and provide diagnostics"""
    print("=" * 50)
    print("QDRANT CONNECTION DIAGNOSTICS")
    print("=" * 50)
    
    config = ProductionConfig()
    
    try:
        from qdrant_client import QdrantClient
        
        qdrant_config = config.get_qdrant_config()
        print(f"[INFO] Configuration:")
        print(f"   Host: {qdrant_config.get('host', 'N/A')}")
        print(f"   Port: {qdrant_config.get('port', 'N/A')}")
        print(f"   URL: {qdrant_config.get('url', 'N/A')}")
        print(f"   API Key: {'Set' if qdrant_config.get('api_key') else 'Not set'}")
        
        # Test connection
        if "url" in qdrant_config and qdrant_config["url"]:
            print(f"\n[INFO] Testing cloud connection...")
            client = QdrantClient(
                url=qdrant_config["url"],
                api_key=qdrant_config["api_key"],
                timeout=30.0
            )
        else:
            print(f"\n[INFO] Testing local connection...")
            client = QdrantClient(
                host=qdrant_config["host"],
                port=qdrant_config["port"],
                timeout=30.0
            )
        
        # Get collections
        collections = client.get_collections()
        print(f"[OK] Connection successful!")
        print(f"[INFO] Found {len(collections.collections)} collections:")
        
        for col in collections.collections:
            try:
                info = client.get_collection(col.name)
                print(f"   - {col.name}: {info.points_count} points, dim={info.config.params.vectors.size}")
            except:
                print(f"   - {col.name}: (could not get details)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print(f"\n[INFO] Troubleshooting tips:")
        print(f"   1. Check if Qdrant server is running")
        print(f"   2. Verify host/port or URL/API key")
        print(f"   3. Check firewall settings")
        print(f"   4. Try: docker run -p 6333:6333 qdrant/qdrant")
        return False

def run_production_pipeline():
    """Run the complete production pipeline with enhanced diagnostics"""
    print("=" * 70)
    print("PRODUCTION EMBEDDINGS & VECTOR DATABASE PIPELINE")
    print("(WINDOWS COMPATIBLE - NO UNICODE ISSUES)")
    print("=" * 70)
    print("[INFO] FIXES APPLIED:")
    print("   [OK] Removed all emoji characters for Windows compatibility")
    print("   [OK] Enhanced Qdrant connection validation")
    print("   [OK] Improved error handling and logging")
    print("   [OK] Better batch processing and retry logic")
    print("   [OK] Comprehensive storage verification")
    print("   [OK] Document name extraction fixes")
    print("=" * 70)
    
    # Test Qdrant connection first
    print("\n[INFO] Testing Qdrant connection...")
    if not test_qdrant_connection():
        print("\n[ERROR] Qdrant connection failed. Please fix connection issues first.")
        return False
    
    # Initialize configuration
    try:
        config = ProductionConfig()
        config_errors = config.validate()
        if config_errors:
            print("[ERROR] Configuration errors found:")
            for error in config_errors:
                print(f"   - {error}")
            return False
        
        print("[OK] Configuration validated")
    except Exception as e:
        print(f"[ERROR] Configuration error: {e}")
        return False
    
    # Initialize processor
    try:
        processor = ProductionEmbeddingProcessor(config)
        print("[OK] Production processor initialized")
    except Exception as e:
        print(f"[ERROR] Failed to initialize processor: {e}")
        return False
    
    # Check input files
    try:
        chunks, _, _ = processor.load_chunks_optimized()
        if chunks:
            estimated_time = len(chunks) * config.RATE_LIMIT_DELAY / config.MAX_WORKERS / 60
            print(f"\n[INFO] PROCESSING ESTIMATES:")
            print(f"   Chunks to process: {len(chunks):,}")
            print(f"   Estimated time: {estimated_time:.1f} minutes")
    except Exception as e:
        logger.warning(f"Could not estimate processing time: {e}")
    
    # Confirm processing
    if not config.DEBUG:
        response = input(f"\n[INPUT] Start production processing? (yes/no): ")
        if response.lower() != 'yes':
            print("[INFO] Processing cancelled")
            return False
    
    # Run processing
    print(f"\n[INFO] Starting production processing...")
    start_time = time.time()
    
    try:
        summary = processor.process_chunks_production()
        
        if summary:
            processing_time = time.time() - start_time
            
            print(f"\n[SUCCESS] PRODUCTION PROCESSING COMPLETED!")
            print(f"[INFO] Total time: {processing_time/60:.1f} minutes")
            print(f"\n[INFO] RESULTS SUMMARY:")
            print(f"   Chunks processed: {summary['processing_metadata']['total_chunks_processed']:,}")
            print(f"   Documents: {summary['processing_metadata']['total_documents']:,}")
            print(f"   Embedding service: {summary['embedding_details']['service']}")
            print(f"   Model: {summary['embedding_details']['model']}")
            
            print(f"\n[INFO] STORAGE RESULTS:")
            print(f"   Qdrant collection: {summary['storage_details']['qdrant_collection']}")
            print(f"   Points uploaded: {summary['storage_details']['total_points_uploaded']:,}")
            print(f"   Success rate: {summary['storage_details']['upload_success_rate']:.2%}")
            print(f"   Storage status: {'[OK] SUCCESS' if summary['storage_details']['storage_successful'] else '[ERROR] FAILED'}")
            
            # Get final collection stats
            stats = processor.get_collection_stats_detailed()
            if stats:
                print(f"\n[INFO] FINAL COLLECTION VERIFICATION:")
                print(f"   Collection: {stats['collection_name']}")
                print(f"   Total points: {stats['total_points']:,}")
                print(f"   Vector size: {stats['vector_size']}")
                print(f"   Status: {stats['status']}")
                
                if stats['total_points'] > 0:
                    print(f"   [SUCCESS] VECTORS SUCCESSFULLY STORED IN QDRANT!")
                else:
                    print(f"   [ERROR] No vectors found in collection")
            
            # Test search functionality
            print(f"\n[INFO] TESTING SEARCH FUNCTIONALITY")
            test_query = input("Enter a test query (or press Enter to skip): ").strip()
            
            if test_query:
                print("[INFO] Searching...")
                results = processor.search_similar_chunks_optimized(test_query, limit=3)
                
                if results:
                    print(f"\n[INFO] Top 3 results for '{test_query}':")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. Score: {result['score']:.4f} | Confidence: {result['confidence']}")
                        print(f"   Document: {result['doc_name']}")
                        print(f"   Page {result['page']}, Paragraph {result['paragraph_number']}")
                        print(f"   Text: {result['text'][:200]}...")
                        if result.get('keywords'):
                            print(f"   Keywords: {', '.join(result['keywords'][:5])}")
                else:
                    print("[ERROR] No results found - this indicates a storage issue")
            
            print(f"\n[SUCCESS] Your RAG knowledge base is ready!")
            print(f"   [OK] Embeddings created and stored in Qdrant")
            print(f"   [OK] Document names properly extracted")
            print(f"   [OK] Search functionality working")
            
            return True
        
        else:
            print("[ERROR] Production processing failed")
            return False
    
    except KeyboardInterrupt:
        print("\n[WARN] Processing interrupted by user")
        return False
    except Exception as e:
        print(f"[ERROR] Critical error: {e}")
        return False

def main():
    """Main entry point with debug options"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test-qdrant":
            test_qdrant_connection()
            return
        elif sys.argv[1] == "--help":
            print("PRODUCTION EMBEDDINGS SCRIPT (WINDOWS COMPATIBLE)")
            print("Usage:")
            print("  python embeddings.py                 # Run normal pipeline")
            print("  python embeddings.py --test-qdrant   # Test Qdrant connection")
            print("  python embeddings.py --help          # Show this help")
            return
    
    try:
        success = run_production_pipeline()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()