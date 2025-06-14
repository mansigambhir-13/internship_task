"""
Production-Ready Keywords Extraction Module
Fast, robust, and scalable keyword extraction using multiple LLM providers
Supports Groq, OpenAI, Gemini with automatic fallback and batch processing
"""

import json
import os
import sys
import time
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import hashlib
import gc
import psutil
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# FIXED: Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent  # Go up 2 levels: keywords.py -> src -> project
sys.path.append(str(PROJECT_ROOT))

# Enhanced environment loading
def load_env_file():
    """Enhanced environment variable loading with validation"""
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            print("[OK] Loaded .env file using python-dotenv")
            return True
    except ImportError:
        pass
    
    # Enhanced fallback loading
    env_files = [Path('.env'), PROJECT_ROOT / '.env', Path.home() / '.env']
    
    for env_file in env_files:
        if env_file.exists():
            print(f"[INFO] Loading .env from: {env_file}")
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
                print(f"[OK] Loaded environment variables from: {env_file}")
                return True
            except Exception as e:
                print(f"[ERROR] Error reading {env_file}: {e}")
    
    print("[WARN] No .env file found")
    return False

load_env_file()

@dataclass
class ProductionConfig:
    """Production configuration with validation and defaults"""
    
    # Directories
    CHUNKED_DOCUMENTS_DIR: str = str(PROJECT_ROOT / "data" / "chunked_documents")
    KEYWORDS_ENHANCED_DIR: str = str(PROJECT_ROOT / "data" / "keywords_enhanced")
    
    # API Configuration
    GROQ_API_KEY: str = os.getenv('GROQ_API_KEY', '')
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    GEMINI_API_KEY: str = os.getenv('GEMINI_API_KEY', '')
    
    # Processing Settings
    BATCH_SIZE: int = int(os.getenv('KEYWORD_BATCH_SIZE', '25'))  # Smaller batches for stability
    MAX_WORKERS: int = int(os.getenv('KEYWORD_MAX_WORKERS', '3'))  # Conservative parallelism
    RATE_LIMIT_DELAY: float = float(os.getenv('KEYWORD_RATE_DELAY', '1.5'))  # Safe rate limiting
    MAX_RETRIES: int = int(os.getenv('KEYWORD_MAX_RETRIES', '3'))
    
    # Model Settings
    GROQ_MODEL: str = os.getenv('GROQ_MODEL', 'llama3-8b-8192')
    OPENAI_MODEL: str = os.getenv('OPENAI_KEYWORD_MODEL', 'gpt-3.5-turbo')
    GEMINI_MODEL: str = os.getenv('GEMINI_MODEL', 'gemini-pro')
    
    # Quality Settings
    MIN_TEXT_LENGTH: int = int(os.getenv('MIN_TEXT_LENGTH', '20'))
    MAX_TEXT_LENGTH: int = int(os.getenv('MAX_TEXT_LENGTH', '2000'))
    KEYWORDS_PER_CHUNK: int = int(os.getenv('KEYWORDS_PER_CHUNK', '5'))
    
    # Performance Settings
    MEMORY_LIMIT_GB: float = float(os.getenv('MEMORY_LIMIT_GB', '4.0'))
    CHECKPOINT_INTERVAL: int = int(os.getenv('CHECKPOINT_INTERVAL', '50'))
    ENABLE_CACHING: bool = os.getenv('ENABLE_KEYWORD_CACHING', 'true').lower() == 'true'
    
    # Operation Settings
    AUTO_CONFIRM: bool = os.getenv('AUTO_CONFIRM_KEYWORDS', 'false').lower() == 'true'
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    SAVE_INTERMEDIATE: bool = os.getenv('SAVE_INTERMEDIATE', 'true').lower() == 'true'
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check API keys
        if not any([self.GROQ_API_KEY, self.OPENAI_API_KEY, self.GEMINI_API_KEY]):
            errors.append("No API keys configured. Set GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY")
        
        # Validate processing parameters
        if self.BATCH_SIZE <= 0 or self.BATCH_SIZE > 100:
            errors.append(f"BATCH_SIZE must be between 1 and 100, got {self.BATCH_SIZE}")
        
        if self.MAX_WORKERS <= 0 or self.MAX_WORKERS > 10:
            errors.append(f"MAX_WORKERS must be between 1 and 10, got {self.MAX_WORKERS}")
        
        if self.KEYWORDS_PER_CHUNK < 1 or self.KEYWORDS_PER_CHUNK > 10:
            errors.append(f"KEYWORDS_PER_CHUNK must be between 1 and 10, got {self.KEYWORDS_PER_CHUNK}")
        
        return errors

# Initialize configuration
config = ProductionConfig()

# Enhanced logging setup
def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup production-grade logging"""
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Create logs directory
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging with proper encoding for Windows
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # File handlers with UTF-8 encoding
    file_handler = logging.FileHandler(
        logs_dir / f"keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    
    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    logger.info("Production keyword extraction logging initialized")
    return logger

logger = setup_logging(config.DEBUG)

# Performance monitoring
class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self, memory_limit_gb: float = 4.0):
        self.memory_limit_bytes = memory_limit_gb * 1024 * 1024 * 1024
        self.start_time = time.time()
        self.processed_count = 0
        self.error_count = 0
        self.last_checkpoint = time.time()
        self.api_call_count = 0
        
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
        """Log processing progress with enhanced metrics"""
        self.processed_count = processed
        current_time = time.time()
        
        if force or (current_time - self.last_checkpoint) > 30:
            elapsed = current_time - self.start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / rate if rate > 0 else 0
            
            logger.info(f"Progress: {processed}/{total} ({processed/total*100:.1f}%) | "
                       f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m | "
                       f"API calls: {self.api_call_count} | Errors: {self.error_count}")
            
            self.last_checkpoint = current_time
    
    def increment_errors(self):
        """Increment error counter"""
        self.error_count += 1
    
    def increment_api_calls(self):
        """Increment API call counter"""
        self.api_call_count += 1

class ProductionKeywordExtractor:
    """Production-ready keyword extractor with multiple LLM providers"""
    
    def __init__(self, config: ProductionConfig = None):
        self.config = config or ProductionConfig()
        self.monitor = PerformanceMonitor(self.config.MEMORY_LIMIT_GB)
        
        # Initialize directories
        self.input_dir = Path(self.config.CHUNKED_DOCUMENTS_DIR)
        self.output_dir = Path(self.config.KEYWORDS_ENHANCED_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing state
        self.processed_chunks = 0
        self.failed_chunks = []
        self.cache = {} if self.config.ENABLE_CACHING else None
        self._shutdown_requested = False
        
        # Initialize API clients
        self.api_clients = self._initialize_api_clients()
        self.primary_provider = self._select_primary_provider()
        
        logger.info("ProductionKeywordExtractor initialized successfully")
        logger.info(f"Primary provider: {self.primary_provider}")
        logger.info(f"Available providers: {list(self.api_clients.keys())}")
    
    def _initialize_api_clients(self) -> Dict[str, Any]:
        """Initialize all available API clients"""
        clients = {}
        
        # Groq client
        if self.config.GROQ_API_KEY:
            clients['groq'] = {
                'api_key': self.config.GROQ_API_KEY,
                'base_url': "https://api.groq.com/openai/v1/chat/completions",
                'model': self.config.GROQ_MODEL,
                'headers': {
                    "Authorization": f"Bearer {self.config.GROQ_API_KEY}",
                    "Content-Type": "application/json"
                }
            }
        
        # OpenAI client
        if self.config.OPENAI_API_KEY:
            try:
                import openai
                clients['openai'] = {
                    'client': openai.OpenAI(api_key=self.config.OPENAI_API_KEY),
                    'model': self.config.OPENAI_MODEL
                }
            except ImportError:
                logger.warning("OpenAI package not available")
        
        # Gemini client
        if self.config.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.GEMINI_API_KEY)
                clients['gemini'] = {
                    'client': genai.GenerativeModel(self.config.GEMINI_MODEL),
                    'model': self.config.GEMINI_MODEL
                }
            except ImportError:
                logger.warning("Google Generative AI package not available")
        
        return clients
    
    def _select_primary_provider(self) -> str:
        """Select primary API provider based on availability and preference"""
        # Priority order: Groq (fastest) -> OpenAI (most reliable) -> Gemini (backup)
        preference_order = ['groq', 'openai', 'gemini']
        
        for provider in preference_order:
            if provider in self.api_clients:
                return provider
        
        raise ValueError("No API providers available")
    
    def check_setup(self) -> bool:
        """Enhanced setup validation"""
        
        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            print("[ERROR] Configuration validation failed:")
            for error in config_errors:
                print(f"   - {error}")
            return False
        
        # Check input directory
        if not self.input_dir.exists():
            print(f"[ERROR] Input directory not found: {self.input_dir}")
            print("[INFO] Please run chunking step first")
            return False
        
        # Check consolidated chunks file
        consolidated_file = self.input_dir / "all_chunks.json"
        if not consolidated_file.exists():
            print(f"[ERROR] Chunks file not found: {consolidated_file}")
            print("[INFO] Please run chunking step first to create all_chunks.json")
            return False
        
        # Test API connections
        if not self._test_api_connections():
            print("[ERROR] API connection tests failed")
            return False
        
        print(f"[OK] Setup validation passed")
        print(f"   Primary provider: {self.primary_provider}")
        print(f"   Available providers: {len(self.api_clients)}")
        print(f"   Configuration valid: ✅")
        
        return True
    
    def _test_api_connections(self) -> bool:
        """Test all configured API connections"""
        test_text = "Test document about business and technology."
        working_providers = []
        
        for provider_name, provider_config in self.api_clients.items():
            try:
                print(f"[INFO] Testing {provider_name} API connection...")
                keywords = self._extract_keywords_with_provider(test_text, provider_name)
                if keywords:
                    working_providers.append(provider_name)
                    print(f"[OK] {provider_name} API working")
                else:
                    print(f"[WARN] {provider_name} API test returned no keywords")
            except Exception as e:
                print(f"[ERROR] {provider_name} API test failed: {e}")
        
        if not working_providers:
            print("[ERROR] No working API providers found")
            return False
        
        # Update primary provider if current one failed
        if self.primary_provider not in working_providers:
            self.primary_provider = working_providers[0]
            print(f"[INFO] Switched primary provider to: {self.primary_provider}")
        
        return True
    
    def create_enhanced_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        """Create enhanced prompt for keyword extraction"""
        
        # Determine domain based on context or text analysis
        domain_hints = []
        if context:
            doc_name = context.get('doc_name', '').lower()
            if any(term in doc_name for term in ['financial', 'finance', 'accounting']):
                domain_hints.append("financial")
            elif any(term in doc_name for term in ['legal', 'law', 'regulation']):
                domain_hints.append("legal")
            elif any(term in doc_name for term in ['technical', 'tech', 'engineering']):
                domain_hints.append("technical")
        
        domain_guidance = ""
        if domain_hints:
            domain_guidance = f"This appears to be {'/'.join(domain_hints)} content. "
        
        prompt = f"""Extract exactly {self.config.KEYWORDS_PER_CHUNK} relevant keywords from this text. {domain_guidance}Focus on:

1. Key entities (companies, organizations, people, places)
2. Important concepts and terminology
3. Specific products, services, or technologies
4. Regulatory or compliance terms
5. Quantifiable metrics or amounts

TEXT: {text}

Requirements:
- Return exactly {self.config.KEYWORDS_PER_CHUNK} keywords
- Use lowercase
- No duplicates
- Single words or short phrases (2-3 words max)
- Focus on the most important and unique terms

Return ONLY a JSON array:
["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]"""

        return prompt
    
    def _extract_keywords_with_provider(self, text: str, provider: str, 
                                      context: Dict[str, Any] = None) -> Optional[List[str]]:
        """Extract keywords using specific provider"""
        
        if not text or len(text.strip()) < self.config.MIN_TEXT_LENGTH:
            return []
        
        # Truncate text if too long
        if len(text) > self.config.MAX_TEXT_LENGTH:
            text = text[:self.config.MAX_TEXT_LENGTH] + "..."
        
        # Check cache
        if self.cache is not None:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.cache:
                return self.cache[text_hash]
        
        self.monitor.increment_api_calls()
        
        try:
            if provider == 'groq':
                return self._extract_with_groq(text, context)
            elif provider == 'openai':
                return self._extract_with_openai(text, context)
            elif provider == 'gemini':
                return self._extract_with_gemini(text, context)
            else:
                logger.error(f"Unknown provider: {provider}")
                return None
                
        except Exception as e:
            logger.error(f"Error with {provider}: {e}")
            return None
    
    def _extract_with_groq(self, text: str, context: Dict[str, Any] = None) -> Optional[List[str]]:
        """Extract keywords using Groq API"""
        import requests
        
        config = self.api_clients['groq']
        prompt = self.create_enhanced_prompt(text, context)
        
        payload = {
            "model": config['model'],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 100,
            "top_p": 0.9
        }
        
        response = requests.post(
            config['base_url'],
            headers=config['headers'],
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip()
            return self._parse_keywords_response(content, text)
        
        elif response.status_code == 429:
            # Rate limit - exponential backoff
            wait_time = min(self.config.RATE_LIMIT_DELAY * 2, 10)
            logger.warning(f"Rate limit hit, waiting {wait_time}s")
            time.sleep(wait_time)
            return self._extract_with_groq(text, context)
        
        else:
            logger.error(f"Groq API error: {response.status_code}")
            return None
    
    def _extract_with_openai(self, text: str, context: Dict[str, Any] = None) -> Optional[List[str]]:
        """Extract keywords using OpenAI API"""
        config = self.api_clients['openai']
        prompt = self.create_enhanced_prompt(text, context)
        
        response = config['client'].chat.completions.create(
            model=config['model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100
        )
        
        content = response.choices[0].message.content.strip()
        return self._parse_keywords_response(content, text)
    
    def _extract_with_gemini(self, text: str, context: Dict[str, Any] = None) -> Optional[List[str]]:
        """Extract keywords using Gemini API"""
        config = self.api_clients['gemini']
        prompt = self.create_enhanced_prompt(text, context)
        
        response = config['client'].generate_content(prompt)
        content = response.text.strip()
        return self._parse_keywords_response(content, text)
    
    def _parse_keywords_response(self, response: str, original_text: str) -> List[str]:
        """Enhanced parsing of keyword extraction response"""
        keywords = []
        
        try:
            # Try JSON parsing first
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                json_str = response[start:end]
                parsed_keywords = json.loads(json_str)
                
                # Clean and validate keywords
                for kw in parsed_keywords:
                    if isinstance(kw, str) and len(kw.strip()) > 1:
                        cleaned = kw.strip().lower()
                        # Remove quotes and special characters
                        cleaned = cleaned.strip('"\'').strip()
                        if cleaned and not cleaned.isdigit() and len(cleaned) <= 50:
                            keywords.append(cleaned)
                
                # Cache successful result
                if self.cache is not None and keywords:
                    text_hash = hashlib.md5(original_text.encode()).hexdigest()
                    self.cache[text_hash] = keywords
                
                return keywords[:self.config.KEYWORDS_PER_CHUNK]
        
        except json.JSONDecodeError:
            pass
        
        # Fallback: manual extraction
        return self._extract_keywords_manually(response)
    
    def _extract_keywords_manually(self, response_text: str) -> List[str]:
        """Enhanced manual keyword extraction from malformed responses"""
        import re
        
        keywords = []
        
        # Strategy 1: Find quoted terms
        quoted_words = re.findall(r'"([^"]*)"', response_text)
        for word in quoted_words:
            if len(word.strip()) > 1:
                keywords.append(word.strip().lower())
        
        # Strategy 2: Find listed items
        if not keywords:
            # Look for numbered or bulleted lists
            list_items = re.findall(r'(?:^\d+\.|\*|\-)\s*([^\n]+)', response_text, re.MULTILINE)
            for item in list_items:
                cleaned = re.sub(r'[^\w\s]', '', item).strip().lower()
                if cleaned and len(cleaned) > 1:
                    keywords.append(cleaned)
        
        # Strategy 3: Split by common delimiters
        if not keywords:
            # Clean response and split
            clean_text = re.sub(r'^.*?:', '', response_text)
            potential_keywords = re.split(r'[,\n\-\*\d+\.\s]{2,}', clean_text)
            
            for kw in potential_keywords:
                cleaned = re.sub(r'[^\w\s]', '', kw).strip().lower()
                if cleaned and len(cleaned) > 1 and not cleaned.isdigit():
                    keywords.append(cleaned)
        
        return keywords[:self.config.KEYWORDS_PER_CHUNK]
    
    def extract_keywords_from_text(self, text: str, context: Dict[str, Any] = None) -> List[str]:
        """Main keyword extraction with fallback providers"""
        
        # Try primary provider
        keywords = self._extract_keywords_with_provider(text, self.primary_provider, context)
        
        if keywords:
            return keywords
        
        # Try fallback providers
        for provider in self.api_clients.keys():
            if provider != self.primary_provider:
                logger.info(f"Trying fallback provider: {provider}")
                keywords = self._extract_keywords_with_provider(text, provider, context)
                if keywords:
                    logger.info(f"Successful extraction with {provider}")
                    return keywords
        
        # All providers failed
        logger.error("All API providers failed for text extraction")
        self.monitor.increment_errors()
        return []
    
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
    
    def process_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of chunks with enhanced error handling"""
        enhanced_chunks = []
        
        for chunk in chunks:
            try:
                text = chunk.get('text', '')
                
                if not text or len(text.strip()) < self.config.MIN_TEXT_LENGTH:
                    # Keep chunk but with empty keywords
                    enhanced_chunk = self._create_enhanced_chunk(chunk, [])
                    enhanced_chunks.append(enhanced_chunk)
                    continue
                
                # Create context for better keyword extraction
                context = {
                    'doc_name': chunk.get('doc_id', ''),
                    'page': chunk.get('page', 1),
                    'paragraph': chunk.get('paragraph_number', 1)
                }
                
                # Extract keywords with rate limiting
                with self._rate_limit_context():
                    keywords = self.extract_keywords_from_text(text, context)
                
                # Create enhanced chunk
                enhanced_chunk = self._create_enhanced_chunk(chunk, keywords)
                enhanced_chunks.append(enhanced_chunk)
                
                # Memory check
                if not self.monitor.check_memory():
                    logger.warning("Memory pressure detected, collecting garbage")
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing chunk: {e}")
                self.monitor.increment_errors()
                # Add chunk with empty keywords
                enhanced_chunk = self._create_enhanced_chunk(chunk, [])
                enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
    
    def _create_enhanced_chunk(self, original_chunk: Dict[str, Any], 
                             keywords: List[str]) -> Dict[str, Any]:
        """Create enhanced chunk with consistent field names"""
        return {
            "doc_id": original_chunk.get('doc_id'),
            "page": original_chunk.get('page'),
            "paragraph_number": original_chunk.get('paragraph_number', original_chunk.get('para_id')),
            "text": original_chunk.get('text'),
            "keywords": keywords,
            "keyword_count": len(keywords),
            "text_length": len(original_chunk.get('text', '')),
            "processed_at": datetime.now().isoformat(),
            "provider_used": self.primary_provider
        }
    
    def process_chunks_parallel(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process chunks in parallel with optimized batching"""
        logger.info(f"Processing {len(chunks)} chunks with {self.config.MAX_WORKERS} workers")
        
        all_enhanced_chunks = []
        batch_size = self.config.BATCH_SIZE
        
        # Create batches
        batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.MAX_WORKERS) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_chunks_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                
                try:
                    batch_results = future.result(timeout=300)
                    all_enhanced_chunks.extend(batch_results)
                    
                    # Update progress
                    self.monitor.log_progress(len(all_enhanced_chunks), len(chunks))
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    self.monitor.increment_errors()
                    # Add original chunks with empty keywords
                    for chunk in batches[batch_idx]:
                        enhanced_chunk = self._create_enhanced_chunk(chunk, [])
                        all_enhanced_chunks.append(enhanced_chunk)
        
        logger.info(f"Parallel processing completed: {len(all_enhanced_chunks)} chunks processed")
        return all_enhanced_chunks
    
    def save_checkpoint(self, enhanced_chunks: List[Dict[str, Any]], 
                       checkpoint_name: str = "checkpoint"):
        """Save processing checkpoint"""
        if not self.config.SAVE_INTERMEDIATE:
            return
        
        try:
            checkpoint_file = self.output_dir / f"{checkpoint_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            checkpoint_data = {
                "checkpoint_created_at": datetime.now().isoformat(),
                "chunks_processed": len(enhanced_chunks),
                "primary_provider": self.primary_provider,
                "chunks": enhanced_chunks
            }
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Checkpoint saved: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def run_extraction(self) -> bool:
        """Main extraction process with enhanced robustness"""
        
        # Load consolidated chunks
        consolidated_file = self.input_dir / "all_chunks.json"
        
        try:
            logger.info(f"Loading chunks from: {consolidated_file}")
            
            with open(consolidated_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            total_chunks = len(chunks)
            
            if total_chunks == 0:
                print("[ERROR] No chunks found in input file")
                return False
            
            print(f"[INFO] Loaded {total_chunks} chunks for processing")
            
            # Estimate processing time
            estimated_time = (total_chunks * self.config.RATE_LIMIT_DELAY) / self.config.MAX_WORKERS / 60
            print(f"[INFO] Estimated processing time: {estimated_time:.1f} minutes")
            print(f"[INFO] Primary provider: {self.primary_provider}")
            print(f"[INFO] Batch size: {self.config.BATCH_SIZE}")
            print(f"[INFO] Workers: {self.config.MAX_WORKERS}")
            
            # Confirmation (unless auto-confirm is enabled)
            if not self.config.AUTO_CONFIRM:
                response = input(f"\n[INPUT] Start keyword extraction? (yes/no): ").strip().lower()
                print(f"DEBUG: You entered: '{response}' (length: {len(response)})")
                
                valid_responses = ['yes', 'y', 'ye']
                if response not in valid_responses:
                    print(f"[INFO] Processing cancelled (received: '{response}')")
                    print(f"[INFO] Valid responses: {valid_responses}")
                    return False
                
                print("[OK] Starting keyword extraction...")
            else:
                print("[INFO] Auto-confirm enabled, starting extraction...")
            
            # Process chunks
            start_time = time.time()
            
            if self.config.MAX_WORKERS > 1:
                enhanced_chunks = self.process_chunks_parallel(chunks)
            else:
                enhanced_chunks = self.process_chunks_batch(chunks)
            
            end_time = time.time()
            
            # Save checkpoint
            self.save_checkpoint(enhanced_chunks, "final")
            
            # Update data structure with enhanced metadata
            enhanced_data = data.copy()
            enhanced_data['chunks'] = enhanced_chunks
            enhanced_data['metadata'].update({
                'keywords_extracted': True,
                'extraction_completed_at': datetime.now().isoformat(),
                'primary_provider': self.primary_provider,
                'available_providers': list(self.api_clients.keys()),
                'processing_config': {
                    'batch_size': self.config.BATCH_SIZE,
                    'max_workers': self.config.MAX_WORKERS,
                    'keywords_per_chunk': self.config.KEYWORDS_PER_CHUNK,
                    'rate_limit_delay': self.config.RATE_LIMIT_DELAY
                },
                'performance_metrics': {
                    'total_api_calls': self.monitor.api_call_count,
                    'error_count': self.monitor.error_count,
                    'processing_time_seconds': end_time - start_time,
                    'chunks_per_second': total_chunks / (end_time - start_time) if end_time > start_time else 0
                }
            })
            
            # Save main results
            output_file = self.output_dir / "all_chunks_with_keywords.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
            
            # Calculate comprehensive statistics
            chunks_with_keywords = sum(1 for chunk in enhanced_chunks if chunk.get('keywords'))
            processing_time = (end_time - start_time) / 60
            avg_keywords_per_chunk = sum(len(chunk.get('keywords', [])) for chunk in enhanced_chunks) / len(enhanced_chunks)
            
            print(f"\n[SUCCESS] KEYWORD EXTRACTION COMPLETED!")
            print(f"   Processing time: {processing_time:.1f} minutes")
            print(f"   Total chunks: {total_chunks:,}")
            print(f"   Chunks with keywords: {chunks_with_keywords:,}")
            print(f"   Success rate: {chunks_with_keywords/total_chunks*100:.1f}%")
            print(f"   Average keywords per chunk: {avg_keywords_per_chunk:.1f}")
            print(f"   Total API calls: {self.monitor.api_call_count:,}")
            print(f"   Error count: {self.monitor.error_count}")
            print(f"   Processing rate: {total_chunks/((end_time - start_time)/60):.1f} chunks/minute")
            print(f"   Output saved to: {output_file}")
            
            # Show sample results
            self._display_sample_results(enhanced_chunks)
            
            # Generate and save comprehensive statistics
            self._save_extraction_statistics(enhanced_chunks, enhanced_data['metadata'])
            
            return True
            
        except Exception as e:
            logger.error(f"Critical error during extraction: {e}")
            print(f"[ERROR] Extraction failed: {e}")
            return False
    
    def _display_sample_results(self, enhanced_chunks: List[Dict[str, Any]]):
        """Display sample extraction results"""
        print(f"\n[SAMPLE] Keyword Extraction Examples:")
        
        sample_count = 0
        for i, chunk in enumerate(enhanced_chunks):
            keywords = chunk.get('keywords', [])
            if keywords and sample_count < 3:
                text_preview = chunk.get('text', '')[:100] + "..." if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
                print(f"\n   Example {sample_count + 1}:")
                print(f"   Document: {chunk.get('doc_id', 'Unknown')}")
                print(f"   Page: {chunk.get('page', 'N/A')}")
                print(f"   Keywords: {keywords}")
                print(f"   Text: {text_preview}")
                sample_count += 1
        
        if sample_count == 0:
            print("   No keywords found in sample chunks")
    
    def _save_extraction_statistics(self, enhanced_chunks: List[Dict[str, Any]], 
                                  metadata: Dict[str, Any]):
        """Save comprehensive extraction statistics"""
        
        # Calculate detailed statistics
        total_chunks = len(enhanced_chunks)
        chunks_with_keywords = sum(1 for chunk in enhanced_chunks if chunk.get('keywords'))
        
        # Keyword frequency analysis
        keyword_counts = {}
        for chunk in enhanced_chunks:
            for keyword in chunk.get('keywords', []):
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:25]
        
        # Provider performance
        provider_usage = {}
        for chunk in enhanced_chunks:
            provider = chunk.get('provider_used', 'unknown')
            provider_usage[provider] = provider_usage.get(provider, 0) + 1
        
        # Text length analysis
        text_lengths = [len(chunk.get('text', '')) for chunk in enhanced_chunks]
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        # Keyword count distribution
        keyword_counts_dist = {}
        for chunk in enhanced_chunks:
            kw_count = len(chunk.get('keywords', []))
            keyword_counts_dist[kw_count] = keyword_counts_dist.get(kw_count, 0) + 1
        
        # Compile comprehensive statistics
        stats = {
            "extraction_summary": {
                "total_chunks": total_chunks,
                "chunks_with_keywords": chunks_with_keywords,
                "success_rate": round(chunks_with_keywords/total_chunks*100, 2) if total_chunks > 0 else 0,
                "processing_time_minutes": metadata.get('performance_metrics', {}).get('processing_time_seconds', 0) / 60,
                "chunks_per_minute": metadata.get('performance_metrics', {}).get('chunks_per_second', 0) * 60,
                "total_api_calls": metadata.get('performance_metrics', {}).get('total_api_calls', 0),
                "error_count": metadata.get('performance_metrics', {}).get('error_count', 0),
                "primary_provider": metadata.get('primary_provider', 'unknown')
            },
            "keyword_analysis": {
                "total_unique_keywords": len(keyword_counts),
                "average_keywords_per_chunk": sum(len(chunk.get('keywords', [])) for chunk in enhanced_chunks) / total_chunks if total_chunks > 0 else 0,
                "top_keywords": [{"keyword": kw, "count": count} for kw, count in top_keywords],
                "keyword_count_distribution": keyword_counts_dist
            },
            "text_analysis": {
                "average_text_length": round(avg_text_length, 2),
                "min_text_length": min(text_lengths) if text_lengths else 0,
                "max_text_length": max(text_lengths) if text_lengths else 0,
                "total_text_characters": sum(text_lengths)
            },
            "provider_performance": {
                "provider_usage": provider_usage,
                "available_providers": list(self.api_clients.keys()),
                "primary_provider": self.primary_provider
            },
            "configuration": {
                "batch_size": self.config.BATCH_SIZE,
                "max_workers": self.config.MAX_WORKERS,
                "keywords_per_chunk": self.config.KEYWORDS_PER_CHUNK,
                "rate_limit_delay": self.config.RATE_LIMIT_DELAY,
                "memory_limit_gb": self.config.MEMORY_LIMIT_GB
            },
            "processed_at": datetime.now().isoformat(),
            "extraction_version": "2.0_production"
        }
        
        # Save statistics
        stats_file = self.output_dir / "extraction_summary.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Save latest statistics (for easy access)
        latest_stats_file = self.output_dir / "extraction_summary_latest.json"
        with open(latest_stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"[INFO] Statistics saved to: {stats_file}")
        
        # Display key statistics
        print(f"\n[STATISTICS] Key Metrics:")
        print(f"   Unique keywords extracted: {len(keyword_counts):,}")
        print(f"   Average keywords per chunk: {stats['keyword_analysis']['average_keywords_per_chunk']:.1f}")
        print(f"   Most common keyword: {top_keywords[0][0]} ({top_keywords[0][1]} times)" if top_keywords else "   No keywords found")
        print(f"   Average text length: {avg_text_length:.0f} characters")
    
    def get_top_keywords(self, chunks: List[Dict[str, Any]], top_n: int = 15) -> List[Dict[str, Any]]:
        """Get top keywords by frequency with enhanced analysis"""
        
        keyword_counts = {}
        
        for chunk in chunks:
            keywords = chunk.get('keywords', [])
            for keyword in keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Sort by frequency
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        return [{"keyword": kw, "count": count, "percentage": round(count/len(chunks)*100, 2)} 
                for kw, count in top_keywords]

# Batch processing function for large datasets
def batch_process_keywords(input_file: str, output_dir: str, 
                         batch_size: int = 1000) -> bool:
    """Process very large datasets in batches to manage memory"""
    
    logger.info(f"Starting batch processing: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        total_chunks = len(chunks)
        
        if total_chunks <= batch_size:
            # Process normally if dataset is small enough
            extractor = ProductionKeywordExtractor()
            return extractor.run_extraction()
        
        # Process in batches
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_enhanced_chunks = []
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            logger.info(f"Processing batch {batch_num}: chunks {i+1} to {min(i+batch_size, total_chunks)}")
            
            # Create temporary extractor for this batch
            extractor = ProductionKeywordExtractor()
            enhanced_batch = extractor.process_chunks_batch(batch_chunks)
            
            all_enhanced_chunks.extend(enhanced_batch)
            
            # Save intermediate results
            batch_file = output_path / f"batch_{batch_num}_keywords.json"
            with open(batch_file, 'w', encoding='utf-8') as f:
                json.dump({"chunks": enhanced_batch}, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Batch {batch_num} completed and saved")
            
            # Clean up memory
            del extractor
            gc.collect()
        
        # Combine all results
        final_data = data.copy()
        final_data['chunks'] = all_enhanced_chunks
        final_data['metadata']['keywords_extracted'] = True
        final_data['metadata']['batch_processed'] = True
        final_data['metadata']['extraction_completed_at'] = datetime.now().isoformat()
        
        # Save final combined results
        final_file = output_path / "all_chunks_with_keywords.json"
        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Batch processing completed: {final_file}")
        return True
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return False

# CLI interface
def main():
    """Enhanced main function with comprehensive CLI"""
    
    print("[INFO] PRODUCTION KEYWORD EXTRACTION")
    print("=" * 60)
    print("[FEATURES] Multi-provider support (Groq/OpenAI/Gemini)")
    print("[FEATURES] Parallel processing, caching, error recovery")
    print("=" * 60)
    
    # Initialize extractor
    try:
        extractor = ProductionKeywordExtractor()
    except Exception as e:
        print(f"[ERROR] Failed to initialize extractor: {e}")
        return False
    
    # Check setup
    if not extractor.check_setup():
        print("[ERROR] Setup validation failed")
        return False
    
    # Display configuration
    print(f"\n[CONFIG] Processing Configuration:")
    print(f"   Primary provider: {extractor.primary_provider}")
    print(f"   Available providers: {list(extractor.api_clients.keys())}")
    print(f"   Batch size: {extractor.config.BATCH_SIZE}")
    print(f"   Max workers: {extractor.config.MAX_WORKERS}")
    print(f"   Keywords per chunk: {extractor.config.KEYWORDS_PER_CHUNK}")
    print(f"   Rate limit delay: {extractor.config.RATE_LIMIT_DELAY}s")
    print(f"   Auto-confirm: {extractor.config.AUTO_CONFIRM}")
    
    # Run extraction
    success = extractor.run_extraction()
    
    if success:
        print("\n[SUCCESS] Keyword extraction completed successfully!")
        print("[INFO] Your chunks now have enhanced keywords for better RAG retrieval")
        print("[INFO] Check the output directory for detailed statistics and results")
    else:
        print("\n[ERROR] Keyword extraction failed")
        print("[INFO] Check the logs for detailed error information")
    
    return success

# Production entry point
if __name__ == "__main__":
    import sys
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch" and len(sys.argv) >= 4:
            # Batch processing mode
            input_file = sys.argv[2]
            output_dir = sys.argv[3]
            batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
            
            success = batch_process_keywords(input_file, output_dir, batch_size)
            sys.exit(0 if success else 1)
        
        elif sys.argv[1] == "--config":
            # Show configuration
            config = ProductionConfig()
            print("Current Configuration:")
            print(json.dumps(config.__dict__, indent=2))
            sys.exit(0)
        
        elif sys.argv[1] == "--help":
            print("Production Keyword Extraction Tool")
            print("Usage:")
            print("  python keywords.py                    # Normal processing")
            print("  python keywords.py --batch INPUT OUTPUT [SIZE]  # Batch processing")
            print("  python keywords.py --config           # Show configuration")
            print("  python keywords.py --help             # Show this help")
            sys.exit(0)
    
    # Normal processing
    success = main()
    sys.exit(0 if success else 1)