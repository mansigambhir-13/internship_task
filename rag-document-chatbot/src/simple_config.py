"""
Simple Configuration Module for RAG Pipeline
Creates a basic configuration when the full config module is not available
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimpleSettings:
    """Simple settings class with basic configuration"""
    
    # Directories
    PDF_DIRECTORY: str = "./documents"
    PROCESSED_DOCUMENTS_DIR: str = "./data/processed_documents"
    CHUNKED_DOCUMENTS_DIR: str = "./data/chunked_documents"
    KEYWORDS_ENHANCED_DIR: str = "./data/keywords_enhanced"
    EMBEDDING_STORAGE_DIR: str = "./data/embeddings"
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    BATCH_SIZE: int = 50
    
    # Qdrant Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    COLLECTION_NAME: str = "document_chunks"
    
    # Processing Options
    USE_OCR: bool = True
    EXTRACT_KEYWORDS: bool = True
    USE_ADVANCED_NLP: bool = True
    DEDUPLICATE: bool = True
    DEBUG: bool = False
    
    # Performance Settings
    MAX_WORKERS: int = 2
    MAX_RETRIES: int = 3
    RATE_LIMIT_DELAY: float = 0.5
    MEMORY_LIMIT_GB: float = 8.0
    CHECKPOINT_INTERVAL: int = 100
    ENABLE_MONITORING: bool = True
    
    def __post_init__(self):
        """Load settings from environment variables"""
        self.load_from_env()
        self.create_directories()
    
    def load_from_env(self):
        """Load configuration from environment variables"""
        # Load environment variables
        env_mappings = {
            'PDF_DIRECTORY': 'PDF_DIRECTORY',
            'OPENAI_API_KEY': 'OPENAI_API_KEY',
            'GEMINI_API_KEY': 'GEMINI_API_KEY',
            'EMBEDDING_MODEL': 'EMBEDDING_MODEL',
            'BATCH_SIZE': 'BATCH_SIZE',
            'QDRANT_URL': 'QDRANT_URL',
            'QDRANT_HOST': 'QDRANT_HOST',
            'QDRANT_PORT': 'QDRANT_PORT',
            'QDRANT_API_KEY': 'QDRANT_API_KEY',
            'COLLECTION_NAME': 'COLLECTION_NAME',
            'USE_OCR': 'USE_OCR',
            'EXTRACT_KEYWORDS': 'EXTRACT_KEYWORDS',
            'USE_ADVANCED_NLP': 'USE_ADVANCED_NLP',
            'DEDUPLICATE': 'DEDUPLICATE',
            'DEBUG': 'DEBUG',
            'MAX_WORKERS': 'MAX_WORKERS',
            'MAX_RETRIES': 'MAX_RETRIES',
            'RATE_LIMIT_DELAY': 'RATE_LIMIT_DELAY',
            'MEMORY_LIMIT_GB': 'MEMORY_LIMIT_GB',
        }
        
        for attr_name, env_name in env_mappings.items():
            env_value = os.getenv(env_name)
            if env_value is not None:
                # Convert to appropriate type
                if hasattr(self, attr_name):
                    current_value = getattr(self, attr_name)
                    if isinstance(current_value, bool):
                        setattr(self, attr_name, env_value.lower() in ('true', '1', 'yes', 'on'))
                    elif isinstance(current_value, int):
                        try:
                            setattr(self, attr_name, int(env_value))
                        except ValueError:
                            pass
                    elif isinstance(current_value, float):
                        try:
                            setattr(self, attr_name, float(env_value))
                        except ValueError:
                            pass
                    else:
                        setattr(self, attr_name, env_value)
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            self.PDF_DIRECTORY,
            self.PROCESSED_DOCUMENTS_DIR,
            self.CHUNKED_DOCUMENTS_DIR,
            self.KEYWORDS_ENHANCED_DIR,
            self.EMBEDDING_STORAGE_DIR,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> bool:
        """Basic configuration validation"""
        # Check if at least one API key is set
        if not self.OPENAI_API_KEY and not self.GEMINI_API_KEY:
            return False
        
        # Check if PDF directory exists or can be created
        try:
            Path(self.PDF_DIRECTORY).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False
    
    def get_embedding_service(self) -> str:
        """Get the configured embedding service"""
        if self.OPENAI_API_KEY and self.EMBEDDING_MODEL.startswith("text-embedding"):
            return "OpenAI"
        elif self.GEMINI_API_KEY:
            return "Gemini"
        else:
            return "None"

# Create a default settings instance
settings = SimpleSettings()

# Also create a settings.py compatibility layer
class SettingsModule:
    """Compatibility layer that mimics a settings module"""
    
    def __init__(self, settings_obj):
        self._settings = settings_obj
        
        # Copy all attributes from settings object
        for attr_name in dir(settings_obj):
            if not attr_name.startswith('_'):
                setattr(self, attr_name, getattr(settings_obj, attr_name))
    
    def validate_config(self):
        return self._settings.validate_config()
    
    def create_directories(self):
        return self._settings.create_directories()

# For backward compatibility
if 'config' not in globals():
    import sys
    import types
    
    # Create a fake config module
    config_module = types.ModuleType('config')
    settings_module = types.ModuleType('config.settings')
    
    # Add the settings to the module
    settings_module.settings = SettingsModule(settings)
    
    # Add to sys.modules so imports work
    sys.modules['config'] = config_module
    sys.modules['config.settings'] = settings_module
    config_module.settings = settings_module