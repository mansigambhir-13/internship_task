"""
Clean Streamlit App for RAG Pipeline Coordinator
Eliminates ScriptRunContext warnings by avoiding early imports and initialization
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Configure logging first to avoid issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_python_path():
    """Setup Python path without importing streamlit"""
    try:
        if '__file__' in globals():
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            src_dir = current_file.parent
        else:
            project_root = Path.cwd()
            src_dir = project_root / "src"
    except:
        project_root = Path.cwd()
        src_dir = project_root / "src"

    # Add paths
    paths_to_add = [
        str(project_root),
        str(src_dir),
        str(project_root / "src"),
        str(Path.cwd()),
        str(Path.cwd() / "src")
    ]

    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    return project_root, src_dir

# Setup paths
project_root, src_dir = setup_python_path()

def main():
    """Main function - imports streamlit only when needed"""
    
    # Import streamlit only when we're actually running the app
    try:
        import streamlit as st
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as e:
        print(f"Error importing required packages: {e}")
        print("Please install required packages: pip install streamlit pandas plotly")
        return

    # Configure Streamlit page
    st.set_page_config(
        page_title="RAG Pipeline Controller",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'pipeline_status' not in st.session_state:
        st.session_state.pipeline_status = {}
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = {}
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        
        .status-card {
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }
        
        .success-card {
            border-left-color: #28a745;
            background-color: #d4edda;
        }
        
        .error-card {
            border-left-color: #dc3545;
            background-color: #f8d7da;
        }
        
        .warning-card {
            border-left-color: #ffc107;
            background-color: #fff3cd;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 RAG Pipeline Controller</h1>
        <p>Document Research & Theme Identification System</p>
        <p><em>Wasserstoff AI Internship Project</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Dashboard", "⚙️ Configuration", "🚀 Pipeline Execution", "🔍 Document Search", "📊 Analytics", "📝 Logs"]
    )

    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "⚙️ Configuration":
        show_configuration()
    elif page == "🚀 Pipeline Execution":
        show_pipeline_execution()
    elif page == "🔍 Document Search":
        show_document_search()
    elif page == "📊 Analytics":
        show_analytics()
    elif page == "📝 Logs":
        show_logs()

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        ('fitz', 'PyMuPDF'),
        ('pytesseract', 'pytesseract'),
        ('PIL', 'Pillow'),
        ('qdrant_client', 'qdrant-client'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn')
    ]
    
    optional_packages = [
        ('openai', 'openai'),
        ('google.generativeai', 'google-generativeai'),
        ('spacy', 'spacy')
    ]
    
    status = {'required': {}, 'optional': {}}
    
    for package, pip_name in required_packages:
        try:
            __import__(package)
            status['required'][pip_name] = True
        except ImportError:
            status['required'][pip_name] = False
    
    for package, pip_name in optional_packages:
        try:
            __import__(package)
            status['optional'][pip_name] = True
        except ImportError:
            status['optional'][pip_name] = False
    
    return status

def get_or_create_settings():
    """Get settings with fallback to simple configuration"""
    try:
        # Try to import from the actual config module
        from config.settings import settings
        return settings, "Full configuration loaded"
    except ImportError:
        try:
            # Try to import our simple config
            from simple_config import settings
            return settings, "Simple configuration loaded"
        except ImportError:
            # Create inline simple config as last resort
            return create_inline_config(), "Inline configuration created"

def create_inline_config():
    """Create a minimal inline configuration"""
    class InlineConfig:
        PDF_DIRECTORY = "./documents"
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        EMBEDDING_MODEL = "text-embedding-3-small"
        QDRANT_URL = "http://localhost:6333"
        BATCH_SIZE = 50
        USE_OCR = True
        EXTRACT_KEYWORDS = True
        DEBUG = False
        
        def validate_config(self):
            return bool(self.OPENAI_API_KEY or self.GEMINI_API_KEY)
        
        def get_embedding_service(self):
            if self.OPENAI_API_KEY:
                return "OpenAI"
            elif self.GEMINI_API_KEY:
                return "Gemini"
            else:
                return "None"
    
    return InlineConfig()

def validate_configuration():
    """Validate configuration with enhanced error handling"""
    try:
        settings, config_source = get_or_create_settings()
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {'config_source': config_source}
        }
        
        # Check PDF directory
        pdf_dir = Path(settings.PDF_DIRECTORY)
        try:
            pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf_files = list(pdf_dir.glob("*.pdf"))
            validation_result['info']['pdf_count'] = len(pdf_files)
            validation_result['info']['pdf_directory'] = str(pdf_dir)
            
            if len(pdf_files) == 0:
                validation_result['warnings'].append(f"No PDF files found in {pdf_dir}")
        except Exception as e:
            validation_result['errors'].append(f"PDF directory error: {str(e)}")
        
        # Check embedding service
        embedding_service = "None"
        if hasattr(settings, 'get_embedding_service'):
            embedding_service = settings.get_embedding_service()
        elif hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
            embedding_service = "OpenAI"
        elif hasattr(settings, 'GEMINI_API_KEY') and settings.GEMINI_API_KEY:
            embedding_service = "Gemini"
        
        validation_result['info']['embedding_service'] = embedding_service
        
        if embedding_service == "None":
            validation_result['warnings'].append("No embedding service configured. Set OPENAI_API_KEY or GEMINI_API_KEY environment variable.")
        
        # Check other settings
        validation_result['info']['qdrant_url'] = getattr(settings, 'QDRANT_URL', 'http://localhost:6333')
        validation_result['info']['batch_size'] = getattr(settings, 'BATCH_SIZE', 50)
        validation_result['info']['use_ocr'] = getattr(settings, 'USE_OCR', True)
        validation_result['info']['extract_keywords'] = getattr(settings, 'EXTRACT_KEYWORDS', True)
        
        # Overall validation
        if hasattr(settings, 'validate_config'):
            if not settings.validate_config():
                validation_result['valid'] = False
                validation_result['errors'].append("Configuration validation failed")
        
        return validation_result
        
    except Exception as e:
        return {
            'valid': False,
            'errors': [f"Configuration error: {str(e)}"],
            'warnings': [],
            'info': {'config_source': 'Error loading configuration'}
        }

def safe_import_embeddings():
    """Safely import embeddings module"""
    try:
        import embeddings
        if hasattr(embeddings, 'ProductionEmbeddingProcessor'):
            return embeddings
        else:
            return None
    except ImportError:
        return None

def show_dashboard():
    """Dashboard page"""
    import streamlit as st
    
    st.header("📊 System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Dependencies Status")
        deps = check_dependencies()
        
        st.write("**Required Packages:**")
        for package, status in deps['required'].items():
            status_icon = "✅" if status else "❌"
            st.write(f"{status_icon} {package}")
        
        st.write("**Optional Packages:**")
        for package, status in deps['optional'].items():
            status_icon = "✅" if status else "⚠️"
            st.write(f"{status_icon} {package}")
    
    with col2:
        st.subheader("⚙️ Configuration Status")
        config = validate_configuration()
        
        if config['valid']:
            st.markdown('<div class="status-card success-card"><b>✅ Configuration Valid</b></div>', 
                       unsafe_allow_html=True)
            
            info = config.get('info', {})
            if info:
                st.write(f"📁 PDF Directory: `{info.get('pdf_directory', 'N/A')}`")
                st.write(f"📄 PDF Files: {info.get('pdf_count', 0)}")
                st.write(f"🤖 Embedding Service: {info.get('embedding_service', 'N/A')}")
                st.write(f"🗄️ Qdrant URL: `{info.get('qdrant_url', 'N/A')}`")
        else:
            st.markdown('<div class="status-card error-card"><b>❌ Configuration Issues</b></div>', 
                       unsafe_allow_html=True)
            
            for error in config['errors']:
                st.error(error)
            
            for warning in config['warnings']:
                st.warning(warning)

def show_configuration():
    """Configuration page"""
    import streamlit as st
    
    st.header("⚙️ System Configuration")
    
    config = validate_configuration()
    
    # Show configuration source
    st.info(f"📋 {config['info'].get('config_source', 'Unknown configuration source')}")
    
    if config['valid'] or len(config['errors']) == 0:
        st.success("✅ Configuration is functional!")
        
        # Show configuration details
        info = config.get('info', {})
        if info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📁 Directories & Files")
                st.write(f"**PDF Directory:** `{info.get('pdf_directory', 'N/A')}`")
                st.write(f"**PDF Files Found:** {info.get('pdf_count', 0)}")
                
                st.subheader("🔧 Processing Options")
                st.write(f"**Use OCR:** {info.get('use_ocr', 'N/A')}")
                st.write(f"**Extract Keywords:** {info.get('extract_keywords', 'N/A')}")
            
            with col2:
                st.subheader("🤖 AI Services")
                embedding_service = info.get('embedding_service', 'None')
                if embedding_service == "None":
                    st.warning(f"**Embedding Service:** ⚠️ {embedding_service}")
                else:
                    st.write(f"**Embedding Service:** ✅ {embedding_service}")
                
                st.subheader("🗄️ Vector Database")
                st.write(f"**Qdrant URL:** `{info.get('qdrant_url', 'N/A')}`")
                st.write(f"**Batch Size:** {info.get('batch_size', 'N/A')}")
        
        # Show warnings if any
        for warning in config.get('warnings', []):
            st.warning(f"⚠️ {warning}")
    
    else:
        st.error("❌ Configuration has issues")
        
        for error in config['errors']:
            st.error(f"❌ {error}")
    
    # Configuration help section
    st.subheader("💡 Configuration Help")
    
    with st.expander("🔧 How to Set Up Configuration"):
        st.write("""
        **Option 1: Environment Variables (Recommended)**
        Set these environment variables in your system:
        ```bash
        # Required: At least one API key
        OPENAI_API_KEY=your_openai_key_here
        # OR
        GEMINI_API_KEY=your_gemini_key_here
        
        # Optional: Customize directories and settings
        PDF_DIRECTORY=./documents
        QDRANT_URL=http://localhost:6333
        BATCH_SIZE=50
        ```
        
        **Option 2: Create .env file**
        Create a `.env` file in your project root:
        """)
        
        st.code("""
# .env file content
PDF_DIRECTORY=./documents
USE_OCR=true
EXTRACT_KEYWORDS=true

# Choose one embedding service
OPENAI_API_KEY=your_openai_key_here
# OR
GEMINI_API_KEY=your_gemini_key_here

EMBEDDING_MODEL=text-embedding-3-small
BATCH_SIZE=50
QDRANT_URL=http://localhost:6333
COLLECTION_NAME=document_chunks
DEBUG=false
        """)
        
        st.write("""
        **Option 3: Create config/settings.py**
        Create a proper configuration module (for advanced users).
        """)
    
    with st.expander("🚀 Quick Setup for Testing"):
        st.write("""
        **Minimal setup to get started:**
        
        1. **Set an API key** (choose one):
           - OpenAI: `set OPENAI_API_KEY=your_key` (Windows) or `export OPENAI_API_KEY=your_key` (Linux/Mac)
           - Gemini: `set GEMINI_API_KEY=your_key` (Windows) or `export GEMINI_API_KEY=your_key` (Linux/Mac)
        
        2. **Create documents folder:**
           - Create a `documents` folder in your project directory
           - Add some PDF files to test with
        
        3. **Start Qdrant (optional for full functionality):**
           - `docker run -p 6333:6333 qdrant/qdrant` (if you have Docker)
           - Or install Qdrant locally
        
        4. **Restart the Streamlit app** to reload configuration
        """)
        
        if st.button("🔄 Reload Configuration"):
            st.rerun()

def show_pipeline_execution():
    """Pipeline execution page"""
    import streamlit as st
    
    st.header("🚀 Pipeline Execution")
    
    # Check configuration
    config = validate_configuration()
    if not config['valid']:
        st.error("Configuration is not valid. Please fix configuration issues first.")
        return
    
    # Check if embeddings module is available
    embeddings_module = safe_import_embeddings()
    if not embeddings_module:
        st.error("❌ Embeddings module not found. Please ensure embeddings.py is in the correct location.")
        st.info("💡 Try running the import diagnostic script to identify the issue.")
        return
    
    st.success("✅ All dependencies are ready!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Complete Pipeline")
        st.write("Run all steps in sequence: OCR → Chunking → Embeddings")
        
        if st.button("🚀 Run Complete Pipeline", type="primary"):
            st.info("🚧 Complete pipeline execution would be implemented here")
            st.write("This would:")
            st.write("1. ✅ Run OCR and Document Parsing")
            st.write("2. ✅ Run Document Chunking")
            st.write("3. ✅ Create Embeddings and Store in Vector DB")
    
    with col2:
        st.subheader("⚙️ Individual Steps")
        st.write("Run pipeline steps individually for testing")
        
        if st.button("📄 Run OCR Only"):
            st.info("🚧 OCR step would be implemented here")
        
        if st.button("🔪 Run Chunking Only"):
            st.info("🚧 Chunking step would be implemented here")
        
        if st.button("🧮 Run Embeddings Only"):
            st.info("🚧 Embeddings step would be implemented here")

def show_document_search():
    """Document search page"""
    import streamlit as st
    
    st.header("🔍 Document Search")
    
    # Check if embeddings module is available
    embeddings_module = safe_import_embeddings()
    if not embeddings_module:
        st.warning("⚠️ Embeddings module not found. Please run the pipeline first to enable search.")
        return
    
    st.subheader("Search Your Documents")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter your search query:", placeholder="e.g., regulatory compliance, financial penalties...")
    
    with col2:
        limit = st.number_input("Results limit:", min_value=1, max_value=20, value=5)
    
    if st.button("🔍 Search") and query:
        st.info(f"🚧 Search functionality would be implemented here for: '{query}'")
        st.write("This would:")
        st.write("1. ✅ Create query embedding")
        st.write("2. ✅ Search vector database")
        st.write("3. ✅ Return ranked results")
        
        # Add to search history
        st.session_state.search_history.append({
            'query': query,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results_count': 0  # Placeholder
        })
    
    # Show search history
    if st.session_state.search_history:
        st.subheader("🕐 Search History")
        
        import pandas as pd
        history_df = pd.DataFrame(st.session_state.search_history)
        st.dataframe(history_df, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.search_history = []
            st.rerun()

def show_analytics():
    """Analytics page"""
    import streamlit as st
    
    st.header("📊 Pipeline Analytics")
    
    # Check if we have pipeline results
    if not st.session_state.pipeline_results:
        st.info("Run the pipeline to see analytics here.")
        return
    
    st.write("Analytics would be displayed here based on pipeline results.")

def show_logs():
    """Logs page"""
    import streamlit as st
    
    st.header("📝 System Logs")
    
    log_file = project_root / "pipeline.log"
    
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                log_content = f.read()
            
            st.subheader("📄 Pipeline Logs")
            st.text_area("Log Content", log_content, height=400)
            
            # Download logs
            st.download_button(
                label="💾 Download Logs",
                data=log_content,
                file_name=f"pipeline_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        except Exception as e:
            st.error(f"Error reading log file: {e}")
    else:
        st.info("No log file found. Logs will appear here after running the pipeline.")

if __name__ == "__main__":
    # Only run if executed directly by Streamlit
    main()