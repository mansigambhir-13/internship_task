import streamlit as st
import os
import sys
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import asyncio

# Enhanced page configuration
st.set_page_config(
    page_title="RAG Pipeline Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/rag-pipeline',
        'Report a bug': "https://github.com/your-repo/rag-pipeline/issues",
        'About': "# RAG Pipeline Dashboard\nA comprehensive document processing and search system."
    }
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Enhanced CSS with modern design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styles */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 2rem 0;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e74c3c;
    }
    
    /* Card styles */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    .status-card {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    
    .success-status {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
        color: #155724;
    }
    
    .warning-status {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left-color: #ffc107;
        color: #856404;
    }
    
    .error-status {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left-color: #dc3545;
        color: #721c24;
    }
    
    .info-status {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left-color: #17a2b8;
        color: #0c5460;
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Progress bar styles */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Sidebar styles */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Metric styles */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
    
    /* Alert styles */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
    
    /* Expander styles */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 10px;
        border: 1px solid #e9ecef;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Animation classes */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Loading spinner */
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
@st.cache_resource
def load_env_file():
    """Load environment variables from .env file with better error handling"""
    try:
        from dotenv import load_dotenv
        if load_dotenv():
            return True
    except ImportError:
        pass
    
    env_files = [Path('.env'), PROJECT_ROOT / '.env']
    for env_file in env_files:
        if env_file.exists():
            try:
                with open(env_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ[key.strip()] = value.strip().strip('"').strip("'")
                return True
            except Exception as e:
                st.error(f"Error loading env file {env_file}: {e}")
                continue
    return False

# Load environment
load_env_file()

# Enhanced imports with error handling
@st.cache_resource
def import_rag_components():
    """Import RAG components with proper error handling"""
    try:
        from main import RAGPipelineCoordinator
        from src.search import SemanticSearchEngine, SearchConfig
        from src.theme_identifier import ConciseThemeIdentifier
        return RAGPipelineCoordinator, SemanticSearchEngine, SearchConfig, ConciseThemeIdentifier
    except ImportError as e:
        st.error(f"""
        **Import Error:** {e}
        
        Please ensure all required modules are installed and accessible:
        - main.py (RAGPipelineCoordinator)
        - src/search.py (SemanticSearchEngine, SearchConfig)
        - src/theme_identifier.py (ConciseThemeIdentifier)
        """)
        st.stop()

RAGPipelineCoordinator, SemanticSearchEngine, SearchConfig, ConciseThemeIdentifier = import_rag_components()

# Enhanced session state initialization
def initialize_session_state():
    """Initialize session state with proper defaults"""
    defaults = {
        'coordinator': RAGPipelineCoordinator(),
        'search_engine': None,
        'theme_identifier': None,
        'search_history': [],
        'current_page': '🏠 Dashboard',
        'pipeline_running': False,
        'last_refresh': datetime.now(),
        'notifications': [],
        'user_preferences': {
            'theme': 'light',
            'auto_refresh': True,
            'notifications_enabled': True
        }
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Enhanced utility functions
def show_notification(message: str, type: str = "info", duration: int = 5):
    """Show animated notification"""
    notification = {
        'message': message,
        'type': type,
        'timestamp': datetime.now(),
        'duration': duration
    }
    st.session_state.notifications.append(notification)

def render_notifications():
    """Render active notifications"""
    current_time = datetime.now()
    active_notifications = []
    
    for notification in st.session_state.notifications:
        elapsed = (current_time - notification['timestamp']).seconds
        if elapsed < notification['duration']:
            active_notifications.append(notification)
    
    st.session_state.notifications = active_notifications
    
    for notification in active_notifications:
        if notification['type'] == 'success':
            st.success(notification['message'])
        elif notification['type'] == 'warning':
            st.warning(notification['message'])
        elif notification['type'] == 'error':
            st.error(notification['message'])
        else:
            st.info(notification['message'])

def create_metric_card(title: str, value: str, delta: str = None, help_text: str = None):
    """Create a beautiful metric card"""
    delta_html = f"<small style='color: #28a745;'>📈 {delta}</small>" if delta else ""
    help_html = f"<small style='color: #6c757d;'>{help_text}</small>" if help_text else ""
    
    return f"""
    <div class="metric-card fade-in">
        <h3 style="margin: 0; color: #2c3e50; font-size: 0.9rem; font-weight: 600;">{title}</h3>
        <div style="font-size: 2rem; font-weight: 700; color: #667eea; margin: 0.5rem 0;">{value}</div>
        {delta_html}
        {help_html}
    </div>
    """

def create_status_indicator(status: bool, text: str):
    """Create animated status indicator"""
    if status:
        return f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span style="color: #28a745; font-size: 1.2rem; margin-right: 0.5rem;">✅</span>
            <span style="color: #28a745; font-weight: 500;">{text}</span>
        </div>
        """
    else:
        return f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <span style="color: #dc3545; font-size: 1.2rem; margin-right: 0.5rem; animation: pulse 2s infinite;">❌</span>
            <span style="color: #dc3545; font-weight: 500;">{text}</span>
        </div>
        """

def initialize_search_engine():
    """Initialize search engine with error handling"""
    try:
        if st.session_state.search_engine is None:
            with st.spinner("🔍 Initializing search engine..."):
                config = SearchConfig()
                st.session_state.search_engine = SemanticSearchEngine(config)
                show_notification("Search engine initialized successfully!", "success")
        return True
    except Exception as e:
        st.error(f"Failed to initialize search engine: {e}")
        show_notification(f"Search engine initialization failed: {e}", "error")
        return False

def initialize_theme_identifier():
    """Initialize theme identifier with error handling"""
    try:
        if st.session_state.theme_identifier is None:
            with st.spinner("🎯 Initializing theme identifier..."):
                st.session_state.theme_identifier = ConciseThemeIdentifier()
                show_notification("Theme identifier initialized successfully!", "success")
        return True
    except Exception as e:
        st.error(f"Failed to initialize theme identifier: {e}")
        show_notification(f"Theme identifier initialization failed: {e}", "error")
        return False

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_pipeline_status():
    """Get current pipeline status with caching"""
    coordinator = st.session_state.coordinator
    coordinator.update_pipeline_status()
    config = coordinator.check_configuration()
    return coordinator.pipeline_status, config

def render_enhanced_sidebar():
    """Enhanced sidebar with better navigation and status"""
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #667eea; margin: 0; font-size: 1.8rem;">🤖 RAG Pipeline</h1>
            <p style="color: #6c757d; margin: 0.5rem 0; font-size: 0.9rem;">Intelligent Document Processing</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown("### 📋 Navigation")
        pages = [
            ("🏠", "Dashboard", "Overview and quick actions"),
            ("⚙️", "Pipeline Management", "Control pipeline stages"),
            ("🔍", "Search & Query", "Semantic document search"),
            ("🎯", "Theme Analysis", "Identify document themes"),
            ("📊", "Analytics", "Performance metrics"),
            ("⚙️", "Settings", "Configuration options")
        ]
        
        for icon, name, description in pages:
            full_name = f"{icon} {name}"
            if st.button(
                full_name, 
                key=f"nav_{name}",
                use_container_width=True,
                help=description
            ):
                st.session_state.current_page = full_name
                st.rerun()
        
        st.markdown("---")
        
        # Enhanced Pipeline Status
        st.markdown("### 📊 Pipeline Status")
        pipeline_status, config = get_pipeline_status()
        
        stages = [
            ("📄 OCR & Parsing", pipeline_status["ocr_completed"]),
            ("🔪 Chunking", pipeline_status["chunking_completed"]),
            ("🔑 Keywords", pipeline_status["keywords_completed"]),
            ("🧮 Embeddings", pipeline_status["embeddings_completed"]),
            ("🔍 Ready for Search", pipeline_status["ready_for_search"])
        ]
        
        completed_stages = sum(status for _, status in stages)
        total_stages = len(stages)
        progress = completed_stages / total_stages
        
        st.progress(progress)
        st.markdown(f"**Progress:** {completed_stages}/{total_stages} stages completed")
        
        for stage, completed in stages:
            st.markdown(create_status_indicator(completed, stage), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Configuration Status
        st.markdown("### 🔧 Configuration")
        api_keys = config["api_keys"]
        
        st.markdown(create_status_indicator(api_keys['groq_api_key'], "Groq API"), unsafe_allow_html=True)
        st.markdown(create_status_indicator(api_keys['openai_api_key'], "OpenAI API"), unsafe_allow_html=True)
        st.markdown(create_status_indicator(api_keys['gemini_api_key'], "Gemini API"), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        pdf_count = config["pdf_directory"]["pdf_count"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📁 PDFs", pdf_count)
        with col2:
            search_count = len(st.session_state.search_history)
            st.metric("🔍 Searches", search_count)
        
        # Auto-refresh toggle
        if st.checkbox("🔄 Auto-refresh", value=st.session_state.user_preferences['auto_refresh']):
            st.session_state.user_preferences['auto_refresh'] = True
            if st.button("🔄 Refresh Now", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        else:
            st.session_state.user_preferences['auto_refresh'] = False

def render_dashboard():
    """Enhanced dashboard with better visualizations"""
    st.markdown('<h1 class="main-header">🤖 RAG Pipeline Dashboard</h1>', unsafe_allow_html=True)
    
    # Render notifications
    render_notifications()
    
    # Get status
    pipeline_status, config = get_pipeline_status()
    
    # Enhanced overview metrics
    st.markdown('<h2 class="section-header">📊 Overview Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pdf_count = config["pdf_directory"]["pdf_count"]
        st.markdown(create_metric_card(
            "📁 PDF Documents", 
            str(pdf_count),
            help_text="Total PDF files in directory"
        ), unsafe_allow_html=True)
    
    with col2:
        completed_stages = sum(pipeline_status.values())
        st.markdown(create_metric_card(
            "✅ Completed Stages", 
            f"{completed_stages}/5",
            delta=f"{completed_stages/5*100:.0f}% complete" if completed_stages > 0 else None,
            help_text="Pipeline completion progress"
        ), unsafe_allow_html=True)
    
    with col3:
        api_configured = sum(config["api_keys"].values())
        st.markdown(create_metric_card(
            "🔑 API Keys", 
            f"{api_configured}/3",
            help_text="Configured API endpoints"
        ), unsafe_allow_html=True)
    
    with col4:
        ready = "Ready" if pipeline_status["ready_for_search"] else "Not Ready"
        color = "#28a745" if pipeline_status["ready_for_search"] else "#dc3545"
        st.markdown(f"""
        <div class="metric-card fade-in">
            <h3 style="margin: 0; color: #2c3e50; font-size: 0.9rem; font-weight: 600;">🚀 Search Status</h3>
            <div style="font-size: 1.5rem; font-weight: 700; color: {color}; margin: 0.5rem 0;">{ready}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced pipeline visualization
    st.markdown('<h2 class="section-header">🔄 Pipeline Progress</h2>', unsafe_allow_html=True)
    
    stages_data = {
        "Stage": ["OCR & Parsing", "Chunking", "Keywords", "Embeddings", "Search Ready"],
        "Status": [
            pipeline_status["ocr_completed"],
            pipeline_status["chunking_completed"],
            pipeline_status["keywords_completed"],
            pipeline_status["embeddings_completed"],
            pipeline_status["ready_for_search"]
        ],
        "Progress": [1 if s else 0 for s in [
            pipeline_status["ocr_completed"],
            pipeline_status["chunking_completed"],
            pipeline_status["keywords_completed"],
            pipeline_status["embeddings_completed"],
            pipeline_status["ready_for_search"]
        ]]
    }
    
    df = pd.DataFrame(stages_data)
    df["Status_Text"] = df["Status"].map({True: "Completed", False: "Pending"})
    
    # Create enhanced visualization
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=df["Stage"],
        y=df["Progress"],
        marker_color=['#28a745' if s else '#e74c3c' for s in df["Status"]],
        text=df["Status_Text"],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Status: %{text}<extra></extra>',
        name='Pipeline Stage'
    ))
    
    fig.update_layout(
        title={
            'text': 'Pipeline Stages Status',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'family': 'Inter'}
        },
        yaxis=dict(
            tickmode='array', 
            tickvals=[0, 1], 
            ticktext=['Pending', 'Completed'],
            gridcolor='rgba(128,128,128,0.2)'
        ),
        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick Actions with enhanced styling
    st.markdown('<h2 class="section-header">🚀 Quick Actions</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Run Complete Pipeline", use_container_width=True, type="primary"):
            show_notification("Starting complete pipeline...", "info")
            st.session_state.current_page = "⚙️ Pipeline Management"
            st.rerun()
    
    with col2:
        if st.button(
            "🔍 Test Search", 
            use_container_width=True, 
            disabled=not pipeline_status["ready_for_search"],
            help="Search functionality requires completed embeddings"
        ):
            st.session_state.current_page = "🔍 Search & Query"
            st.rerun()
    
    with col3:
        if st.button("🎯 Analyze Themes", use_container_width=True):
            st.session_state.current_page = "🎯 Theme Analysis"
            st.rerun()
    
    with col4:
        if st.button("📊 View Analytics", use_container_width=True):
            st.session_state.current_page = "📊 Analytics"
            st.rerun()
    
    # Recent Activity with enhanced display
    st.markdown('<h2 class="section-header">📈 Recent Activity</h2>', unsafe_allow_html=True)
    
    # Create tabs for different activity types
    tab1, tab2, tab3 = st.tabs(["📄 Recent Files", "🔍 Search History", "⚡ System Events"])
    
    with tab1:
        recent_files = []
        coordinator = st.session_state.coordinator
        
        # Check processed documents
        if coordinator.processed_dir.exists():
            for file in sorted(coordinator.processed_dir.glob("*.json"), 
                             key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                recent_files.append({
                    "Type": "📄 Processed Document",
                    "File": file.name,
                    "Size": f"{file.stat().st_size / 1024:.1f} KB",
                    "Modified": datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
        
        if recent_files:
            st.dataframe(
                pd.DataFrame(recent_files), 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No recent files found. Run the pipeline to see activity here.")
    
    with tab2:
        if st.session_state.search_history:
            history_df = pd.DataFrame(st.session_state.search_history[-10:])  # Show last 10
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.strftime('%H:%M:%S')
            st.dataframe(
                history_df[['timestamp', 'query', 'confidence', 'sources_count']], 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No search history yet. Try the search functionality!")
    
    with tab3:
        # System events (you can expand this based on your logging)
        events = [
            {"Time": "09:15", "Event": "🔄 Pipeline initialized", "Status": "✅"},
            {"Time": "09:10", "Event": "📁 Environment loaded", "Status": "✅"},
            {"Time": "09:05", "Event": "🚀 Dashboard started", "Status": "✅"}
        ]
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)

def render_pipeline_management():
    """Enhanced pipeline management with better UX"""
    st.markdown('<h1 class="main-header">⚙️ Pipeline Management</h1>', unsafe_allow_html=True)
    
    coordinator = st.session_state.coordinator
    pipeline_status, config = get_pipeline_status()
    
    # Pipeline overview with progress
    st.markdown('<h2 class="section-header">📊 Pipeline Overview</h2>', unsafe_allow_html=True)
    
    # Create a visual pipeline flow
    col1, col2, col3, col4, col5 = st.columns(5)
    stages = [
        ("📄", "OCR", pipeline_status["ocr_completed"]),
        ("🔪", "Chunk", pipeline_status["chunking_completed"]),
        ("🔑", "Keywords", pipeline_status["keywords_completed"]),
        ("🧮", "Embed", pipeline_status["embeddings_completed"]),
        ("🔍", "Search", pipeline_status["ready_for_search"])
    ]
    
    for i, (col, (icon, name, completed)) in enumerate(zip([col1, col2, col3, col4, col5], stages)):
        with col:
            status_color = "#28a745" if completed else "#dc3545"
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; border-radius: 10px; 
                        background: {'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)' if completed else 'linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100)'};
                        border: 2px solid {status_color};">
                <div style="font-size: 2rem;">{icon}</div>
                <div style="font-weight: 600; color: {status_color};">{name}</div>
                <div style="color: {status_color};">{'✅' if completed else '⏳'}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced individual pipeline steps
    st.markdown('<h2 class="section-header">🔧 Pipeline Steps</h2>', unsafe_allow_html=True)
    
    # Step 1: OCR
    with st.expander("📄 Step 1: OCR & Document Parsing", expanded=not pipeline_status['ocr_completed']):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Status:** {'✅ Completed' if pipeline_status['ocr_completed'] else '❌ Pending'}")
            st.write(f"**PDF Files:** {config['pdf_directory']['pdf_count']}")
            st.write("**Description:** Extract text from PDF documents using OCR technology")
        
        with col2:
            if st.button("🚀 Run OCR", key="run_ocr", type="primary", use_container_width=True):
                with st.spinner("🔄 Running OCR processing..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)  # Simulate processing
                        progress_bar.progress(i + 1)
                    
                    success = coordinator.run_ocr_parsing()
                    if success:
                        show_notification("OCR completed successfully!", "success")
                        st.rerun()
                    else:
                        show_notification("OCR processing failed!", "error")
            
            force_ocr = st.checkbox("🔥 Force rerun", key="force_ocr")
            if st.button("🔄 Force Run OCR", key="force_run_ocr", disabled=not force_ocr, use_container_width=True):
                with st.spinner("🔄 Force running OCR..."):
                    success = coordinator.run_ocr_parsing(force=True)
                    if success:
                        show_notification("OCR force completed!", "success")
                        st.rerun()
                    else:
                        show_notification("OCR force run failed!", "error")
    
    # Step 2: Chunking
    with st.expander("🔪 Step 2: Document Chunking", expanded=pipeline_status['ocr_completed'] and not pipeline_status['chunking_completed']):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Status:** {'✅ Completed' if pipeline_status['chunking_completed'] else '❌ Pending'}")
            st.write("**Requirements:** Completed OCR processing")
            st.write("**Description:** Split documents into optimized chunks for processing")
        
        with col2:
            if st.button("🚀 Run Chunking", key="run_chunking", 
                        disabled=not pipeline_status['ocr_completed'], 
                        type="primary", use_container_width=True):
                with st.spinner("🔄 Running document chunking..."):
                    success = coordinator.run_chunking()
                    if success:
                        show_notification("Chunking completed successfully!", "success")
                        st.rerun()
                    else:
                        show_notification("Chunking failed!", "error")
            
            force_chunking = st.checkbox("🔥 Force rerun", key="force_chunking")
            if st.button("🔄 Force Run Chunking", key="force_run_chunking", 
                        disabled=not force_chunking, use_container_width=True):
                with st.spinner("🔄 Force running chunking..."):
                    success = coordinator.run_chunking(force=True)
                    if success:
                        show_notification("Chunking force completed!", "success")
                        st.rerun()
                    else:
                        show_notification("Chunking force run failed!", "error")
    
    # Step 3: Keywords
    with st.expander("🔑 Step 3: Keyword Extraction", expanded=pipeline_status['chunking_completed'] and not pipeline_status['keywords_completed']):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Status:** {'✅ Completed' if pipeline_status['keywords_completed'] else '❌ Pending'}")
            st.write("**Requirements:** Groq API Key, Completed Chunking")
            st.write("**Description:** Extract relevant keywords from document chunks")
            if not config['api_keys']['groq_api_key']:
                st.warning("⚠️ Groq API key required. Configure in Settings.")
        
        with col2:
            can_run_keywords = pipeline_status['chunking_completed'] and config['api_keys']['groq_api_key']
            if st.button("🚀 Run Keywords", key="run_keywords", 
                        disabled=not can_run_keywords, 
                        type="primary", use_container_width=True):
                with st.spinner("🔄 Extracting keywords..."):
                    success = coordinator.run_keyword_extraction()
                    if success:
                        show_notification("Keyword extraction completed!", "success")
                        st.rerun()
                    else:
                        show_notification("Keyword extraction failed!", "error")
            
            force_keywords = st.checkbox("🔥 Force rerun", key="force_keywords")
            if st.button("🔄 Force Run Keywords", key="force_run_keywords", 
                        disabled=not (force_keywords and can_run_keywords), use_container_width=True):
                with st.spinner("🔄 Force running keyword extraction..."):
                    success = coordinator.run_keyword_extraction(force=True)
                    if success:
                        show_notification("Keywords force completed!", "success")
                        st.rerun()
                    else:
                        show_notification("Keywords force run failed!", "error")
    
    # Step 4: Embeddings
    with st.expander("🧮 Step 4: Embeddings & Vector Storage", expanded=pipeline_status['chunking_completed'] and not pipeline_status['embeddings_completed']):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**Status:** {'✅ Completed' if pipeline_status['embeddings_completed'] else '❌ Pending'}")
            st.write("**Requirements:** OpenAI/Gemini API Key, Qdrant, Completed Chunking")
            st.write("**Description:** Create vector embeddings and store in Qdrant database")
            if not (config['api_keys']['openai_api_key'] or config['api_keys']['gemini_api_key']):
                st.warning("⚠️ OpenAI or Gemini API key required. Configure in Settings.")
        
        with col2:
            can_run_embeddings = (pipeline_status['chunking_completed'] and 
                                 (config['api_keys']['openai_api_key'] or config['api_keys']['gemini_api_key']))
            if st.button("🚀 Run Embeddings", key="run_embeddings", 
                        disabled=not can_run_embeddings, 
                        type="primary", use_container_width=True):
                with st.spinner("🔄 Creating embeddings and storing in vector database..."):
                    success = coordinator.run_embeddings()
                    if success:
                        show_notification("Embeddings completed successfully!", "success")
                        st.rerun()
                    else:
                        show_notification("Embeddings failed!", "error")
            
            force_embeddings = st.checkbox("🔥 Force rerun", key="force_embeddings")
            if st.button("🔄 Force Run Embeddings", key="force_run_embeddings", 
                        disabled=not (force_embeddings and can_run_embeddings), use_container_width=True):
                with st.spinner("🔄 Force running embeddings..."):
                    success = coordinator.run_embeddings(force=True)
                    if success:
                        show_notification("Embeddings force completed!", "success")
                        st.rerun()
                    else:
                        show_notification("Embeddings force run failed!", "error")
    
    # Complete pipeline section
    st.markdown('<h2 class="section-header">🚀 Complete Pipeline</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-status">
            <h4>🔄 Run Complete Pipeline</h4>
            <p>Execute all pipeline stages in sequence. This will run only missing stages.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Run Complete Pipeline", key="complete_pipeline", 
                    type="primary", use_container_width=True):
            st.session_state.pipeline_running = True
            
            with st.container():
                st.markdown("### 🚀 Pipeline Execution")
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    steps = ["OCR & Parsing", "Chunking", "Keywords", "Embeddings"]
                    
                    for i, step in enumerate(steps):
                        status_text.markdown(f"**🔄 Running {step}...**")
                        progress_bar.progress((i + 1) / len(steps))
                        time.sleep(1)  # Simulate processing time
                    
                    status_text.markdown("**✅ Executing complete pipeline...**")
                    success = coordinator.run_complete_pipeline()
                    
                    if success:
                        st.success("🎉 Complete pipeline finished successfully!")
                        st.balloons()
                        show_notification("Pipeline completed successfully!", "success")
                        st.rerun()
                    else:
                        st.error("❌ Pipeline failed!")
                        show_notification("Pipeline execution failed!", "error")
                
                st.session_state.pipeline_running = False
    
    with col2:
        st.markdown("""
        <div class="warning-status">
            <h4>🔥 Force Run Complete Pipeline</h4>
            <p>Re-run all stages regardless of current status. Use with caution!</p>
        </div>
        """, unsafe_allow_html=True)
        
        force_all = st.checkbox("⚠️ I understand this will re-run all stages", key="force_all")
        if st.button("🔥 Force Run Complete Pipeline", key="force_complete_pipeline", 
                    disabled=not force_all, use_container_width=True):
            if st.checkbox("🔴 Final confirmation", key="final_confirm"):
                with st.spinner("🔄 Force running complete pipeline..."):
                    success = coordinator.run_complete_pipeline(force_all=True)
                    if success:
                        st.success("🎉 Force complete pipeline finished!")
                        st.balloons()
                        show_notification("Force pipeline completed!", "success")
                        st.rerun()
                    else:
                        st.error("❌ Force pipeline failed!")
                        show_notification("Force pipeline failed!", "error")

def render_enhanced_search():
    """Enhanced search interface with better UX"""
    st.markdown('<h1 class="main-header">🔍 Search & Query</h1>', unsafe_allow_html=True)
    
    pipeline_status, _ = get_pipeline_status()
    
    if not pipeline_status["ready_for_search"]:
        st.markdown("""
        <div class="warning-status">
            <h3>🚧 Pipeline Not Ready</h3>
            <p>Search functionality requires completed embeddings stage. Please complete the pipeline first.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 Go to Pipeline Management", type="primary"):
            st.session_state.current_page = "⚙️ Pipeline Management"
            st.rerun()
        return
    
    # Initialize search engine
    if not initialize_search_engine():
        return
    
    # Enhanced search interface
    st.markdown('<h2 class="section-header">🔍 Semantic Search</h2>', unsafe_allow_html=True)
    
    # Search input with better styling
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "🔍 Enter your question:",
            placeholder="What are the main topics discussed in the documents?",
            help="Ask questions about your documents using natural language",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Advanced search options in expandable section
    with st.expander("⚙️ Advanced Search Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            top_k = st.slider("Number of results", 1, 20, 5, help="Maximum number of search results to return")
        
        with col2:
            confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.3, 0.1, 
                                            help="Minimum confidence score for results")
        
        with col3:
            detailed_results = st.checkbox("Show detailed results", value=False, 
                                         help="Include additional search metadata")
    
    # Search execution
    if (search_button or st.session_state.get('auto_search', False)) and query:
        with st.spinner("🔍 Searching through your documents..."):
            try:
                start_time = time.time()
                
                # Create progress indicator
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulate search progress
                for i, step in enumerate(["Analyzing query", "Searching vectors", "Ranking results", "Generating answer"]):
                    status_text.text(f"🔄 {step}...")
                    progress_bar.progress((i + 1) / 4)
                    time.sleep(0.2)
                
                # Perform actual search
                result = st.session_state.search_engine.search_and_answer(
                    query, 
                    detailed_output=detailed_results
                )
                
                search_time = time.time() - start_time
                progress_bar.empty()
                status_text.empty()
                
                # Add to history
                st.session_state.search_history.append({
                    "query": query,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": result["confidence"],
                    "sources_count": len(result["sources"]),
                    "processing_time": search_time
                })
                
                # Display results with enhanced styling
                st.markdown('<h2 class="section-header">💬 Search Results</h2>', unsafe_allow_html=True)
                
                # Confidence indicator with color coding
                confidence_colors = {
                    "very_high": "#28a745",
                    "high": "#17a2b8", 
                    "medium": "#ffc107",
                    "low": "#fd7e14",
                    "very_low": "#dc3545"
                }
                
                confidence_color = confidence_colors.get(result["confidence"], "#6c757d")
                confidence_text = result["confidence"].replace("_", " ").title()
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {confidence_color}, {confidence_color}dd); 
                           color: white; padding: 1rem; border-radius: 12px; margin-bottom: 1rem;
                           box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        🎯 Confidence: {confidence_text}
                    </h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Answer section
                st.markdown("### 💬 Answer")
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                           padding: 1.5rem; border-radius: 12px; border-left: 4px solid #667eea;
                           box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                    {result["answer"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱️ Processing Time", f"{result['processing_time']:.2f}s")
                with col2:
                    st.metric("📊 Search Results", result["search_results_count"])
                with col3:
                    st.metric("📚 Sources Used", len(result["sources"]))
                
                # Sources with enhanced display
                if result["sources"]:
                    st.markdown("### 📚 Sources")
                    
                    for i, source in enumerate(result["sources"], 1):
                        with st.expander(f"📖 Source {i}: {source['doc_name']} (Page {source['page']})", expanded=i<=2):
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown(f"**Document:** {source['doc_name']}")
                                st.markdown(f"**Page:** {source['page']}")
                                st.markdown(f"**Paragraph:** {source['paragraph']}")
                                if source.get('keywords'):
                                    st.markdown(f"**Keywords:** {', '.join(source['keywords'][:5])}")
                            
                            with col2:
                                # Create a relevance score gauge
                                score = source['relevance_score']
                                st.markdown(f"""
                                <div style="text-align: center;">
                                    <div style="font-size: 0.8rem; color: #666;">Relevance Score</div>
                                    <div style="font-size: 1.5rem; font-weight: bold; color: {'#28a745' if score > 0.7 else '#ffc107' if score > 0.4 else '#dc3545'};">
                                        {score:.3f}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Detailed results table
                if detailed_results and "all_search_results" in result:
                    st.markdown("### 🔍 Detailed Search Results")
                    
                    search_df = pd.DataFrame([
                        {
                            "Document": r["doc_name"],
                            "Page": r["page"],
                            "Score": f"{r['score']:.3f}",
                            "Confidence": r["confidence"],
                            "Preview": r["text"][:100] + "..."
                        }
                        for r in result["all_search_results"][:10]
                    ])
                    
                    st.dataframe(search_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"🚨 Search failed: {e}")
                show_notification(f"Search failed: {e}", "error")
    
    # Enhanced search history
    if st.session_state.search_history:
        st.markdown('<h2 class="section-header">📈 Search History</h2>', unsafe_allow_html=True)
        
        # History controls
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.search_history = []
                show_notification("Search history cleared!", "info")
                st.rerun()
        
        with col2:
            if st.button("📊 Export History", use_container_width=True):
                # Create downloadable CSV
                history_df = pd.DataFrame(st.session_state.search_history)
                csv = history_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download CSV",
                    data=csv,
                    file_name=f"search_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            show_analytics = st.checkbox("📊 Show Analytics", value=False)
        
        # Display history
        history_df = pd.DataFrame(st.session_state.search_history)
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        history_df = history_df.sort_values("timestamp", ascending=False)
        
        if show_analytics:
            # Search analytics
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence distribution
                confidence_counts = history_df['confidence'].value_counts()
                fig_confidence = px.pie(
                    values=confidence_counts.values,
                    names=confidence_counts.index,
                    title="Search Confidence Distribution",
                    color_discrete_map=confidence_colors
                )
                st.plotly_chart(fig_confidence, use_container_width=True)
            
            with col2:
                # Processing time trend
                fig_time = px.line(
                    history_df.head(20),
                    x='timestamp',
                    y='processing_time',
                    title="Search Processing Time Trend",
                    markers=True
                )
                st.plotly_chart(fig_time, use_container_width=True)
        
        # Recent searches
        st.markdown("#### 🔍 Recent Searches")
        for i, (_, search) in enumerate(history_df.head(5).iterrows()):
            with st.expander(f"🔍 {search['query'][:50]}{'...' if len(search['query']) > 50 else ''} ({search['confidence']})", expanded=i==0):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write(f"**Time:** {search['timestamp'].strftime('%H:%M:%S')}")
                with col2:
                    st.write(f"**Sources:** {search['sources_count']}")
                with col3:
                    st.write(f"**Duration:** {search['processing_time']:.2f}s")
                with col4:
                    if st.button("🔄 Repeat Search", key=f"repeat_{i}"):
                        st.session_state.repeat_query = search['query']
                        st.rerun()

def main():
    """Enhanced main application with better routing"""
    
    # Render enhanced sidebar
    render_enhanced_sidebar()
    
    # Auto-refresh logic
    if st.session_state.user_preferences['auto_refresh']:
        if (datetime.now() - st.session_state.last_refresh).seconds > 30:
            st.cache_data.clear()
            st.session_state.last_refresh = datetime.now()
    
    # Route to appropriate page based on session state
    current_page = st.session_state.current_page
    
    # Page routing with enhanced error handling
    try:
        if current_page == "🏠 Dashboard":
            render_dashboard()
        
        elif current_page == "⚙️ Pipeline Management":
            render_pipeline_management()
        
        elif current_page == "🔍 Search & Query":
            render_enhanced_search()
        
        elif current_page == "🎯 Theme Analysis":
            render_theme_analysis()  # You'll need to implement this
        
        elif current_page == "📊 Analytics":
            render_analytics()  # You'll need to implement this
        
        elif current_page == "⚙️ Settings":
            render_settings()  # You'll need to implement this
        
        else:
            st.error(f"Unknown page: {current_page}")
            st.session_state.current_page = "🏠 Dashboard"
            st.rerun()
    
    except Exception as e:
        st.error(f"Error rendering page {current_page}: {e}")
        show_notification(f"Page error: {e}", "error")
        st.session_state.current_page = "🏠 Dashboard"
        
        # Debug information
        with st.expander("🐛 Debug Information"):
            st.code(f"Error: {e}")
            st.code(f"Page: {current_page}")
    
    # Footer with enhanced styling
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #666; font-size: 0.9rem; 
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                border-radius: 12px; margin-top: 2rem;'>
        🤖 <strong>RAG Pipeline Dashboard</strong> | Built with ❤️ using Streamlit<br>
        <small>Version 2.0 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small><br>
        <a href='https://github.com/your-username/rag-pipeline' target='_blank' style='color: #667eea; text-decoration: none;'>
            📚 Documentation
        </a> | 
        <a href='https://github.com/your-username/rag-pipeline/issues' target='_blank' style='color: #667eea; text-decoration: none;'>
            🐛 Report Issues
        </a>
    </div>
    """, unsafe_allow_html=True)

# Placeholder functions for remaining pages (you'll need to implement these)
def render_theme_analysis():
    st.markdown('<h1 class="main-header">🎯 Theme Analysis</h1>', unsafe_allow_html=True)
    st.info("🚧 Theme Analysis functionality - Implementation needed")

def render_analytics():
    st.markdown('<h1 class="main-header">📊 Analytics</h1>', unsafe_allow_html=True)
    st.info("🚧 Analytics functionality - Implementation needed")

def render_settings():
    st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
    st.info("🚧 Settings functionality - Implementation needed")

if __name__ == "__main__":
    main()