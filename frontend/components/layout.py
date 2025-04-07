import streamlit as st
from typing import Callable
from pages.settings import render_settings_page
from utils.api_client import APIClient

# Constants
BASE_URL = "http://backend:8000"
LLM_API_URL = "http://llm:8100"

def render_header():
    """Render the header with title, icon, and settings."""
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        st.image("static/logo.png", width=50)  # You'll need to add a logo image
    
    with col2:
        st.title("Data Down Under")
    
    with col3:
        if st.button("⚙️", key="settings_button"):
            # Store the current page before switching to settings
            if "previous_page" not in st.session_state:
                st.session_state.previous_page = st.session_state.current_page
            st.session_state.current_page = "Settings"

def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("Navigation")

    # Custom CSS to hide the default page links in the sidebar.
    st.markdown("""
                <style>
                [data-testid="stSidebarNav"] {
                    display: none;
                }
                </style>
                """, unsafe_allow_html=True)
    
    # Define pages
    pages = {
        "Dashboard": ("📊", "pages/dashboard.py"),
        "Upload": ("📤", "pages/upload.py"),
        "View": ("🎥", "pages/view.py"),
        "Analytics": ("📈", "pages/analytics.py"),
        "Settings": ("⚙️", "pages/settings.py")
    }
    
    # Initialize session state for current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Render navigation buttons
    for page, (icon, path) in pages.items():
        if st.sidebar.button(f"{icon} {page}", key=f"nav_{page}"):
            # Store the previous page when navigating
            if page != "Settings":  # Don't overwrite previous page when going to settings
                st.session_state.previous_page = st.session_state.current_page
            st.session_state.current_page = page

def render_layout(page_content: Callable):
    """Main layout wrapper that includes header and sidebar."""
    # Initialize session state
    if "previous_page" not in st.session_state:
        st.session_state.previous_page = "Dashboard"
    
    # Initialize API client
    api_client = APIClient(BASE_URL, LLM_API_URL)
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content based on current page
    if st.session_state.current_page == "Settings":
        render_settings_page(api_client)
    else:
        page_content() 