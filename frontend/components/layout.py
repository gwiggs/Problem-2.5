import streamlit as st
from typing import Callable

def render_header():
    """Render the header with title, icon, and settings."""
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        st.image("static/logo.png", width=50)  # You'll need to add a logo image
    
    with col2:
        st.title("Data Down Under")
    
    with col3:
        if st.button("⚙️", key="settings_button"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)

def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("Navigation")
    
    # Define pages
    pages = {
        "Dashboard": "📊",
        "Upload Files": "📤",
        "View Files": "🎥",
        "Analytics": "📈"
    }
    
    # Initialize session state for current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Render navigation buttons
    for page, icon in pages.items():
        if st.sidebar.button(f"{icon} {page}", key=f"nav_{page}"):
            st.session_state.current_page = page

def render_layout(page_content: Callable):
    """Main layout wrapper that includes header and sidebar."""
    # Initialize session state
    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Render settings if toggled
    if st.session_state.show_settings:
        with st.expander("Settings", expanded=True):
            st.write("Settings content will go here")
    
    # Render main content
    page_content() 