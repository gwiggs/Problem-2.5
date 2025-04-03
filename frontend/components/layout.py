import streamlit as st
from typing import Callable

# Set Streamlit page configuration to enable wide mode
st.set_page_config(
    page_title="Data Down Under",
    page_icon="🌏",
    layout="wide",  # Enable wide mode
)

def render_header():
    """Render the header with title, icon, and settings."""
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        # Increase the width of the logo
        st.image("static/logo.png", width=150)  # Adjust the width as needed
    
    with col2:
        st.title("Data Down Under")
    
    with col3:
        if st.button("⚙️", key="settings_button"):
            st.session_state.show_settings = not st.session_state.get("show_settings", False)

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
        "Analytics": ("📈", "pages/analytics.py")
    }
    
    # Initialize session state for current page
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Render navigation buttons
    for page, (icon, path) in pages.items():
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