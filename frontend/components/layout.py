import streamlit as st
from typing import Callable

# Custom CSS for layout
st.markdown("""
    <style>
    /* Hide the default Streamlit header */
    .stApp > header {
        display: none;
    }
    
    /* Custom header styling */
    .custom-header {
        background-color: #f0f2f6;
        padding: 1rem;
        margin: -1rem -1rem 1rem -1rem;
        border-bottom: 1px solid #e6e9ef;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        margin-top: 0;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the header with title, icon, and settings."""
    # Create a custom header div
    st.markdown("""
        <div class="custom-header">
            <div style="display: flex; align-items: center; justify-content: space-between; max-width: 1200px; margin: 0 auto;">
                <div style="display: flex; align-items: center;">
                    <img src="static/logo.png" style="width: 50px; height: 50px; margin-right: 1rem;">
                    <h1 style="margin: 0;">Data Down Under</h1>
                </div>
                <div>
                    <button onclick="document.dispatchEvent(new CustomEvent('settings-toggle'))" style="background: none; border: none; font-size: 1.5rem; cursor: pointer;">⚙️</button>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Handle settings toggle
    st.markdown("""
        <script>
        document.addEventListener('settings-toggle', function() {
            const event = new CustomEvent('streamlit:componentMessage', {
                detail: {type: 'settings-toggle'}
            });
            window.parent.document.dispatchEvent(event);
        });
        </script>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar navigation."""
    # Create a horizontal navigation bar
    st.markdown("""
        <style>
        .nav-container {
            display: flex;
            gap: 1rem;
            padding: 1rem;
            background-color: #f0f2f6;
            border-bottom: 1px solid #e6e9ef;
            margin: -1rem -1rem 1rem -1rem;
        }
        .nav-button {
            padding: 0.5rem 1rem;
            border: none;
            background: none;
            cursor: pointer;
            font-size: 1rem;
            color: #262730;
        }
        .nav-button:hover {
            background-color: #e6e9ef;
            border-radius: 0.25rem;
        }
        .nav-button.active {
            background-color: #e6e9ef;
            border-radius: 0.25rem;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)
    
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
    
    # Create navigation buttons
    nav_html = '<div class="nav-container">'
    for page, icon in pages.items():
        is_active = "active" if st.session_state.current_page == page else ""
        nav_html += f'''
            <button class="nav-button {is_active}" 
                    onclick="document.dispatchEvent(new CustomEvent('nav-click', {{detail: {{page: '{page}'}}}}))">
                {icon} {page}
            </button>
        '''
    nav_html += '</div>'
    
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Handle navigation clicks
    st.markdown("""
        <script>
        document.addEventListener('nav-click', function(e) {
            const event = new CustomEvent('streamlit:componentMessage', {
                detail: {type: 'nav-click', page: e.detail.page}
            });
            window.parent.document.dispatchEvent(event);
        });
        </script>
    """, unsafe_allow_html=True)

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