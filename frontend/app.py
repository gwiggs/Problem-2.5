# from classes.frontend_class_example import HelloWorld
import streamlit as st
from utils.api_client import APIClient
from components.layout import render_layout
from pages.dashboard import render_dashboard
from pages.upload import render_upload_page
from pages.view_files import render_view_files_page
from pages.analytics import render_analytics_page

# Constants
BASE_URL = "http://backend:8000"
LLM_API_URL = "http://llm:8100"

def main():
    """Main application entry point."""
    # Initialize API client
    api_client = APIClient(BASE_URL, LLM_API_URL)
    
    # Initialize session state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    def render_content():
        """Render the appropriate page based on current selection."""
        if st.session_state.current_page == "Dashboard":
            render_dashboard(api_client)
        elif st.session_state.current_page == "Upload":
            render_upload_page(api_client)
        elif st.session_state.current_page == "View":
            render_view_files_page(api_client)
        elif st.session_state.current_page == "Analytics":
            render_analytics_page(api_client)
    
    # Render the layout with the current page content
    render_layout(render_content)

if __name__ == "__main__":
    main()
