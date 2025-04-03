# from classes.frontend_class_example import HelloWorld
import streamlit as st
from utils.api_client import APIClient
from routers.file_upload import render_file_upload
from routers.file_display import render_file_grid

# Constants
BASE_URL = "http://backend:8000"

def main():
    """Main application entry point."""
    st.title("Batch Files")
    
    # Initialize API client
    api_client = APIClient(BASE_URL)
    
    # Initialize session state for refresh
    if "refresh" not in st.session_state:
        st.session_state.refresh = False
    
    # Render file upload section
    render_file_upload(api_client)
    
    # Render file display section
    render_file_grid(api_client)

if __name__ == "__main__":
    main()
