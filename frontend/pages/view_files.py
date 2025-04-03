import streamlit as st
from utils.api_client import APIClient
from routers.file_display import render_file_grid

def render_view_files_page(api_client: APIClient):
    """Render the view files page."""
    st.header("View Files")
    render_file_grid(api_client) 