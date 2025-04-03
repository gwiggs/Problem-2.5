import streamlit as st
from utils.api_client import APIClient
from routers.file_upload import render_file_upload

def render_upload_page(api_client: APIClient):
    """Render the upload page."""
    st.header("Upload Files")
    render_file_upload(api_client) 