import streamlit as st
from utils.api_client import APIClient

def render_analytics_page(api_client: APIClient):
    """Render the analytics page."""
    st.header("Analytics")
    st.info("Analytics features coming soon!") 