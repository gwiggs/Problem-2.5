from utils.api_client import APIClient
from routers.file_upload import render_file_upload

def render_upload_page(api_client: APIClient):
    """Render the upload page."""
    render_file_upload(api_client) 