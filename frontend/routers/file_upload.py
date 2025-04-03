from typing import List, Optional
import streamlit as st
from utils.api_client import APIClient
from models.schemas import FileResponse

def render_file_upload(api_client: APIClient) -> Optional[List[FileResponse]]:
    """Render the file upload section and handle uploads."""
    st.header("Upload Files")
    
    uploaded_files = st.file_uploader(
        "Choose files (videos or images)",
        type=["mp4", "avi", "mov", "jpg", "png"],
        accept_multiple_files=True
    )
    
    if not uploaded_files:
        return None
        
    upload_results = []
    for uploaded_file in uploaded_files:
        try:
            result = api_client.upload_file(
                file_data=uploaded_file.getvalue(),
                filename=uploaded_file.name,
                content_type=uploaded_file.type
            )
            st.success(f"Upload successful: {result.message} for {uploaded_file.name}")
            upload_results.append(result)
        except ValueError as e:
            st.error(f"Upload failed for {uploaded_file.name}: {str(e)}")
            
    return upload_results 