from typing import List
import streamlit as st
from utils.api_client import APIClient
from models.schemas import FileMetadata

def render_metadata(metadata: FileMetadata) -> None:
    """Render file metadata in an expander."""
    with st.expander("Metadata Details"):
        for key, value in metadata.dict().items():
            if key == "additional_metadata":
                continue
                
            if isinstance(value, list):
                st.write(f"**{key.capitalize()}**:")
                for item in value:
                    if isinstance(item, dict):
                        for sub_key, sub_value in item.items():
                            st.write(f"- **{sub_key.capitalize()}**: {sub_value}")
                    else:
                        st.write(f"- {item}")
            elif isinstance(value, dict):
                st.write(f"**{key.capitalize()}**:")
                for sub_key, sub_value in value.items():
                    st.write(f"- **{sub_key.capitalize()}**: {sub_value}")
            else:
                st.write(f"**{key.capitalize()}**: {value}")

def render_file_grid(api_client: APIClient) -> None:
    """Render the file grid display section."""
    st.header("View Uploaded Files")
    
    try:
        file_list = api_client.get_files()
    except ValueError as e:
        st.error(f"Failed to fetch file list: {str(e)}")
        return
        
    if not file_list:
        st.info("No files uploaded yet.")
        return
        
    # Create a 2-column grid layout for files
    for i in range(0, len(file_list), 2):
        cols = st.columns(2)
        for j, filename in enumerate(file_list[i:i+2]):
            with cols[j]:
                st.write(filename)
                
                try:
                    file_content = api_client.get_file(filename)
                    if filename.lower().endswith((".mp4", ".avi", ".mov")):
                        st.video(file_content)
                    elif filename.lower().endswith((".jpg", ".png")):
                        st.image(file_content)
                except ValueError as e:
                    st.error(f"Failed to load file: {str(e)}")
                
                # Create action columns for delete and metadata
                action_cols = st.columns([1, 5])
                
                with action_cols[0]:
                    if st.button("🗑️", key=f"delete-{filename}"):
                        try:
                            api_client.delete_file(filename)
                            st.success(f"Deleted {filename} successfully!")
                            st.experimental_rerun()
                        except ValueError as e:
                            st.error(f"Failed to delete file: {str(e)}")
                
                with action_cols[1]:
                    try:
                        metadata = api_client.get_metadata(filename)
                        render_metadata(metadata)
                    except ValueError as e:
                        st.error(f"Failed to fetch metadata: {str(e)}") 