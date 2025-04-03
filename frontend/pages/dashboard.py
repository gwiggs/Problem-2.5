import streamlit as st
from utils.api_client import APIClient

def render_dashboard(api_client: APIClient):
    """Render the dashboard page."""
    st.header("Dashboard")
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    try:
        files = api_client.get_files()
        
        with col1:
            st.metric("Total Files", len(files))
        
        with col2:
            video_count = sum(1 for f in files if f.lower().endswith((".mp4", ".avi", ".mov")))
            st.metric("Video Files", video_count)
        
        with col3:
            image_count = sum(1 for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")))
            st.metric("Image Files", image_count)
            
        # Recent uploads
        st.subheader("Recent Uploads")
        if files:
            recent_files = files[-5:]  # Show last 5 files
            for file in recent_files:
                st.write(f"📄 {file}")
        else:
            st.info("No files uploaded yet.")
            
    except Exception as e:
        st.error(f"Failed to load dashboard data: {str(e)}") 