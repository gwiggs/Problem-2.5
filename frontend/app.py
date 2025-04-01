# frontend/app.py
import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "http://backend:8000"

st.set_page_config(page_title="JSON Data Explorer", layout="wide")
st.title("🔍 JSON Data Explorer with Advanced Search")

# File Upload Section
with st.sidebar:
    uploaded_file = st.file_uploader("Upload JSON File", type=["json"])
    if uploaded_file:
        try:
            response = requests.post(
                f"{BACKEND_URL}/upload-json/",
                files={"file": uploaded_file}
            )
            if response.status_code == 200:
                st.success("File uploaded successfully!")
                st.session_state.uploaded = True
            else:
                st.error("Upload failed")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

# Search and Display Section
if st.session_state.get('uploaded'):
    col1, col2 = st.columns([3, 1])
    with col2:
        with st.expander("Search Help"):
            st.markdown("""
            **Search Syntax:**
            - `AND`: Both terms must match
            - `OR`: Either term matches
            - `NOT`: Exclude term
            - Combine with parentheses for complex queries
            
            **Regex Examples:**
            - `error\d+`: Match error with numbers
            - `^2023`: Start with 2023
            - `active|pending`: Match active or pending
            """)
    
    with col1:
        search_query = st.text_input(
            "Search (use AND/OR with regex)",
            placeholder="Example: (error AND 5\\d{2}) OR warning"
        )

    try:
        response = requests.get(
            f"{BACKEND_URL}/search/",
            params={"query": search_query}
        )
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            
            # Display dataframe with enhanced styling
            if not df.empty:
                st.data_editor(
                    df,
                    use_container_width=True,
                    height=600,
                    column_config={
                        col: {"width": "medium"} for col in df.columns
                    },
                    num_rows="fixed",
                    hide_index=True
                )
                
                # Summary stats
                st.caption(f"Showing {len(df)} records | {len(df.columns)} columns | ")
            else:
                st.info("No matching records found")

    except requests.ConnectionError:
        st.error("Failed to connect to backend")

# Always show data even without search
elif st.session_state.get('uploaded'):
    try:
        response = requests.get(f"{BACKEND_URL}/get-data/")
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
    except requests.ConnectionError:
        st.error("Failed to connect to backend")
