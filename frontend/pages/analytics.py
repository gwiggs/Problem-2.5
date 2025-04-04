import streamlit as st
from utils.api_client import APIClient
from typing import List, Tuple, Optional
from models.file_formats import is_video_format, is_image_format
from models.schemas import LLMPrompt, LLMResponse

def get_available_files(api_client: APIClient) -> Tuple[List[str], List[str], List[str]]:
    """Get available files from the API and categorize them."""
    try:
        files = api_client.get_files()
        video_files = [f for f in files if is_video_format(f)]
        image_files = [f for f in files if is_image_format(f)]
        return files, video_files, image_files
    except ValueError as e:
        st.error(f"Failed to fetch files: {str(e)}")
        return [], [], []

def get_available_prompts(api_client: APIClient) -> List[LLMPrompt]:
    """Get available prompts from the API."""
    try:
        return api_client.get_llm_prompts()
    except ValueError as e:
        st.error(f"Failed to fetch prompts: {str(e)}")
        return []

def render_file_selection(video_files: List[str], image_files: List[str]) -> List[str]:
    """Render the file selection UI and return selected files."""
    selected_files = st.multiselect(
        "Select videos and images to analyze",
        options=video_files + image_files,
        help="You can select multiple videos and images for analysis"
    )
    
    if not video_files and not image_files:
        st.info("No files available for analysis. Please upload some files first.")
    elif not selected_files:
        st.info("Please select one or more files to analyze.")
        
    return selected_files

def render_prompt_selection(prompts: List[LLMPrompt]) -> Optional[str]:
    """Render the prompt selection UI and return the selected prompt."""
    if not prompts:
        st.warning("No analysis prompts available.")
        return None
        
    selected_prompt = st.selectbox(
        "Select analysis type",
        options=[p.name for p in prompts],
        format_func=lambda x: next((p.description for p in prompts if p.name == x), x),
        help="Choose the type of analysis to perform"
    )
    
    return selected_prompt

def render_analysis_button(api_client: APIClient, selected_files: List[str], selected_prompt: str) -> None:
    """Render the analysis button and handle the analysis process."""
    if not selected_files:
        st.button("Analyze Files", disabled=True, help="Please select at least one file to analyze")
        return
        
    if st.button("Analyze Files"):
        try:
            with st.spinner("Analyzing files..."):
                result = api_client.analyze_videos(selected_files, selected_prompt)
                
            # Store the analysis result in session state
            st.session_state.analysis_result = result
            st.session_state.analyzed_files = selected_files
            
            # Force a rerun to display the results
            st.rerun()
                
        except ValueError as e:
            st.error(f"Analysis failed: {str(e)}")

def render_analytics_page(api_client: APIClient):
    """Render the analytics page."""
    st.header("Analytics")
    
    # Get available files and prompts
    files, video_files, image_files = get_available_files(api_client)
    prompts = get_available_prompts(api_client)
    
    # Create two columns for file selection and prompt selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("File Selection")
        selected_files = render_file_selection(video_files, image_files)
    
    with col2:
        st.subheader("Analysis Type")
        selected_prompt = render_prompt_selection(prompts)
        
        if selected_prompt:
            render_analysis_button(api_client, selected_files, selected_prompt)
        else:
            st.info("Please select an analysis type.")
            
    # Add a divider between selections and results
    st.markdown("---")
    
    # Create a container for analysis results that will be updated
    results_container = st.container()
    
    # Store the analysis state in session state
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
        st.session_state.analyzed_files = []
        
    # Display analysis results in full width if available
    if st.session_state.analysis_result:
        with results_container:
            st.subheader("Analysis Results")
            st.write(st.session_state.analysis_result.analysis)
            
            # Show metadata for reference
            with st.expander("File Details"):
                for filename in st.session_state.analyzed_files:
                    st.write(f"**{filename}**")
                    try:
                        metadata = api_client.get_metadata(filename)
                        st.write(f"Duration: {metadata.duration}s")
                        st.write(f"Resolution: {metadata.width}x{metadata.height}")
                    except ValueError as e:
                        st.error(f"Failed to fetch metadata for {filename}: {str(e)}")    