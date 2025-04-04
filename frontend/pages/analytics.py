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
    # Create two columns for the checkboxes
    col1, col2 = st.columns(2)
    
    # Add checkboxes to select all videos and all images
    with col1:
        select_all_videos = st.checkbox("Select all videos", value=False)
    with col2:
        select_all_images = st.checkbox("Select all images", value=False)
    
    # Determine default selection based on checkboxes
    default_selection = []
    if select_all_videos:
        default_selection.extend(video_files)
    if select_all_images:
        default_selection.extend(image_files)
    
    # Create separate sections for videos and images
    st.subheader("Videos")
    selected_videos = st.multiselect(
        "Select videos to analyze",
        options=video_files,
        default=video_files if select_all_videos else [],
        help="You can select multiple videos for analysis"
    )
    
    st.subheader("Images")
    selected_images = st.multiselect(
        "Select images to analyze",
        options=image_files,
        default=image_files if select_all_images else [],
        help="You can select multiple images for analysis"
    )
    
    # Combine selected videos and images
    selected_files = selected_videos + selected_images
    
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
    
    # Initialize session state for checkboxes if not already present
    if 'select_all_videos' not in st.session_state:
        st.session_state.select_all_videos = False
    if 'select_all_images' not in st.session_state:
        st.session_state.select_all_images = False
    
    # Get available files and prompts
    files, video_files, image_files = get_available_files(api_client)
    prompts = get_available_prompts(api_client)
    
    # Prompt selection at the top (full width)
    st.subheader("Analysis Type")
    selected_prompt = render_prompt_selection(prompts)
    
    # Add a divider between prompt selection and file selection
    st.markdown("---")
    
    # File selection in two columns
    st.subheader("File Selection")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Videos")
        # Checkbox for selecting all videos
        if video_files:
            select_all_videos = st.checkbox(
                "Select all videos", 
                key="select_all_videos_checkbox",
                value=st.session_state.select_all_videos
            )
            # Update session state only if the checkbox value changed
            if select_all_videos != st.session_state.select_all_videos:
                st.session_state.select_all_videos = select_all_videos
                st.rerun()
        
        # Use the session state value to determine default selection
        selected_videos = st.multiselect(
            "Select videos to analyze",
            options=video_files,
            default=video_files if st.session_state.select_all_videos else [],
            help="You can select multiple videos for analysis"
        )
    
    with col2:
        st.markdown("### Images")
        # Checkbox for selecting all images
        if image_files:
            select_all_images = st.checkbox(
                "Select all images", 
                key="select_all_images_checkbox",
                value=st.session_state.select_all_images
            )
            # Update session state only if the checkbox value changed
            if select_all_images != st.session_state.select_all_images:
                st.session_state.select_all_images = select_all_images
                st.rerun()
        
        # Use the session state value to determine default selection
        selected_images = st.multiselect(
            "Select images to analyze",
            options=image_files,
            default=image_files if st.session_state.select_all_images else [],
            help="You can select multiple images for analysis"
        )
    
    # Combine selected videos and images
    selected_files = selected_videos + selected_images
    
    # Show appropriate message based on file availability
    if not video_files and not image_files:
        st.info("No files available for analysis. Please upload some files first.")
    elif not selected_files:
        st.info("Please select one or more files to analyze.")
    
    # Analysis button below file selection
    if selected_prompt:
        render_analysis_button(api_client, selected_files, selected_prompt)
    else:
        st.info("Please select an analysis type.")
    
    # Add a divider between file selection and results
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