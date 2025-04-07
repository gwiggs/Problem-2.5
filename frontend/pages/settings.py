import streamlit as st
from utils.api_client import APIClient
from models.llm_prompt import LLMPrompt
from typing import List, Dict

def render_prompt_editor(prompt: LLMPrompt, api_client: APIClient, index: int) -> None:
    """Render editor for a single prompt."""
    # Create a unique key for this prompt's name input
    name_key = f"name_{index}"
    
    # Get the current name from session state or use the prompt's name
    current_name = st.session_state.get(name_key, prompt.name)
    
    # Use the current name in the expander title (without key parameter)
    with st.expander(f"📝 {current_name}", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Update the name input to use the current name
            new_name = st.text_input("Name", value=current_name, key=name_key)
            new_description = st.text_area("Description", value=prompt.description, key=f"desc_{index}")
            new_template = st.text_area("Template", value=prompt.template, height=200, key=f"template_{index}")
            
        with col2:
            # Check if is_default exists, otherwise default to False
            is_default = getattr(prompt, "is_default", False)
            new_is_default = st.checkbox("Default Prompt", value=is_default, key=f"default_{index}")
            
            # Check if version exists, otherwise default to empty string
            version = getattr(prompt, "version", "")
            new_version = st.text_input("Version", value=version, key=f"version_{index}")
            
            if st.button("💾 Save Changes", key=f"save_{index}"):
                try:
                    updated_prompt = LLMPrompt(
                        name=new_name,
                        description=new_description,
                        template=new_template,
                        is_default=new_is_default,
                        version=new_version or None
                    )
                    api_client.update_prompt(updated_prompt)
                    st.success("✅ Changes saved successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to save changes: {str(e)}")
                    
            if st.button("🗑️ Delete Prompt", key=f"delete_{index}"):
                try:
                    api_client.delete_prompt(prompt.name)
                    st.success("✅ Prompt deleted successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to delete prompt: {str(e)}")

def render_new_prompt_form(api_client: APIClient) -> None:
    """Render form for adding a new prompt."""
    with st.expander("➕ Add New Prompt", expanded=True):
        # Initialize session state for form fields if they don't exist
        if "new_name" not in st.session_state:
            st.session_state.new_name = ""
        if "new_desc" not in st.session_state:
            st.session_state.new_desc = ""
        if "new_template" not in st.session_state:
            st.session_state.new_template = ""
        if "new_default" not in st.session_state:
            st.session_state.new_default = False
        if "new_version" not in st.session_state:
            st.session_state.new_version = ""
        
        new_name = st.text_input("Name", value=st.session_state.new_name, key="new_name")
        new_description = st.text_area("Description", value=st.session_state.new_desc, key="new_desc")
        new_template = st.text_area("Template", value=st.session_state.new_template, height=200, key="new_template")
        new_is_default = st.checkbox("Default Prompt", value=st.session_state.new_default, key="new_default")
        new_version = st.text_input("Version", value=st.session_state.new_version, key="new_version")
        
        if st.button("➕ Add Prompt"):
            if not new_name or not new_description or not new_template:
                st.error("❌ Please fill in all required fields")
                return
                
            try:
                new_prompt = LLMPrompt(
                    name=new_name,
                    description=new_description,
                    template=new_template,
                    is_default=new_is_default,
                    version=new_version or None
                )
                api_client.add_prompt(new_prompt)
                
                # Clear form fields after successful addition
                st.session_state.new_name = ""
                st.session_state.new_desc = ""
                st.session_state.new_template = ""
                st.session_state.new_default = False
                st.session_state.new_version = ""
                
                st.success("✅ Prompt added successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to add prompt: {str(e)}")

def render_settings_page(api_client: APIClient) -> None:
    """Render the settings page."""
    # Add a back button at the top
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back", key="back_button"):
            # Return to the previous page
            st.session_state.current_page = st.session_state.previous_page
            st.rerun()
    
    with col2:
        st.title("⚙️ Settings")
    
    # Load current prompts
    try:
        prompts = api_client.get_llm_prompts()
    except Exception as e:
        st.error(f"❌ Failed to load prompts: {str(e)}")
        prompts = []
    
    # Render prompt management section
    st.header("📝 Prompt Management")
    
    # Show existing prompts
    for i, prompt in enumerate(prompts):
        render_prompt_editor(prompt, api_client, i)
    
    # Show new prompt form
    render_new_prompt_form(api_client)

if __name__ == "__main__":
    # For standalone testing
    from utils.api_client import APIClient
    api_client = APIClient("http://backend:8000", "http://llm:8100")
    render_settings_page(api_client) 