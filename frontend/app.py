# from classes.frontend_class_example import HelloWorld
import streamlit as st
import requests

UPLOAD_DIR = "uploaded_files"
BASE_URL = "http://backend:8000"

def main():
    st.title("Batch Files")

    # File upload section
    st.header("Upload Files")
    uploaded_files = st.file_uploader("Choose files (videos or images)", type=["mp4", "avi", "mov", "jpg", "png"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Send each file to the backend
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post(f"{BASE_URL}/upload/", files=files)
            if response.status_code == 200:
                st.success(f"Upload successful: {response.json()['message']} for {uploaded_file.name}")
            else:
                st.error(f"Upload failed for {uploaded_file.name}: {response.text}")

    # Initialize refresh state
    if "refresh" not in st.session_state:
        st.session_state.refresh = False

    # File display section
    st.header("View Uploaded Files")
    if st.session_state.refresh:
        response = requests.get(f"{BASE_URL}/files/")
        st.session_state.refresh = False  # Reset refresh state
    else:
        response = requests.get(f"{BASE_URL}/files/")

    if response.status_code == 200:
        file_list = response.json().get("files", [])
        if file_list:
            # Create a 2-column grid layout for files
            for i in range(0, len(file_list), 2):
                cols = st.columns(2)  # Create 2 columns
                for j, file in enumerate(file_list[i:i+2]):  # Slice 2 files at a time
                    with cols[j]:
                        st.write(file)  # Display the file name
                        file_url = f"{BASE_URL}/file/{file}"
                        response = requests.get(file_url)
                        if response.status_code == 200:
                            if file.lower().endswith((".mp4", ".avi", ".mov")):
                                st.video(response.content)
                            elif file.lower().endswith((".jpg", ".png")):
                                st.image(response.content)
                        else:
                            st.error(f"Failed to load file: {file}")

                        # Create a horizontal layout for the delete button and metadata dropdown
                        action_cols = st.columns([1, 5])  # Adjust column widths as needed
                        with action_cols[0]:  # Column for the delete button
                            if st.button("🗑️", key=f"delete-{file}"):
                                delete_response = requests.delete(f"{BASE_URL}/file/{file}")
                                if delete_response.status_code == 200:
                                    st.success(f"Deleted {file} successfully!")
                                    st.session_state.refresh = True  # Trigger a refresh
                                else:
                                    st.error(f"Failed to delete {file}: {delete_response.text}")

                        with action_cols[1]:  # Column for the metadata dropdown
                            metadata_response = requests.get(f"{BASE_URL}/metadata/{file}")
                            if metadata_response.status_code == 200:
                                metadata = metadata_response.json()
                                with st.expander("Metadata Details"):
                                    for key, value in metadata.items():
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
                            else:
                                st.error(f"Failed to fetch metadata for {file}: {metadata_response.text}")
        else:
            st.info("No files uploaded yet.")
    else:
        st.error("Failed to fetch file list.")

if __name__ == "__main__":
    main()
