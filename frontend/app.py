# from classes.frontend_class_example import HelloWorld
import streamlit as st
import requests

UPLOAD_DIR = "uploaded_videos"
BASE_URL = "http://backend:8000"

def main():
    st.title("Batch videos")

    # Video upload section
    st.header("Upload Videos")
    uploaded_files = st.file_uploader("Choose video files", type=["mp4", "avi", "mov"], accept_multiple_files=True)
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

    # Video display section
    st.header("View Uploaded Videos")
    if st.session_state.refresh:
        response = requests.get(f"{BASE_URL}/videos/")
        st.session_state.refresh = False  # Reset refresh state
    else:
        response = requests.get(f"{BASE_URL}/videos/")

    if response.status_code == 200:
        video_list = response.json().get("videos", [])
        if video_list:
            # Create a 2-column grid layout for videos
            for i in range(0, len(video_list), 2):
                cols = st.columns(2)  # Create 2 columns
                for j, video in enumerate(video_list[i:i+2]):  # Slice 2 videos at a time
                    with cols[j]:
                        st.write(video)  # Display the video name
                        video_url = f"{BASE_URL}/video/{video}"
                        response = requests.get(video_url)
                        if response.status_code == 200:
                            st.video(response.content)
                        else:
                            st.error(f"Failed to load video: {video}")

                        # Create a horizontal layout for the delete button and metadata dropdown
                        action_cols = st.columns([1, 5])  # Adjust column widths as needed
                        with action_cols[0]:  # Column for the delete button
                            if st.button("🗑️", key=f"delete-{video}"):
                                delete_response = requests.delete(f"http://backend:8000/video/{video}")
                                if delete_response.status_code == 200:
                                    st.success(f"Deleted {video} successfully!")
                                    st.session_state.refresh = True  # Trigger a refresh
                                else:
                                    st.error(f"Failed to delete {video}: {delete_response.text}")

                        with action_cols[1]:  # Column for the metadata dropdown
                            metadata_response = requests.get(f"http://backend:8000/metadata/{video}")
                            if metadata_response.status_code == 200:
                                metadata = metadata_response.json()
                                with st.expander("Metadata Details"):
                                    # Format metadata for better presentation with no spacing and smaller font
                                    for key, value in metadata.items():
                                        if isinstance(value, list):
                                            st.markdown(f"<p style='font-size:0.8em; margin:0;'><strong>{key.capitalize()}</strong>:</p>", unsafe_allow_html=True)
                                            for item in value:
                                                if isinstance(item, dict):
                                                    for sub_key, sub_value in item.items():
                                                        st.markdown(f"<p style='font-size:0.8em; margin:0;'>- <strong>{sub_key.capitalize()}</strong>: {sub_value}</p>", unsafe_allow_html=True)
                                                else:
                                                    st.markdown(f"<p style='font-size:0.8em; margin:0;'>- {item}</p>", unsafe_allow_html=True)
                                        elif isinstance(value, dict):
                                            st.markdown(f"<p style='font-size:0.8em; margin:0;'><strong>{key.capitalize()}</strong>:</p>", unsafe_allow_html=True)
                                            for sub_key, sub_value in value.items():
                                                st.markdown(f"<p style='font-size:0.8em; margin:0;'>- <strong>{sub_key.capitalize()}</strong>: {sub_value}</p>", unsafe_allow_html=True)
                                        else:
                                            st.markdown(f"<p style='font-size:0.8em; margin:0;'><strong>{key.capitalize()}</strong>: {value}</p>", unsafe_allow_html=True)
                            else:
                                st.error(f"Failed to fetch metadata for {video}: {metadata_response.text}")
        else:
            st.info("No videos uploaded yet.")
    else:
        st.error("Failed to fetch video list.")

if __name__ == "__main__":
    main()
