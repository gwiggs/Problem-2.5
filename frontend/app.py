# from classes.frontend_class_example import HelloWorld
import streamlit as st
import requests

UPLOAD_DIR = "uploaded_videos"

def main():
    st.title("Batch videos")

    # Video upload section
    st.header("Upload Videos")
    uploaded_files = st.file_uploader("Choose video files", type=["mp4", "avi", "mov"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Send each file to the backend
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            response = requests.post("http://backend:8000/upload/", files=files)
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
        response = requests.get("http://backend:8000/videos/")
        st.session_state.refresh = False  # Reset refresh state
    else:
        response = requests.get("http://backend:8000/videos/")

    if response.status_code == 200:
        video_list = response.json().get("videos", [])
        if video_list:
            # Create a 2-column grid layout for videos
            for i in range(0, len(video_list), 2):
                cols = st.columns(2)  # Create 2 columns
                for j, video in enumerate(video_list[i:i+2]):  # Slice 2 videos at a time
                    with cols[j]:
                        st.write(video)  # Display the video name
                        video_url = f"http://backend:8000/video/{video}"
                        response = requests.get(video_url)
                        if response.status_code == 200:
                            st.video(response.content)
                        else:
                            st.error(f"Failed to load video: {video}")

                        # Add a delete button
                        if st.button("🗑️ Delete", key=f"delete-{video}"):
                            delete_response = requests.delete(f"http://backend:8000/video/{video}")
                            if delete_response.status_code == 200:
                                st.success(f"Deleted {video} successfully!")
                                st.session_state.refresh = True  # Trigger a refresh
                            else:
                                st.error(f"Failed to delete {video}: {delete_response.text}")
        else:
            st.info("No videos uploaded yet.")
    else:
        st.error("Failed to fetch video list.")

if __name__ == "__main__":
    main()
