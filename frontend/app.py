from classes.frontend_class_example import HelloWorld
import streamlit as st
import requests
import os

UPLOAD_DIR = "uploaded_videos"

def main():
    st.title("Streamlit and FastAPI Integration")

    # Video upload section
    st.header("Upload a Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Send the file to the backend
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post("http://backend:8000/upload/", files=files)
        if response.status_code == 200:
            st.success(f"Upload successful: {response.json()['message']}")
        else:
            st.error(f"Upload failed: {response.text}")

    # Video display section
    st.header("View Uploaded Videos")
    response = requests.get("http://backend:8000/videos/")
    if response.status_code == 200:
        video_list = response.json().get("videos", [])
        if video_list:
            selected_video = st.selectbox("Select a video to view:", video_list)
            if st.button("Play Selected Video"):
                video_url = f"http://backend:8000/video/{selected_video}"
                st.video(video_url)
        else:
            st.info("No videos uploaded yet.")
    else:
        st.error("Failed to fetch video list.")

if __name__ == "__main__":
    main()
