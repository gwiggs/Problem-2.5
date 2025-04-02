from classes.frontend_class_example import HelloWorld
import streamlit as st
import requests
import os

UPLOAD_DIR = "uploaded_videos"

def main():
    st.title("Batch video upload/view/review")

    # Video upload section
    st.header("Upload a Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Send the file to the backend
        files = {"file": (uploaded_filse.name, uploaded_file, uploaded_file.type)}
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
        else:
            st.info("No videos uploaded yet.")
    else:
        st.error("Failed to fetch video list.")

if __name__ == "__main__":
    main()
