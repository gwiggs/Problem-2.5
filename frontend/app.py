import streamlit as st
import requests

UPLOAD_DIR = "uploaded_videos"

def main():
    st.title("Batch videos")

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

    # Delete video section
    st.header("Delete a Video")

    # Fetch the list of videos from the backend
    response = requests.get("http://backend:8000/videos/")
    if response.status_code == 200:
        video_list = response.json().get("videos", [])
        if video_list:
            # Use a dropdown to select a video to delete
            video_to_delete = st.selectbox("Select a video to delete:", video_list, key="video_to_delete")
            if st.button("Delete Selected Video"):
                try:
                    # Send DELETE request to the backend
                    delete_response = requests.delete(f"http://backend:8000/video/{video_to_delete}")
                    if delete_response.status_code == 200:
                        st.success(delete_response.json().get("message", "Video deleted successfully!"))
                        # Clear the session state to simulate a refresh
                        del st.session_state["video_to_delete"]
                    elif delete_response.status_code == 404:
                        st.error("Video not found.")
                    else:
                        st.error("An error occurred while deleting the video.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        else:
            st.info("No videos available to delete.")
    else:
        st.error("Failed to fetch video list.")

if __name__ == "__main__":
    main()
