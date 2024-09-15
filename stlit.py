import streamlit as st
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# Function to process video frames
def process_video(model, video_source):
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(video_source)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    track_history = defaultdict(lambda: [])
    df = pd.DataFrame(columns=['x', 'y'])

    # Streamlit placeholders
    video_placeholder = st.empty()
    heatmap_placeholder = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, classes=[0])  # 0 is the class index for person
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Create a copy of the frame to draw on
            annotated_frame = frame.copy()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (0, 255, 0), 2)

                # Draw tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                df = pd.concat([df, pd.DataFrame({'x': [float(x)], 'y': [float(y)]})], ignore_index=True)

            # Update video frame
            video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

            # Update heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(data=df, x="x", y="y", cmap="YlOrRd", fill=True, cbar=True, ax=ax)
            ax.set_xlim(0, frame_width)
            ax.set_ylim(frame_height, 0)
            ax.set_title("Real-time Heatmap of Person Detections")
            heatmap_placeholder.pyplot(fig)
            plt.close(fig)
        else:
            video_placeholder.image(frame, channels="BGR", use_column_width=True)

    cap.release()

# Streamlit app
def main():
    st.title("YOLOv8 Person Tracking with Real-time Heatmap")

    # Load the YOLOv8 model
    @st.cache_resource
    def load_model():
        return YOLO("yolov8n.pt")

    model = load_model()

    # Choose between webcam and video upload
    source_option = st.radio("Select input source:", ("Webcam", "Upload Video"))

    if source_option == "Webcam":
        if st.button("Start Webcam"):
            process_video(model, 0)  # 0 is typically the default webcam
    else:
        # File uploader for video
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if video_file is not None:
            # Save uploaded file temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.read())
            # Process the video
            process_video(model, "temp_video.mp4")
        else:
            st.write("Please upload a video file.")

if __name__ == "__main__":
    main()