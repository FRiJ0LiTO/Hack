import streamlit as st
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import time
import openvino as ov

# Function to process video frames
def process_video(compiled_model, video_source):
    if isinstance(video_source, int):
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(video_source)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    track_history = defaultdict(lambda: [])
    df = pd.DataFrame(columns=['x', 'y'])
    processing_times = deque()

    # Streamlit placeholders
    video_placeholder = st.empty()
    heatmap_placeholder = st.empty()
    
    # Get model input and output layers
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    N, C, H, W = input_layer.shape

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and preprocess the frame
        resized_frame = cv2.resize(frame, (W, H))
        input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

        # Perform inference
        start_time = time.time()
        results = compiled_model([input_image])[output_layer]
        stop_time = time.time()

        processing_times.append(stop_time - start_time)
        if len(processing_times) > 200:
            processing_times.popleft()

        # Process the results
        annotated_frame = frame.copy()
        for detection in results[0]:
            score = detection[4]
            if score > 0.5:  # Confidence threshold
                xmin, ymin, xmax, ymax = detection[:4]
                xmin = int(xmin * frame.shape[1])
                xmax = int(xmax * frame.shape[1])
                ymin = int(ymin * frame.shape[0])
                ymax = int(ymax * frame.shape[0])
                
                cv2.rectangle(annotated_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Person: {score:.2f}", (xmin, ymin - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Update tracking and heatmap data
                x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
                df = pd.concat([df, pd.DataFrame({'x': [float(x)], 'y': [float(y)]})], ignore_index=True)

        # Calculate and display FPS
        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time
        cv2.putText(annotated_frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

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

    cap.release()

# Streamlit app
def main():
    st.title("OpenVINO YOLO Person Detection with Real-time Heatmap")

    # Load the OpenVINO model
    @st.cache_resource
    def load_model():
        core = ov.Core()
        model = core.read_model("yolov8n_openvino_model/yolov8n.xml")
        compiled_model = core.compile_model(model, "CPU")
        return compiled_model

    compiled_model = load_model()

    # Choose between webcam and video upload
    source_option = st.radio("Select input source:", ("Webcam", "Upload Video"))

    if source_option == "Webcam":
        if st.button("Start Webcam"):
            process_video(compiled_model, 0)  # 0 is typically the default webcam
    else:
        # File uploader for video
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if video_file is not None:
            # Save uploaded file temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.read())
            # Process the video
            process_video(compiled_model, "temp_video.mp4")
        else:
            st.write("Please upload a video file.")

if __name__ == "__main__":
    main()