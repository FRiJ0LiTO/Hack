import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "Prueba.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Store the track history
track_history = defaultdict(lambda: [])

# Create a DataFrame to store detection coordinates
df = pd.DataFrame(columns=['x', 'y'])

# Create a figure for the heatmap
plt.figure(figsize=(10, 6))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        
        if results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()
            
            # Plot the tracks and update DataFrame
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                print(track_id, x, y, w, h)
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)
                
                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                
                # Add the current position to the DataFrame
                df = pd.concat([df, pd.DataFrame({'x': [float(x)], 'y': [float(y)]})], ignore_index=True)
            
            # Create and update the heatmap
            plt.clf()  # Clear the current figure
            sns.kdeplot(data=df, x="x", y="y", cmap="YlOrRd", shade=True, cbar=True)
            plt.xlim(0, frame_width)
            plt.ylim(frame_height, 0)  # Invert y-axis to match image coordinates
            plt.title("Real-time Heatmap of Detections")
            plt.draw()
            plt.pause(0.001)  # Small pause to update the plot
            
            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
        else:
            cv2.imshow("YOLOv8 Tracking", frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display windows
cap.release()
cv2.destroyAllWindows()
plt.close()