from ultralytics import YOLO

import torch
import openvino as ov

import collections
import time
from IPython import display
import cv2
import numpy as np


# Main processing function to run object detection.
def run_object_detection(
        source=0,
        flip=False,
        use_popup=True,
        skip_first_frames=0,
        model=None,
        device="CPU"
):
    player = None
    ov_config = {}
    if device != "CPU":
        model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    compiled_model = core.compile_model(model, device, ov_config)

    # Get model inputs
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    N, C, H, W = input_layer.shape

    try:
        # Use OpenCV VideoCapture
        player = cv2.VideoCapture("Prueba.mp4")
        if not player.isOpened():
            print("Error: Unable to open video source")
            return

        # Skip first frames if needed
        for _ in range(skip_first_frames):
            player.grab()

        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()
        while True:
            # Grab the frame
            ret, frame = player.read()
            if not ret:
                print("Source ended")
                break

            if flip:
                frame = cv2.flip(frame, 1)

            # Resize and preprocess the frame
            resized_frame = cv2.resize(frame, (W, H))
            input_image = np.expand_dims(resized_frame.transpose(2, 0, 1), 0)

            # Perform inference
            start_time = time.time()
            results = compiled_model([input_image])[output_layer]
            stop_time = time.time()

            # Process the results (assuming YOLO v8 output format)
            print(results)
            for detection in results[0]:
                # print(detection)
                score = detection[4]
                if score > 0.5:  # Confidence threshold
                    xmin, ymin, xmax, ymax = detection[:4]
                    xmin = int(xmin * frame.shape[1])
                    xmax = int(xmax * frame.shape[1])
                    ymin = int(ymin * frame.shape[0])
                    ymax = int(ymax * frame.shape[0])
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person: {score:.2f}", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            # Calculate and display FPS
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(frame, f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                        (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Person Detection", frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break

    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if player is not None:
            player.release()
        if use_popup:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    core = ov.Core()

    det_ov_model = core.read_model("yolov8n_openvino_model/yolov8n.xml")
    run_object_detection(
        source=0,
        flip=True,
        use_popup=True,
        model=det_ov_model,
        device="CPU",
    )
