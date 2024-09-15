from ultralytics import YOLO

import torch
import openvino as ov

import collections
import time
import cv2
import numpy as np

det_model = YOLO("yolov8n.pt")
det_model.classes = [0]
det_model.export(format="openvino", dynamic=True, half=True)


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
    if "GPU" in device or (
            "AUTO" in device and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    compiled_model = core.compile_model(model, device, ov_config)

    def infer(*args):
        result = compiled_model(args)
        return torch.from_numpy(result[0])

    # det_model.predictor.inference = infer

    try:
        # Use OpenCV VideoCapture instead of VideoPlayer
        player = cv2.VideoCapture(source)
        if not player.isOpened():
            print("Error: Unable to open video source")
            return

        # Skip first frames if needed
        for _ in range(skip_first_frames):
            player.grab()

        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(winname=title,
                            flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()
        while True:
            # Grab the frame
            ret, frame = player.read()
            if not ret:
                print("Source ended")
                break

            if flip:
                frame = cv2.flip(frame, 1)

            # Resize if necessary
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(src=frame, dsize=None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)

            # Get the results
            input_image = np.array(frame)

            start_time = time.time()
            detections = det_model(input_image, verbose=False)
            stop_time = time.time()
            frame = detections[0].plot()

            processing_times.append(stop_time - start_time)
            if len(processing_times) > 200:
                processing_times.popleft()

            # Calculate FPS and processing time
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            cv2.putText(
                img=frame,
                text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                org=(20, 40),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

            cv2.imshow(winname="YOLOv8 Tracking", mat=frame)
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


# Load a YOLOv8n PyTorch model
model = YOLO("yolov8n.pt")
model.classes = [0]

# Export the model
model.export(format="openvino")  # creates 'yolov8n_openvino_model/'


if __name__ == '__main__':
    # Load a YOLOv8n PyTorch model
    model = YOLO("yolov8x.pt")

    # Export the model
    model.export(format="openvino")

    # Load the exported OpenVINO model
    ov_model = YOLO("yolov8x_openvino_model/")
