import cv2
import numpy as np
from ultralytics import YOLO
import threading

# Load YOLOv8 model for vehicle detection
model = YOLO("yolov8n.pt")  # Replace with your own trained YOLO model

# Function to estimate distance based on bounding box size
def estimate_distance(x1, y1, x2, y2, confidence, calibration_factor=0.01):
    """
    Estimate the distance based on the bounding box size.
    Larger bounding box -> closer distance.
    """
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    distance = max(1, 50 - int(w * h * calibration_factor * (1 + (1 - confidence))))
    return distance

# Function to check if the object is in the same lane
def is_in_same_lane(frame_width, x1, x2, lane_left_bound, lane_right_bound):
    """
    Check if the object's bounding box is within the lane bounds (left and right of the lane).
    """
    bbox_center = (x1 + x2) // 2
    return lane_left_bound <= bbox_center <= lane_right_bound

# Optimized lane detection using simple edge detection
def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (100, height),
        (width - 100, height),
        (width // 2, height // 2)
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=50)
    
    lane_left_bound, lane_right_bound = None, None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < width // 2:
                lane_left_bound = max(lane_left_bound, x1) if lane_left_bound else x1
            else:
                lane_right_bound = min(lane_right_bound, x1) if lane_right_bound else x1

    return lane_left_bound, lane_right_bound

# Main function to process the video and trigger warnings
def forward_collision_warning(video_path, distance_threshold=30):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video file could not be opened.")
        return

    print("Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        frame = cv2.resize(frame, (1420, 780))  # Resize for faster processing

        # Detect lanes in a separate thread
        lane_thread = threading.Thread(target=detect_lanes, args=(frame,))
        lane_thread.start()

        # Object detection using YOLOv8
        results = model(frame)
        annotated_frame = frame.copy()  # Create a copy to draw on
        lane_thread.join()  # Wait for lane detection to finish

        collision_warning_triggered = False
        lane_left_bound, lane_right_bound = detect_lanes(frame)  # Get lane bounds after detection

        # Process detection results
        for r in results:
            for obj in r.boxes.data:
                x1, y1, x2, y2, confidence, class_id = obj[:6]
                class_id = int(class_id)

                # Check if the object is a "car" (COCO class ID for car: 2)
                if class_id == 2:
                    # Estimate the distance of the object
                    distance = estimate_distance(x1, y1, x2, y2, confidence)

                    # Check if the object is within the same lane
                    if is_in_same_lane(frame_width, x1, x2, lane_left_bound, lane_right_bound):
                        # Draw bounding box and display distance
                        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f"Car ({distance}m)"
                        cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Check for collision warning
                        if distance < distance_threshold:
                            collision_warning_triggered = True

        # Display collision warning on the frame
        if collision_warning_triggered:
            cv2.putText(annotated_frame, "Collision Warning!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the annotated frame
        cv2.imshow("Forward Collision Warning", annotated_frame)

        # Quit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video resources
    cap.release()
    cv2.destroyAllWindows()

# Run the model with your test video
video_path = "/Users/piyushpal/Downloads/yolov8/test_video3.mp4"  # Replace with the correct path
forward_collision_warning(video_path, distance_threshold=30)
