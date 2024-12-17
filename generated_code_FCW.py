global collision_warning_triggered
def detect_objects(frame):
    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    frame = cv2.resize(frame, (1420,780))  
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    height, width, _ = frame.shape
    objects = []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = int(np.argmax(scores))
            confidence = scores[class_id]
            if confidence > 0.5:  
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                distance = max(1, 50 - int(confidence * 50))  

                objects.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "distance": distance 
                })

                label = f"Object {class_id} ({confidence:.2f}) Dist: {distance}m"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, .25, (0, 255, 0), 2)

    return objects, frame


def collision_warning(objects, vehicle_speed):
    collision_warning_triggered= False

    for obj in objects:
        if 'distance' in obj and obj['distance'] < 5 and vehicle_speed > 10:
            if not collision_warning_triggered:
                print("Collision Warning Triggered!")
                collision_warning_triggered = True
            return True 

    if collision_warning_triggered:
        print("Collision Warning Cleared!")
        collision_warning_triggered = False

    return False

import unittest
class TestFCWSystem(unittest.TestCase):
    def test_object_detection(self):
        frame = cv2.imread("test_image.jpg")
        objects = detect_objects(frame)
        self.assertGreater(len(objects), 0, "No objects detected!")

    def test_collision_warning(self):
        objects = [{"distance": 4}]
        result = collision_warning(objects, vehicle_speed=15)
        self.assertTrue(result, "Collision warning not triggered!")

if __name__ == "//main//":
    unittest.main()

import os

def run_static_analysis():
    os.system("cppcheck --enable=all --std=c++17 fcw_generated_code.py")
    print("Static analysis completed.")


import tkinter as tk
from tkinter import Label
def display_warning():
    root = tk.Tk()
    root.title("FCW Warning")
    Label(root, text="Collision Warning Triggered!", fg="red", font=("Helvetica", 24)).pack()
    root.after(2000, root.destroy)
    root.update_idletasks()
    root.update()


import cv2
import numpy as np
cap = cv2.VideoCapture("C://Users//arkya//OneDrive//Documents//Desktop//ADAS//test_video3.mp4")
if not cap.isOpened():
    print("Error: Video file could not be opened.")
    exit()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("not working")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    objects, annotated_frame = detect_objects(frame)

    cv2.imshow("Object Detection", annotated_frame)

    collision_detected = collision_warning(objects, vehicle_speed=20)
    if collision_detected:
        display_warning()

cap.release()
cv2.destroyAllWindows()

unittest.main(exit=False)

run_static_analysis()
