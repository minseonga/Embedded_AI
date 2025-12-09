
import cv2
import time

def test_pipeline(name, pipeline):
    print(f"\nTesting Pipeline: {name}")
    print(f"String: {pipeline}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"[PASS] Success! Frame shape: {frame.shape}")
            cap.release()
            return True
        else:
            print("[FAIL] Opened but no frame.")
    else:
        print("[FAIL] Could not open.")
    return False

def main():
    pipelines = [
        (
            "Original (Mode 2, 640x480)",
            "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        ),
        (
            "No Sensor Mode (Auto)",
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        ),
        (
            "720p (Standard)",
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        ),
        (
            "Minimal (No caps on source)",
            "nvarguscamerasrc sensor-id=0 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink drop=1"
        )
    ]

    for name, p in pipelines:
        if test_pipeline(name, p):
            print(f"\n>>> RECOMMENDED: Use '{name}' pipeline.")
            break

if __name__ == "__main__":
    main()
