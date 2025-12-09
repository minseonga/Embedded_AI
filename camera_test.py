
import cv2
import time

def test_gstreamer():
    print("Testing GStreamer (nvarguscamerasrc)...")
    gst_str = (
        "nvarguscamerasrc sensor-id=0 sensor-mode=2 ! "
        "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("[PASS] GStreamer camera opened!")
        ret, frame = cap.read()
        if ret:
            print(f"[PASS] Frame captured: {frame.shape}")
        else:
            print("[FAIL] Camera opened but no frame.")
        cap.release()
    else:
        print("[FAIL] Could not open GStreamer pipeline.")

def test_v4l2():
    print("\nTesting V4L2 (/dev/video0)...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[PASS] V4L2 camera opened!")
        ret, frame = cap.read()
        if ret:
            print(f"[PASS] Frame captured: {frame.shape}")
        else:
            print("[FAIL] Camera opened but no frame.")
        cap.release()
    else:
        print("[FAIL] Could not open /dev/video0.")

if __name__ == "__main__":
    test_gstreamer()
    test_v4l2()
