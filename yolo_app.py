def run_yolo(frame):
    print("🎯 YOLOv8 Frame Processing...")
    return [("car", 0.92, [100, 150, 300, 400])]

def main():
    detections = run_yolo(None)
    print(f"✓ Detections Count: {len(detections)}")

if __name__ == "__main__":
    main()
