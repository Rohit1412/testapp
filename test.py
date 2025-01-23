import cv2
import time
import random
import re
from threading import Thread

# Protocol patterns
PROTOCOLS = {
    'rtsp': r'^rtsp://',
    'rtmp': r'^rtmp://',
    'http': r'^http://',
    'https': r'^https://',
    'udp': r'^udp://',
    'tcp': r'^tcp://',
    'ip': r'^(?:\d{1,3}\.){3}\d{1,3}'
}

# Simulated protocol streaming URLs
STREAM_URLS = {
    'rtsp': 'rtsp://127.0.0.1:554/stream',
    'rtmp': 'rtmp://127.0.0.1:1935/stream',
    'http': 'http://127.0.0.1:8080/stream',
    'https': 'https://127.0.0.1:8443/stream',
    'udp': 'udp://127.0.0.1:5000',
    'tcp': 'tcp://127.0.0.1:5001',
    'ip': '192.168.1.1'
}

# Function to simulate a protocol
def simulate_protocol(protocol, url, cap):
    print(f"Simulating {protocol.upper()} streaming to {url}")
    frame_count = 0
    while frame_count < 100:  # Send 100 frames for each protocol simulation
        ret, frame = cap.read()
        if not ret:
            break

        # Simulate frame delay (real-world frame rate variability)
        frame_rate = random.randint(10, 30)  # Random FPS between 10 and 30
        delay = 1 / frame_rate
        time.sleep(delay)

        # Here, you can send the frame to the URL or log the details
        # For this simulation, we'll just display the frame and the protocol
        cv2.putText(frame, f"{protocol.upper()} - FPS: {frame_rate}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(f"{protocol.upper()} Simulation", frame)

        frame_count += 1

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Main function to capture webcam feed and simulate protocols
def main():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    threads = []
    for protocol, url in STREAM_URLS.items():
        # Validate URL with regex pattern
        if re.match(PROTOCOLS[protocol], url):
            # Run each protocol simulation in a separate thread
            t = Thread(target=simulate_protocol, args=(protocol, url, cap))
            threads.append(t)
            t.start()
        else:
            print(f"Invalid URL for {protocol.upper()}: {url}")

    # Wait for all threads to finish
    for t in threads:
        t.join()

    cap.release()

if __name__ == "__main__":
    main()
