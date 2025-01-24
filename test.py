import cv2
import time
import random
import re
from threading import Thread, Event
import socket
import logging
import argparse
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver
import sys
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('protocol_test.log'),
        logging.StreamHandler()
    ]
)

# Protocol configurations with default ports
PROTOCOLS = {
    'rtsp': {
        'pattern': r'^rtsp://',
        'port': 554,
        'timeout': 5,
        'buffer_size': 1024,
        'retry_attempts': 3
    },
    'rtmp': {
        'pattern': r'^rtmp://',
        'port': 1935,
        'timeout': 5,
        'buffer_size': 1024,
        'retry_attempts': 3
    },
    'http': {
        'pattern': r'^http://',
        'port': 8080,
        'timeout': 5,
        'buffer_size': 1024,
        'retry_attempts': 3
    },
    'https': {
        'pattern': r'^https://',
        'port': 8443,
        'timeout': 5,
        'buffer_size': 1024,
        'retry_attempts': 3
    },
    'tcp': {
        'pattern': r'^tcp://',
        'port': 5001,
        'timeout': 5,
        'buffer_size': 1024,
        'retry_attempts': 3
    }
}

class StreamMetrics:
    """Class to track streaming metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0
        self.dropped_frames = 0
        self.latency_samples = []
        self.bandwidth_samples = []
    
    def update(self, frame_size, latency):
        self.frame_count += 1
        self.latency_samples.append(latency)
        self.bandwidth_samples.append(frame_size)
    
    def get_metrics(self):
        duration = time.time() - self.start_time
        avg_fps = self.frame_count / duration if duration > 0 else 0
        avg_latency = sum(self.latency_samples) / len(self.latency_samples) if self.latency_samples else 0
        avg_bandwidth = sum(self.bandwidth_samples) / len(self.bandwidth_samples) if self.bandwidth_samples else 0
        
        return {
            "protocol": self.protocol,
            "average_fps": round(avg_fps, 2),
            "average_latency_ms": round(avg_latency * 1000, 2),
            "average_bandwidth_kbps": round((avg_bandwidth * 8) / 1000, 2),
            "dropped_frames": self.dropped_frames,
            "total_frames": self.frame_count,
            "duration_seconds": round(duration, 2)
        }

class StreamServer:
    """Base class for stream servers"""
    def __init__(self, host, port, protocol):
        self.host = host
        self.port = port
        self.protocol = protocol
        self.stop_event = Event()
        self.metrics = StreamMetrics()
        self.metrics.protocol = protocol
    
    def start(self):
        raise NotImplementedError
    
    def stop(self):
        self.stop_event.set()

class HTTPStreamServer(StreamServer):
    """HTTP streaming server implementation"""
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/stream':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                
                try:
                    while not self.server.stop_event.is_set():
                        frame = self.server.get_frame()
                        if frame is None:
                            continue
                        
                        # Convert frame to JPEG
                        _, jpeg = cv2.imencode('.jpg', frame)
                        
                        # Send frame
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(jpeg))
                        self.end_headers()
                        self.wfile.write(jpeg.tobytes())
                        self.wfile.write(b'\r\n')
                        
                except Exception as e:
                    logging.error(f"HTTP streaming error: {e}")
    
    def __init__(self, host, port):
        super().__init__(host, port, 'http')
        self.server = HTTPServer((host, port), self.Handler)
        self.server.stop_event = self.stop_event
        self.server.get_frame = self.get_frame
        self.cap = None
    
    def start(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        Thread(target=self.server.serve_forever, daemon=True).start()
        logging.info(f"HTTP server started on {self.host}:{self.port}")
    
    def get_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def stop(self):
        super().stop()
        self.server.shutdown()
        if self.cap:
            self.cap.release()

class ProtocolTester:
    """Class to test different streaming protocols"""
    def __init__(self, server_host, metrics_enabled=True):
        self.server_host = server_host
        self.metrics_enabled = metrics_enabled
        self.servers = {}
        self.metrics = {}
    
    def start_servers(self):
        """Start streaming servers for each protocol"""
        for protocol, config in PROTOCOLS.items():
            try:
                if protocol == 'http':
                    server = HTTPStreamServer(self.server_host, config['port'])
                    server.start()
                    self.servers[protocol] = server
                    logging.info(f"Started {protocol.upper()} server on port {config['port']}")
            except Exception as e:
                logging.error(f"Failed to start {protocol} server: {e}")
    
    def test_protocol(self, protocol, url):
        """Test a specific protocol"""
        config = PROTOCOLS[protocol]
        metrics = StreamMetrics()
        metrics.protocol = protocol
        
        try:
            cap = cv2.VideoCapture(url)
            if not cap.isOpened():
                raise Exception(f"Could not open stream: {url}")
            
            window_name = f"{protocol.upper()} Test"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            while not cv2.waitKey(1) & 0xFF == ord('q'):
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    metrics.dropped_frames += 1
                    continue
                
                # Calculate metrics
                latency = time.time() - start_time
                frame_size = frame.size * frame.itemsize
                metrics.update(frame_size, latency)
                
                # Display metrics on frame
                cv2.putText(frame, f"FPS: {metrics.get_metrics()['average_fps']:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Latency: {metrics.get_metrics()['average_latency_ms']:.1f}ms",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow(window_name, frame)
            
            cv2.destroyWindow(window_name)
            cap.release()
            
        except Exception as e:
            logging.error(f"Error testing {protocol}: {e}")
        finally:
            self.metrics[protocol] = metrics.get_metrics()
    
    def save_results(self):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"protocol_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            logging.info(f"Results saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        for server in self.servers.values():
            try:
                server.stop()
            except Exception as e:
                logging.error(f"Error stopping server: {e}")

def main():
    parser = argparse.ArgumentParser(description='Protocol Testing Tool')
    parser.add_argument('--server', action='store_true', help='Run in server mode')
    parser.add_argument('--client', action='store_true', help='Run in client mode')
    parser.add_argument('--host', default='localhost', help='Server host address')
    args = parser.parse_args()
    
    if not (args.server or args.client):
        parser.error("Must specify either --server or --client")
    
    try:
        if args.server:
            # Server mode
            tester = ProtocolTester(args.host)
            tester.start_servers()
            logging.info("Press Ctrl+C to stop the servers")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logging.info("Stopping servers...")
                tester.cleanup()
        
        elif args.client:
            # Client mode
            tester = ProtocolTester(args.host)
            for protocol, config in PROTOCOLS.items():
                url = f"{protocol}://{args.host}:{config['port']}/stream"
                logging.info(f"Testing {protocol.upper()}: {url}")
                tester.test_protocol(protocol, url)
            
            tester.save_results()
    
    except Exception as e:
        logging.error(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
