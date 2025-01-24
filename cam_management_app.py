import cv2
import threading
import queue
import time
from dataclasses import dataclass
from typing import Dict, Optional, Union, List, Tuple
import numpy as np
import logging
from collections import deque
from datetime import datetime
import os
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
from tkinter import filedialog
from pathlib import Path
import psutil
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import sys
import tempfile
import subprocess
import signal
import sys
import urllib.parse
if sys.platform.startswith('win'):
    import winreg

# Add at the start of your main script
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # Suppress non-critical OpenCV warnings

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class CameraConfig:
    """Camera configuration class"""
    source: Union[str, int]
    name: str
    fps: int = 20
    retry_interval: int = 5
    buffer_size: int = 24
    enable_detection: bool = True
    
    def to_dict(self):
        return {
            "source": self.source,
            "name": self.name,
            "fps": self.fps,
            "retry_interval": self.retry_interval,
            "buffer_size": self.buffer_size,
            "enable_detection": self.enable_detection
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

class FPSCounter:
    """Handles FPS calculation using a rolling window"""
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
    
    def update(self):
        current_time = time.time()
        if self.last_time is not None:
            self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
    
    def get_fps(self) -> float:
        if not self.frame_times:
            return 0.0
        return len(self.frame_times) / sum(self.frame_times)

class FaceDetector:
    """Handles face detection using Haar Cascade"""
    def __init__(self):
        try:
            # Use resource_path for cascade file
            cascade_path = resource_path(os.path.join(cv2.data.haarcascades, 
                                                    'haarcascade_frontalface_default.xml'))
            if not os.path.exists(cascade_path):
                # Fallback to direct OpenCV path
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if not os.path.exists(cascade_path):
                    raise FileNotFoundError(f"Cascade file not found at {cascade_path}")
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            logging.info("Initialized face detector")
            
        except Exception as e:
            logging.error(f"Error initializing face detector: {e}")
            self.face_cascade = None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces

class VideoRecorder:
    """Reliable video recorder with MP4 output"""
    def __init__(self, width: int, height: int, fps: int, output_path: Path, hw_acceleration: dict = None):
        self.width = width
        self.height = height
        self.fps = fps
        self.hw_acceleration = hw_acceleration
        self.lock = threading.Lock()
        
        # Ensure valid dimensions (must be even)
        self.width = max(2, width if width % 2 == 0 else width + 1)
        self.height = max(2, height if height % 2 == 0 else height + 1)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_output_path = output_path
        self.output_path = output_path.parent / f"{output_path.stem}_{timestamp}.mp4"
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.writer = None
        self.recording = False
        self.frame_count = 0
        self.start_time = None
        
        # Segment management
        self.max_segment_duration = 15 * 60  # 15 minutes per file
        self.max_segment_size = 2 * 1024 * 1024 * 1024  # 2GB per file
        self.segment_count = 0
        self.current_segment_size = 0
        self.segment_start_time = None
        
        # Initialize FFmpeg process
        self.ffmpeg_process = None
        self.pipe = None
        
        # Add configurable encoding settings
        self.encoding_preset = "ultrafast"
        
        # Dynamic CRF settings
        self.crf_settings = {
            'HD': {  # 1920x1080 or 1280x720
                'min_crf': 18,
                'max_crf': 28,
                'target_bitrate': 5000  # kbps
            },
            'SD': {  # 854x480 or lower
                'min_crf': 20,
                'max_crf': 30,
                'target_bitrate': 2500  # kbps
            }
        }
        
        # Initialize dynamic CRF
        self.quality_crf = self._calculate_optimal_crf()
        self.last_crf_adjustment = time.time()
        self.crf_adjust_interval = 20  # seconds
        
    def _calculate_optimal_crf(self) -> int:
        """Calculate optimal CRF based on resolution and system performance"""
        try:
            # Determine quality tier based on resolution
            if self.height >= 720:
                settings = self.crf_settings['HD']
            else:
                settings = self.crf_settings['SD']
            
            # Get system metrics
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Base CRF calculation
            if cpu_percent > 80 or memory_percent > 90:
                # High system load - use maximum CRF
                return settings['max_crf']
            elif cpu_percent > 60 or memory_percent > 70:
                # Medium load - use middle range
                return (settings['min_crf'] + settings['max_crf']) // 2
            else:
                # Low load - use minimum CRF
                return settings['min_crf']
                
        except Exception as e:
            logging.error(f"Error calculating CRF: {e}")
            return 23  # Default fallback value
            
    def _adjust_crf_if_needed(self):
        """Dynamically adjust CRF based on system performance"""
        try:
            current_time = time.time()
            if current_time - self.last_crf_adjustment < self.crf_adjust_interval:
                return
                
            self.last_crf_adjustment = current_time
            new_crf = self._calculate_optimal_crf()
            
            # Only update if significant change
            if abs(new_crf - self.quality_crf) >= 2:
                logging.info(f"Adjusting CRF from {self.quality_crf} to {new_crf}")
                self.quality_crf = new_crf
                
        except Exception as e:
            logging.error(f"Error adjusting CRF: {e}")
            
    def _create_new_segment(self) -> bool:
        """Create a new video segment using FFmpeg"""
        try:
            # Generate filename with segment number
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            segment_name = f"{self.base_output_path.stem}_{timestamp}_part{self.segment_count:03d}.mp4"
            self.output_path = self.base_output_path.parent / segment_name
            
            # FFmpeg command for x264 encoding
            command = [
                'ffmpeg',
                '-y',  # Overwrite output file if exists
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-s', f'{self.width}x{self.height}',
                '-pix_fmt', 'bgr24',
                '-r', str(self.fps),
                '-i', '-',  # Input from pipe
                '-c:v', 'libx264',
                '-preset', self.encoding_preset,
                '-crf', str(self.quality_crf),
                '-pix_fmt', 'yuv420p',  # Required for compatibility
                '-y',  # Overwrite output file
                str(self.output_path)
            ]
            
            # Start FFmpeg process
            self.ffmpeg_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.pipe = self.ffmpeg_process.stdin
            self.recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.segment_count += 1
            
            logging.info(f"Started new video segment: {segment_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating new segment: {e}")
            self.cleanup()
            return False
            
    def _should_create_new_segment(self) -> bool:
        """Check if we need to create a new segment based on time or size"""
        current_time = time.time()
        
        # Check duration
        if self.segment_start_time and (current_time - self.segment_start_time >= self.max_segment_duration):
            logging.info("Creating new segment due to duration limit")
            return True
            
        # Check file size
        if self.output_path.exists():
            current_size = self.output_path.stat().st_size
            if current_size >= self.max_segment_size:
                logging.info("Creating new segment due to size limit")
                return True
                
        return False
        
    def write_frame(self, frame: np.ndarray) -> bool:
        """Write frame to FFmpeg pipe with dynamic CRF adjustment"""
        if not self.recording or not self.pipe:
            return False
            
        try:
            with self.lock:
                # Adjust CRF if needed
                self._adjust_crf_if_needed()
                
                # Check if we need to start new segment
                if self._should_create_new_segment():
                    self.cleanup()
                    if not self._create_new_segment():
                        return False
                
                # Ensure frame size matches configuration
                if frame.shape[:2] != (self.height, self.width):
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Ensure frame is BGR
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Write frame to pipe
                try:
                    self.pipe.write(frame.tobytes())
                    self.frame_count += 1
                    return True
                except IOError as e:
                    if self.ffmpeg_process:
                        stderr = self.ffmpeg_process.stderr.read().decode()
                        logging.error(f"FFmpeg error: {stderr}")
                    raise
                
        except Exception as e:
            logging.error(f"Error writing frame: {e}")
            self.cleanup()
            return False
            
    def cleanup(self):
        """Clean up FFmpeg process and resources"""
        try:
            if self.pipe:
                self.pipe.close()
            if self.ffmpeg_process:
                self.ffmpeg_process.wait(timeout=5)
                
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")
            if self.ffmpeg_process:
                self.ffmpeg_process.kill()
        finally:
            self.pipe = None
            self.ffmpeg_process = None
            self.recording = False
            
    def start(self) -> bool:
        """Start recording"""
        try:
            self.segment_count = 0
            return self._create_new_segment()
        except Exception as e:
            logging.error(f"Failed to start recording: {e}")
            self.cleanup()
            return False

class CameraStream:
    """Handles individual camera stream in a separate thread"""
    def __init__(self, config: CameraConfig):
        try:
            self.config = config
            self.frame_queue = queue.Queue(maxsize=5)  # Reduced from 24
            self.stop_event = threading.Event()
            self.is_connected = False
            self.last_frame = None
            self.thread = threading.Thread(target=self._stream_camera)
            self.lock = threading.Lock()
            self.face_detector = FaceDetector() if config.enable_detection else None
            self.detected_faces = []
            
            # Recording attributes
            self.recording = False
            self.out = None
            self.base_recording_path = Path("recordings")
            self.camera_recording_path = self.base_recording_path / self.config.name
            
            # Create directory structure
            self.base_recording_path.mkdir(exist_ok=True)
            self.camera_recording_path.mkdir(exist_ok=True)
            
            self.current_recording_file = None
            self.recording_start_time = None
            
            self.stream_suppressed = False
            
            # Add quality settings
            self.quality_settings = {
                'high': {
                    'preview_width': 640,
                    'preview_height': 480,
                    'recording_width': 1920,
                    'recording_height': 1080,
                    'fps': 30
                },
                'medium': {
                    'preview_width': 480,
                    'preview_height': 360,
                    'recording_width': 1280,
                    'recording_height': 720,
                    'fps': 25
                },
                'low': {
                    'preview_width': 320,
                    'preview_height': 240,
                    'recording_width': 854,
                    'recording_height': 480,
                    'fps': 20
                }
            }
            
            # Auto-adjust quality based on system capacity
            self.current_quality = self._determine_quality()
            self._apply_quality_settings()
            
            # Dynamic thread pool management
            self.min_workers = 1
            self.max_workers = multiprocessing.cpu_count()
            self.target_cpu_per_worker = 25  # Target CPU% per worker
            
            # Initialize thread pool with starting workers
            self.current_workers = min(2, self.max_workers)
            self.process_pool = self._create_thread_pool(self.current_workers)
            
            # Add monitoring intervals
            self.last_pool_adjustment = time.time()
            self.pool_adjust_interval = 20.0  # Check every 20 seconds
            self.performance_history = deque(maxlen=10)  # Store recent performance metrics
            
            # Frame processing queue
            self.process_queue = queue.Queue(maxsize=5)
            
            # Start frame processor thread
            self.processor_thread = threading.Thread(
                target=self._frame_processor,
                name=f"processor_{config.name}"
            )
            self.processor_thread.start()
            
            # Validate camera source before starting
            self._validate_camera_source()
            
            self.frame_ready = threading.Event()  # Add frame ready event
            self.latest_frame = None  # Store latest frame
            
            self.recording_queue = queue.Queue(maxsize=15)  # Reduced from 30
            self.recording_thread = None
            self.recorder = None
            self.recording_stop_event = threading.Event()
            
            # Add multi-camera optimized settings
            self.multi_camera_settings = {
                'high_load': {
                    'preview_width': 320,
                    'preview_height': 240,
                    'recording_width': 1920,
                    'recording_height': 1080,
                    'fps': 15,
                    'process_every_nth_frame': 2  # Process every 2nd frame for preview
                },
                'medium_load': {
                    'preview_width': 480,
                    'preview_height': 360,
                    'recording_width': 1280,
                    'recording_height': 720,
                    'fps': 20,
                    'process_every_nth_frame': 1
                }
            }
            
            # Frame skip counter for preview optimization
            self.frame_counter = 0
            self.skip_frames = 1  # Will be adjusted based on camera count
            
            # Add hardware acceleration settings
            self.hw_acceleration = self._setup_hw_acceleration()
            
            # Add frame handling attributes
            self.cap = None
            self.frame_ready = threading.Event()
            self.last_frame = None
            self.frame_queue = queue.Queue(maxsize=5)  # Reduced buffer size for lower latency
            
            # Start the camera thread
            self.thread = threading.Thread(target=self._stream_camera, daemon=True)
            
            # Initialize GPU accelerator
            self.gpu = GPUAccelerator()
            
        except Exception as e:
            logging.error(f"Error initializing camera {config.name}: {e}")
            raise
        
    def _create_thread_pool(self, num_workers):
        """Create a new thread pool with specified number of workers"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=True)
        return ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix=f"cam_{self.config.name}"
        )
    
    def _adjust_thread_pool(self):
        """Dynamically adjust thread pool size based on system performance"""
        current_time = time.time()
        if current_time - self.last_pool_adjustment < self.pool_adjust_interval:
            return

        try:
            # Get CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_per_worker = cpu_percent / self.current_workers if self.current_workers > 0 else 0
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Store performance metrics
            self.performance_history.append({
                'cpu_percent': cpu_percent,
                'cpu_per_worker': cpu_per_worker,
                'memory_percent': memory_percent
            })
            
            # Calculate averages from history
            avg_cpu = sum(p['cpu_percent'] for p in self.performance_history) / len(self.performance_history)
            avg_cpu_per_worker = sum(p['cpu_per_worker'] for p in self.performance_history) / len(self.performance_history)
            
            # Determine if we should adjust workers
            new_workers = self.current_workers
            
            if avg_cpu_per_worker > self.target_cpu_per_worker * 1.2:  # CPU load too high
                if memory_percent < 80 and self.current_workers < self.max_workers:
                    new_workers = min(self.current_workers + 1, self.max_workers)
                    logging.info(f"Increasing workers to {new_workers} due to high CPU load")
            
            elif avg_cpu_per_worker < self.target_cpu_per_worker * 0.8:  # CPU load too low
                if self.current_workers > self.min_workers:
                    new_workers = max(self.current_workers - 1, self.min_workers)
                    logging.info(f"Decreasing workers to {new_workers} due to low CPU load")
            
            # Apply changes if needed
            if new_workers != self.current_workers:
                self.current_workers = new_workers
                self.process_pool = self._create_thread_pool(new_workers)
                
                logging.info(f"Adjusted thread pool for {self.config.name}: "
                           f"Workers={new_workers}, CPU={avg_cpu:.1f}%, "
                           f"Memory={memory_percent:.1f}%")
            
            self.last_pool_adjustment = current_time
            
        except Exception as e:
            logging.error(f"Error adjusting thread pool: {e}")

    def start(self) -> bool:
        """Start the camera stream thread"""
        self.thread.start()
        logging.info(f"Started camera stream: {self.config.name}")
    
    def stop(self):
        """Enhanced stop method with proper cleanup"""
        try:
            # Set stop event first
            self.stop_event.set()
            
            # Stop recording if active
            if self.recording:
                self.stop_recording()
            
            # Release camera resource
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
            
            # Clear all queues
            self._clear_queues()
            
            # Set flags
            self.is_connected = False
            self.recording = False
            
            logging.info(f"Stopped camera stream: {self.config.name}")
            
        except Exception as e:
            logging.error(f"Error stopping camera {self.config.name}: {e}")
        finally:
            # Ensure all resources are released
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None

    def _clear_queues(self):
        """Clear all queues safely"""
        queues = [self.frame_queue, self.process_queue, self.recording_queue]
        for q in queues:
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass
    
    def suppress_stream(self):
        """Suppress live view stream while maintaining other functions"""
        with self.lock:
            self.stream_suppressed = True
            # Clear frame queue to save memory
            while not self.frame_queue.empty():
                self.frame_queue.get()
        logging.info(f"Stream suppressed for camera: {self.config.name}")
        
    def desuppress_stream(self):
        """Resume live view stream"""
        with self.lock:
            self.stream_suppressed = False
        logging.info(f"Stream resumed for camera: {self.config.name}")
        
    def _setup_hw_acceleration(self) -> dict:
        """Setup hardware acceleration based on system capabilities"""
        acceleration = {
            'enabled': False,
            'type': None,
            'codec': None
        }
        
        try:
            # Check for Intel QuickSync
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                acceleration['enabled'] = True
                acceleration['type'] = 'OpenCL'
                logging.info("OpenCL acceleration enabled")
            
            # Check for NVIDIA GPU
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cv2.cuda.setDevice(0)
                acceleration['enabled'] = True
                acceleration['type'] = 'CUDA'
                logging.info("CUDA acceleration enabled")
            
            # Check for Intel MediaSDK support
            if cv2.videoio_registry.getBackendName(cv2.CAP_INTEL_MFX) != "":
                acceleration['enabled'] = True
                acceleration['type'] = 'Intel_MFX'
                acceleration['codec'] = 'H264_MFX'
                logging.info("Intel MediaSDK acceleration enabled")
            
        except Exception as e:
            logging.warning(f"Failed to initialize hardware acceleration: {e}")
        
        return acceleration

    def _stream_camera(self):
        """Modified streaming with improved IP camera handling"""
        retry_count = 0
        MAX_RETRIES = 3
        
        while not self.stop_event.is_set():
            try:
                # Initialize camera with appropriate backend
                if isinstance(self.config.source, str) and (
                    self.config.source.startswith(('http://', 'https://', 'rtsp://'))):
                    self.cap = cv2.VideoCapture(self.config.source, cv2.CAP_FFMPEG)
                else:
                    # For local cameras, try multiple backends
                    for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
                        self.cap = cv2.VideoCapture(self.config.source, backend)
                        if self.cap.isOpened():
                            break
                        self.cap.release()
                
                if not self.cap.isOpened():
                    raise Exception(f"Failed to open camera: {self.config.source}")
                
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.recording_width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.recording_height)
                self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
                
                self.is_connected = True
                retry_count = 0
                logging.info(f"Connected to camera: {self.config.name}")
                
                while not self.stop_event.is_set():
                    ret, frame = self.cap.read()
                    if not ret:
                        logging.warning(f"Failed to read frame from {self.config.name}")
                        break  # Break inner loop to attempt reconnection
                    
                    try:
                        # Process frame for preview
                        preview_frame = cv2.resize(frame, (self.preview_width, self.preview_height))
                        
                        with self.lock:
                            self.last_frame = preview_frame
                            
                            if self.recording and not self.recording_queue.full():
                                self.recording_queue.put_nowait(frame.copy())
                            
                            if not self.stream_suppressed and not self.frame_queue.full():
                                self.frame_queue.put_nowait(preview_frame.copy())
                        
                        # Face detection if enabled
                        if self.face_detector and not self.stream_suppressed:
                            self.detected_faces = self.face_detector.detect_faces(preview_frame)
                            if self.detected_faces:
                                preview_frame = self._draw_detections(preview_frame)
                        
                        self.frame_ready.set()
                        
                    except queue.Full:
                        continue
                    except Exception as e:
                        logging.error(f"Error processing frame: {e}")
                        continue
                    
                    time.sleep(1.0 / self.config.fps)
                
            except Exception as e:
                logging.error(f"Stream error for {self.config.name}: {e}")
                self.is_connected = False
                
                # Handle reconnection attempts
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    wait_time = min(5 * retry_count, 30)  # Progressive backoff
                    logging.info(f"Attempting reconnection {retry_count}/{MAX_RETRIES} in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logging.error(f"Max retries ({MAX_RETRIES}) reached for {self.config.name}")
                    break
            
            finally:
                if hasattr(self, 'cap') and self.cap:
                    self.cap.release()

    def _frame_processor(self):
        """Modified frame processor with dynamic thread pool adjustment"""
        while not self.stop_event.is_set():
            try:
                # Periodically adjust thread pool
                self._adjust_thread_pool()
                
                task = self.process_queue.get(timeout=0.1)
                if task is None:
                    continue
                
                frame, is_recording = task
                
                # Process frame in parallel if needed
                if self.face_detector and not self.stream_suppressed:
                    # Submit task to current thread pool
                    self.process_pool.submit(self._process_frame, frame.copy())
                
                # Handle recording in main processing thread
                if is_recording and self.out:
                    try:
                        if frame.shape[:2] != (self.recording_height, self.recording_width):
                            frame = cv2.resize(frame, (self.recording_width, self.recording_height))
                        self.out.write(frame)
                    except Exception as e:
                        logging.error(f"Error writing frame: {e}")
                        self.stop_recording()
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in frame processor: {e}")

    def _process_frame(self, frame):
        """Process frame with GPU acceleration if available"""
        try:
            if frame is None:
                return None
                
            # Try GPU processing first
            if self.gpu.cuda_available or self.gpu.opencl_available:
                # Resize for preview
                # Fix: Change the shape comparison
                current_height, current_width = frame.shape[:2]
                if (current_height != self.preview_height) or (current_width != self.preview_width):
                    resized = self.gpu.process_frame(
                        frame, 
                        'resize',
                        width=self.preview_width,
                        height=self.preview_height
                    )
                    if resized is not None:
                        frame = resized
                
                # Face detection if enabled
                if self.face_detector and not self.stream_suppressed:
                    if self.gpu.cuda_available:
                        faces = self.gpu.process_frame(frame, 'detect_faces')
                        if faces is not None and len(faces) > 0:  # Fix: Add explicit length check
                            self.detected_faces = faces
                            frame = self._draw_detections(frame)
                    else:
                        # Fall back to CPU face detection
                        self.detected_faces = self.face_detector.detect_faces(frame)
                        if len(self.detected_faces) > 0:  # Fix: Add explicit length check
                            frame = self._draw_detections(frame)
                
            else:
                # Fall back to CPU processing
                # Fix: Change the shape comparison
                current_height, current_width = frame.shape[:2]
                if (current_height != self.preview_height) or (current_width != self.preview_width):
                    frame = cv2.resize(frame, (self.preview_width, self.preview_height))
                
                if self.face_detector and not self.stream_suppressed:
                    self.detected_faces = self.face_detector.detect_faces(frame)
                    if len(self.detected_faces) > 0:  # Fix: Add explicit length check
                        frame = self._draw_detections(frame)
            
            return frame
            
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return frame

    def _draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw face detection results on frame"""
        for (x, y, w, h) in self.detected_faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add detection info
            cv2.putText(frame, 'Face', (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame with improved handling"""
        try:
            if not self.is_connected or self.stream_suppressed:
                return None
            
            # Try to get frame from queue with timeout
            try:
                frame = self.frame_queue.get(timeout=1.0/self.config.fps)
                return frame
            except queue.Empty:
                # If queue is empty, use last frame if available
                with self.lock:
                    if self.last_frame is not None:
                        return self.last_frame.copy()
        
        except Exception as e:
            logging.error(f"Error getting frame from {self.config.name}: {e}")
        return None

    def get_status(self) -> dict:
        """Get simplified camera status without FPS"""
        try:
            return {
                "connected": self.is_connected,
                "recording": self.recording,
                "recording_file": str(self.current_recording_file) if self.current_recording_file else None,
                "resolution": f"{self.width}x{self.height}" if hasattr(self, 'width') else "Unknown"
            }
        except Exception as e:
            logging.error(f"Error getting camera status: {e}")
            return {
                "connected": False,
                "recording": False,
                "recording_file": None,
                "resolution": "Unknown"
            }

    def start_recording(self):
        """Start recording with improved error handling"""
        try:
            if self.recording:
                return False
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.name}_{timestamp}.avi"
            output_path = self.camera_recording_path / filename
            
            # Initialize recorder
            self.recorder = VideoRecorder(
                width=self.recording_width,
                height=self.recording_height,
                fps=self.config.fps,
                output_path=output_path,
                hw_acceleration=self.hw_acceleration
            )
            
            if self.recorder.start():
                self.recording = True
                logging.info(f"Started recording for camera {self.config.name} at {self.recording_width}x{self.recording_height}")
                
                # Start recording thread
                self.recording_stop_event.clear()
                self.recording_thread = threading.Thread(
                    target=self._recording_worker,
                    name=f"recorder_{self.config.name}"
                )
                self.recording_thread.start()
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Failed to start recording: {e}")
            self.recording = False
            return False

    def _recording_worker(self):
        """Simplified recording worker"""
        try:
            while not self.recording_stop_event.is_set() and self.recording:
                try:
                    # Get frame with timeout
                    frame = self.recording_queue.get(timeout=1.0)
                    if frame is not None:
                        # Simple write operation
                        if not self.recorder.write_frame(frame):
                            logging.error("Failed to write frame")
                            break
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Error in recording worker: {e}")
                    break
        except Exception as e:
            logging.error(f"Recording worker error: {e}")
        finally:
            self.stop_recording()

    def stop_recording(self):
        """Stop recording with proper cleanup"""
        try:
            if not self.recording:
                return False
            
            self.recording_stop_event.set()
            
            # Clear recording queue
            while not self.recording_queue.empty():
                try:
                    self.recording_queue.get_nowait()
                except:
                    pass
            
            # Wait for recording thread
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)
            
            # Cleanup recorder
            if self.recorder:
                self.recorder.cleanup()
                self.recorder = None
            
            self.recording = False
            logging.info(f"Stopped recording for camera {self.config.name}")
            return True
            
        except Exception as e:
            logging.error(f"Error stopping recording: {e}")
            return False

    def enable_low_power_mode(self):
        """Enable low power mode while maintaining recording quality"""
        self.low_power_mode = True
        self.preview_width = 320  # Further reduce preview resolution
        self.preview_height = 240
        self.config.fps = max(5, self.config.fps // 2)  # Reduce preview FPS
        logging.info(f"Enabled low power mode for {self.config.name}: Preview {self.preview_width}x{self.preview_height} @ {self.config.fps}fps")

    def disable_low_power_mode(self):
        """Disable low power mode"""
        self.low_power_mode = False
        self.preview_width = 640
        self.preview_height = 480
        self.config.fps = min(30, self.config.fps * 2)  # Restore preview FPS
        logging.info(f"Disabled low power mode for {self.config.name}")

    def get_performance_metrics(self) -> dict:
        """Get current performance metrics"""
        try:
            return {
                'workers': self.current_workers,
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'performance_history': list(self.performance_history)
            }
        except Exception as e:
            logging.error(f"Error getting performance metrics: {e}")
            return {}

    def _setup_ffmpeg(self):
        """Setup FFMPEG configuration"""
        try:
            # Set FFMPEG backend as default
            os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"  # Disable MSMF
            os.environ["OPENCV_VIDEOIO_PRIORITY_INTEL_MFX"] = "0"  # Disable MFX
            os.environ["OPENCV_VIDEOIO_PRIORITY_GSTREAMER"] = "0"  # Disable GStreamer
            
            # Force FFMPEG backend
            if cv2.videoio_registry.getBackendName(cv2.CAP_FFMPEG) != "":
                logging.info("FFMPEG backend available")
                return True
            
            return False
        except Exception as e:
            logging.error(f"Error setting up FFMPEG: {e}")
            return False

    def _validate_camera_source(self):
        """Enhanced validation for camera sources including IP cameras"""
        try:
            # Setup FFMPEG first
            self._setup_ffmpeg()
            
            # Handle IP camera/RTSP/HTTP streams
            if isinstance(self.config.source, str) and (
                self.config.source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://'))):
                
                # Parse URL components
                parsed_url = urllib.parse.urlparse(self.config.source)
                ip_address = parsed_url.hostname
                port = parsed_url.port
                path = parsed_url.path

                # First, verify if the IP is reachable
                try:
                    import socket
                    sock = socket.create_connection((ip_address, port or 4747), timeout=2)
                    sock.close()
                    logging.info(f"IP {ip_address} is reachable")
                except Exception as e:
                    logging.error(f"IP {ip_address} is not reachable: {e}")
                    raise ValueError(f"Cannot reach IP camera at {ip_address}. Please check if the device is connected to the network and the IP address is correct.")

                # DroidCam specific patterns to try
                droidcam_patterns = [
                    f"http://{ip_address}:4747/video",
                    f"http://{ip_address}:4747/videofeed",
                    f"http://{ip_address}:4747/mjpegfeed",
                    f"http://{ip_address}:4747/h264",
                    # Add websocket variant
                    f"ws://{ip_address}:4747/video",
                    # Add legacy port variant
                    f"http://{ip_address}:8080/video"
                ]

                # Generic IP camera patterns
                ip_camera_patterns = [
                    f"rtsp://{ip_address}:{port or 554}{path or '/'}",
                    f"http://{ip_address}:{port or 80}{path or '/'}",
                    f"rtmp://{ip_address}:{port or 1935}{path or '/'}",
                    self.config.source
                ]

                # Combine all patterns
                url_patterns = droidcam_patterns + ip_camera_patterns

                connection_success = False
                working_url = None

                for url in url_patterns:
                    try:
                        logging.info(f"Attempting connection to: {url}")
                        
                        # Try different API backends
                        for api in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]:
                            try:
                                cap = cv2.VideoCapture(url, api)
                                
                                # Configure capture properties
                                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                                
                                if cap.isOpened():
                                    # Try to read a frame with timeout
                                    start_time = time.time()
                                    while time.time() - start_time < 3:  # 3 second timeout
                                        ret, frame = cap.read()
                                        if ret and frame is not None:
                                            connection_success = True
                                            working_url = url
                                            logging.info(f"Successfully connected to {url} using API {api}")
                                            cap.release()
                                            break
                                    if connection_success:
                                        break
                                cap.release()
                            except Exception as api_error:
                                logging.debug(f"Failed with API {api} for {url}: {api_error}")
                                continue
                        
                        if connection_success:
                            break
                            
                    except Exception as e:
                        logging.debug(f"Failed to connect to {url}: {e}")
                        continue

                if connection_success:
                    logging.info(f"Successfully connected to camera at {working_url}")
                    self.config.source = working_url
                    return
                
                raise ValueError(f"Cannot connect to IP camera at {self.config.source}. Please verify:\n"
                               f"1. The IP camera app is running\n"
                               f"2. The device is on the same network\n"
                               f"3. The IP address and port are correct\n"
                               f"4. No firewall is blocking the connection")

            else:  # Local camera handling remains the same
                try:
                    source = int(self.config.source) if str(self.config.source).isdigit() else self.config.source
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        raise ValueError(f"Cannot open local camera: {self.config.source}")
                    cap.release()
                    
                except Exception as e:
                    raise ValueError(f"Invalid local camera source: {str(e)}")
            
        except Exception as e:
            raise ValueError(f"Invalid camera source: {str(e)}")

    def check_codec_availability(self):
        """Check available codecs with cross-platform CPU support"""
        test_size = (640, 480)
        available_codecs = []
        is_windows = sys.platform.startswith('win')
        
        # Define codecs with cross-platform compatibility
        codecs = [
            ('H264', 'avc1'),     # H.264 - widely supported
            ('XVID', 'XVID'),     # XVID - excellent compatibility
            ('MJPG', 'MJPG'),     # Motion JPEG - universal support
            ('MP4V', 'mp4v'),     # MPEG-4 - good compatibility
        ]
        
        # Add platform-specific codecs
        if is_windows:
            codecs.extend([
                ('WMV2', 'WMV2'),  # Windows Media Video
                ('DIVX', 'DIVX'),  # DivX codec
            ])
        else:  # Linux
            codecs.extend([
                ('X264', 'x264'),  # x264 for Linux
                ('FMP4', 'FMP4'),  # MPEG-4 variant
            ])
        
        # Create test frame
        test_frame = np.zeros((test_size[1], test_size[0], 3), dtype=np.uint8)
        
        # Use a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = Path(temp_dir) / ("test.avi" if is_windows else "test.mp4")
            
            for codec_name, fourcc_name in codecs:
                try:
                    # Test codec availability
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                    out = cv2.VideoWriter(
                        str(test_file),
                        fourcc,
                        30.0,
                        test_size,
                        isColor=True
                    )
                    
                    if out.isOpened():
                        if out.write(test_frame):
                            available_codecs.append(codec_name)
                        out.release()
                        
                except Exception as e:
                    logging.debug(f"Codec {codec_name} test failed: {e}")
                    continue
                finally:
                    if 'out' in locals():
                        out.release()
        
        if available_codecs:
            logging.info(f"Available codecs: {', '.join(available_codecs)}")
            
            # Show platform-specific installation instructions
            if not any(codec in available_codecs for codec in ['H264', 'X264']):
                print("\nTo enable better encoding, install recommended packages:")
                if is_windows:
                    print("Please install:")
                    print("1. K-Lite Codec Pack: https://codecguide.com/download_kl.htm")
                    print("2. FFmpeg: https://www.gyan.dev/ffmpeg/builds/")
                else:  # Linux
                    print("Run these commands:")
                    print("sudo apt update")
                    print("sudo apt install -y ffmpeg")
                    print("sudo apt install -y x264")
                    print("sudo apt install -y ubuntu-restricted-extras")
                    print("sudo apt install -y v4l-utils")
        else:
            print("\nNo codecs available. Install required packages:")
            if is_windows:
                print("Please install:")
                print("1. K-Lite Codec Pack (Full): https://codecguide.com/download_kl.htm")
                print("2. FFmpeg: https://www.gyan.dev/ffmpeg/builds/")
                print("3. OpenCV-Python: pip install opencv-python")
            else:  # Linux
                print("Run these commands:")
                print("sudo apt update")
                print("sudo apt install -y ffmpeg")
                print("sudo apt install -y ubuntu-restricted-extras")
                print("sudo apt install -y x264")
                print("sudo apt install -y v4l-utils")

        return available_codecs

    def _determine_quality(self):
        """Determine appropriate quality based on system resources and CPU type"""
        try:
            cpu_count = multiprocessing.cpu_count()
            total_ram = psutil.virtual_memory().total / (1024**3)  # GB
            
            # Get CPU information
            cpu_info = {}
            if sys.platform.startswith('win'):
                # Windows CPU detection
                try:
                    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                                       r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                    cpu_info['name'] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
                except Exception as e:
                    logging.debug(f"Error reading Windows registry: {e}")
                    cpu_info['name'] = "Unknown"
            else:
                # Linux CPU detection
                try:
                    # Try multiple methods for Linux CPU detection
                    cpu_info['name'] = "Unknown"
                    
                    # Method 1: /proc/cpuinfo
                    try:
                        with open('/proc/cpuinfo') as f:
                            for line in f:
                                if line.startswith('model name'):
                                    cpu_info['name'] = line.split(':')[1].strip()
                                    break
                    except:
                        pass
                    
                    # Method 2: lscpu command
                    if cpu_info['name'] == "Unknown":
                        try:
                            result = subprocess.run(['lscpu'], capture_output=True, text=True)
                            for line in result.stdout.split('\n'):
                                if 'Model name:' in line:
                                    cpu_info['name'] = line.split(':')[1].strip()
                                    break
                        except:
                            pass
                    
                    # Method 3: dmidecode (requires sudo)
                    if cpu_info['name'] == "Unknown":
                        try:
                            result = subprocess.run(['sudo', 'dmidecode', '-t', 'processor'], 
                                                 capture_output=True, text=True)
                            for line in result.stdout.split('\n'):
                                if 'Version:' in line:
                                    cpu_info['name'] = line.split(':')[1].strip()
                                    break
                        except:
                            pass
                    
                except Exception as e:
                    logging.debug(f"Error detecting Linux CPU: {e}")
                    cpu_info['name'] = "Unknown"
            
            # Detect CPU generation and capabilities
            cpu_name = cpu_info.get('name', '').lower()
            is_modern_cpu = any(x in cpu_name for x in [
                'ryzen', 'intel core i', '10th', '11th', '12th', '13th', 
                'zen 2', 'zen 3', 'zen 4'
            ])
            
            # If CPU detection failed, use core count as fallback
            if not is_modern_cpu and cpu_count >= 6:
                is_modern_cpu = True
                logging.info("Using CPU core count for quality determination")
            
            # Determine quality based on CPU and RAM
            if is_modern_cpu and cpu_count >= 6 and total_ram >= 16:
                quality = 'high'
                settings = {
                    'preview_width': 640,
                    'preview_height': 480,
                    'recording_width': 1920,
                    'recording_height': 1080,
                    'fps': 30
                }
            elif cpu_count >= 4 and total_ram >= 8:
                quality = 'medium'
                settings = {
                    'preview_width': 480,
                    'preview_height': 360,
                    'recording_width': 1280,
                    'recording_height': 720,
                    'fps': 25
                }
            else:
                quality = 'low'
                settings = {
                    'preview_width': 320,
                    'preview_height': 240,
                    'recording_width': 854,
                    'recording_height': 480,
                    'fps': 20
                }
            
            # Update quality settings
            self.quality_settings[quality].update(settings)
            logging.info(f"Selected quality profile: {quality} for CPU: {cpu_info.get('name', 'Unknown')}")
            
            return quality
            
        except Exception as e:
            logging.error(f"Error determining quality: {e}")
            return 'low'  # Safe fallback

    def _apply_quality_settings(self):
        """Apply quality settings"""
        settings = self.quality_settings[self.current_quality]
        self.preview_width = settings['preview_width']
        self.preview_height = settings['preview_height']
        self.recording_width = settings['recording_width']
        self.recording_height = settings['recording_height']
        self.config.fps = settings['fps']

class PerformanceMonitor:
    """Monitors system and camera performance"""
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        self.last_check = time.time()
        self.check_interval = 5.0  # seconds
        
    def check_performance(self):
        """Check system performance and adjust camera settings"""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return
            
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        
        if cpu_usage > 80 or mem_usage > 80:
            self.camera_manager.optimize_for_camera_count()
            
        self.last_check = current_time

class CameraManager:
    """Manages multiple camera streams"""
    def __init__(self):
        self.cameras = {}
        self.performance_monitor = PerformanceMonitor(self)
    
    def optimize_for_camera_count(self):
        """Adjust settings based on number of active cameras"""
        camera_count = len(self.cameras)
        
        for camera in self.cameras.values():
            if camera_count > 8:
                # High load optimization
                settings = camera.multi_camera_settings['high_load']
                camera.skip_frames = 2
            elif camera_count > 4:
                # Medium load optimization
                settings = camera.multi_camera_settings['medium_load']
                camera.skip_frames = 1
            else:
                # Default settings
                continue
                
            camera.preview_width = settings['preview_width']
            camera.preview_height = settings['preview_height']
            camera.config.fps = settings['fps']
    
    def add_camera(self, config: CameraConfig):
        """Add a new camera with performance optimization"""
        if config.name in self.cameras:
            logging.warning(f"Camera {config.name} already exists")
            return False
        
        try:
            camera = CameraStream(config)
            self.cameras[config.name] = camera
            camera.start()
            
            # Apply optimizations based on camera count
            self.optimize_for_camera_count()
            return True
            
        except Exception as e:
            logging.error(f"Failed to add camera {config.name}: {e}")
            if config.name in self.cameras:
                del self.cameras[config.name]
            return False
    
    def remove_camera(self, camera_name: str):
        """Remove and stop a camera stream"""
        if camera_name in self.cameras:
            self.cameras[camera_name].stop()
            del self.cameras[camera_name]
            logging.info(f"Removed camera: {camera_name}")
    
    def get_all_frames(self) -> Dict[str, Optional[np.ndarray]]:
        """Get latest frames from all cameras"""
        return {name: camera.get_frame() 
                for name, camera in self.cameras.items()}
    
    def get_all_statuses(self) -> Dict[str, Dict]:
        """Get status of all cameras"""
        return {name: camera.get_status() 
                for name, camera in self.cameras.items()}
    
    def stop_all(self):
        """Enhanced stop all cameras with proper cleanup"""
        try:
            # Stop each camera with timeout
            for name, camera in list(self.cameras.items()):
                try:
                    logging.info(f"Stopping camera: {name}")
                    camera.stop()
                except Exception as e:
                    logging.error(f"Error stopping camera {name}: {e}")
            
            # Clear the cameras dictionary
            self.cameras.clear()
            logging.info("All cameras stopped successfully")
            
        except Exception as e:
            logging.error(f"Error stopping cameras: {e}")

    def check_performance(self):
        """Monitor and adjust for performance"""
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        
        if cpu_usage > 80 or mem_usage > 80:
            self.optimize_for_camera_count()
            for camera in self.cameras.values():
                camera.skip_frames = max(camera.skip_frames, 2)

class SystemResourceAnalyzer:
    def __init__(self):
        self.cpu_count = multiprocessing.cpu_count()
        self.total_memory = psutil.virtual_memory().total
        self.target_cpu_per_camera = 15  # Target CPU% per camera
        self.target_memory_per_camera = 200 * 1024 * 1024  # 200MB per camera
        
    def analyze_capacity(self):
        """Analyze system capacity for cameras"""
        try:
            # Get current system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            available_memory = memory.available
            
            # Calculate remaining resources
            remaining_cpu = max(0, 80 - cpu_usage)  # Keep 20% CPU free
            remaining_memory = available_memory
            
            # Calculate capacity
            cpu_capacity = int(remaining_cpu / self.target_cpu_per_camera)
            memory_capacity = int(remaining_memory / self.target_memory_per_camera)
            
            # Take the minimum of both constraints
            safe_camera_count = min(cpu_capacity, memory_capacity)
            
            # Calculate current load per camera
            current_cameras = len([p for p in psutil.process_iter(['name']) 
                                if 'python' in p.info['name']])
            if current_cameras > 0:
                cpu_per_camera = cpu_usage / current_cameras
                memory_per_camera = (memory.total - memory.available) / current_cameras
            else:
                cpu_per_camera = 0
                memory_per_camera = 0
            
            return {
                'safe_additional_cameras': max(0, safe_camera_count),
                'current_cameras': current_cameras,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'cpu_per_camera': cpu_per_camera,
                'memory_per_camera': memory_per_camera / (1024 * 1024),  # Convert to MB
                'recommendation': self._generate_recommendation(
                    safe_camera_count, cpu_usage, memory_usage
                )
            }
            
        except Exception as e:
            logging.error(f"Error analyzing system capacity: {e}")
            return None
            
    def _generate_recommendation(self, safe_count, cpu_usage, memory_usage):
        """Generate human-readable recommendation"""
        if cpu_usage > 80:
            return "System CPU usage is high. Not recommended to add more cameras."
        if memory_usage > 80:
            return "System memory usage is high. Not recommended to add more cameras."
        
        if safe_count == 0:
            return "System is at capacity. Wait for resources to free up."
        
        quality_level = "high" if safe_count >= 4 else "medium" if safe_count >= 2 else "low"
        return (f"You can safely add {safe_count} more camera(s) with {quality_level} "
                f"quality settings (1080p@30fps)")

class ResourceMonitor(ttk.Frame):
    def __init__(self, parent, camera_manager):
        super().__init__(parent)
        self.camera_manager = camera_manager  # Store reference to camera_manager
        self.setup_ui()
        self.last_update = time.time()
        self.update_interval = 1.0  # Update every second
        
    def setup_ui(self):
        """Setup the resource monitor UI"""
        # Create styled frame
        style = ttk.Style()
        style.configure('Resource.TFrame', background='#f0f0f0')
        self.configure(style='Resource.TFrame', padding="5")
        
        # Title
        ttk.Label(self, text="System Resources", 
                 font=('Helvetica', 10, 'bold')).pack(anchor='w')
        
        # CPU Usage
        self.cpu_frame = ttk.Frame(self)
        self.cpu_frame.pack(fill='x', pady=2)
        ttk.Label(self.cpu_frame, text="CPU:").pack(side='left')
        self.cpu_label = ttk.Label(self.cpu_frame, text="0%")
        self.cpu_label.pack(side='right')
        
        # RAM Usage
        self.ram_frame = ttk.Frame(self)
        self.ram_frame.pack(fill='x', pady=2)
        ttk.Label(self.ram_frame, text="RAM:").pack(side='left')
        self.ram_label = ttk.Label(self.ram_frame, text="0/0 GB")
        self.ram_label.pack(side='right')
        
        # Storage
        self.storage_frame = ttk.Frame(self)
        self.storage_frame.pack(fill='x', pady=2)
        ttk.Label(self.storage_frame, text="Storage:").pack(side='left')
        self.storage_label = ttk.Label(self.storage_frame, text="0/0 GB")
        self.storage_label.pack(side='right')
        
        # Recording Space
        self.recording_frame = ttk.Frame(self)
        self.recording_frame.pack(fill='x', pady=2)
        ttk.Label(self.recording_frame, text="Recordings:").pack(side='left')
        self.recording_label = ttk.Label(self.recording_frame, text="0 GB")
        self.recording_label.pack(side='right')
        
        # Add Thread Pool Monitor
        ttk.Label(self, text="Thread Pools", 
                 font=('Helvetica', 10, 'bold')).pack(pady=5)
        
        self.pool_frame = ttk.Frame(self)
        self.pool_frame.pack(fill='x', pady=2)
        self.pool_label = ttk.Label(self.pool_frame, text="No active pools")
        self.pool_label.pack(side='left')
    
    def update_metrics(self):
        """Update resource metrics including thread pool status"""
        try:
            current_time = time.time()
            if current_time - self.last_update >= self.update_interval:
                # CPU Usage
                cpu_percent = psutil.cpu_percent()
                self.cpu_label.config(
                    text=f"{cpu_percent:.1f}%",
                    foreground='red' if cpu_percent > 80 else 'black'
                )
                
                # RAM Usage
                ram = psutil.virtual_memory()
                ram_used = ram.used / (1024**3)  # Convert to GB
                ram_total = ram.total / (1024**3)
                self.ram_label.config(
                    text=f"{ram_used:.1f}/{ram_total:.1f} GB",
                    foreground='red' if ram.percent > 80 else 'black'
                )
                
                # Storage Usage
                storage = psutil.disk_usage('/')
                storage_used = storage.used / (1024**3)
                storage_total = storage.total / (1024**3)
                self.storage_label.config(
                    text=f"{storage_used:.1f}/{storage_total:.1f} GB",
                    foreground='red' if storage.percent > 80 else 'black'
                )
                
                # Recording Space Usage
                recording_path = Path("recordings")
                if recording_path.exists():
                    total_size = sum(f.stat().st_size for f in recording_path.rglob('*') if f.is_file())
                    recording_size = total_size / (1024**3)  # Convert to GB
                    self.recording_label.config(text=f"{recording_size:.1f} GB")
                
                # Update thread pool information
                pool_info = []
                for name, camera in self.camera_manager.cameras.items():
                    metrics = camera.get_performance_metrics()
                    pool_info.append(
                        f"{name}: {metrics.get('workers', 0)} workers "
                        f"({metrics.get('cpu_usage', 0):.1f}% CPU)"
                    )
                
                if pool_info:
                    self.pool_label.config(text="\n".join(pool_info))
                else:
                    self.pool_label.config(text="No active pools")
                
                self.last_update = current_time
        except Exception as e:
            logging.error(f"Error updating resource metrics: {e}")

class CameraApp(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Add shutdown flag
        self.shutting_down = False
        
        # Bind window close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.title("Camera Management System")
        self.state('normal')
        
        # Check codec availability and show warning if needed
        self._check_system_requirements()
        
        # Initialize camera manager first
        self.camera_manager = CameraManager()
        self.active_cameras = {}
        self.camera_frames = {}
        
        self.setup_ui()
        self.load_camera_config()
        
        # Start update loop
        self.update_loop()
    
    def setup_ui(self):
        """Setup the main UI components"""
        # Create main containers
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        self.camera_container = ttk.Frame(self)
        self.camera_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                      
        # Add control buttons
        ttk.Label(self.control_frame, text="Camera Controls", 
                 font=('Helvetica', 12, 'bold')).pack(pady=5)
        
        # Create button frames for each row
        button_frame1 = ttk.Frame(self.control_frame)
        button_frame1.pack(fill=tk.X, pady=2)
        button_frame2 = ttk.Frame(self.control_frame)
        button_frame2.pack(fill=tk.X, pady=2)
        
       # First row: Add and Remove buttons
        ttk.Button(button_frame1, text="Add Camera", command=self.show_add_camera_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame1, text="Remove Camera", command=self.show_remove_camera_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Second row: Save and Load buttons
        ttk.Button(button_frame2, text="Save Config", command=self.save_camera_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(button_frame2, text="Load Config", command=self.load_camera_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        
        # Add recording controls
        ttk.Label(self.control_frame, text="Recording Controls", 
                 font=('Helvetica', 0, 'bold')).pack(pady=5)
        # Create button frame for recording controls
        recording_frame = ttk.Frame(self.control_frame)
        recording_frame.pack(fill=tk.X, pady=2)

        self.record_button = ttk.Button(recording_frame,
                                      text="Start Recording",
                                      command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.suppress_button = ttk.Button(recording_frame,
                                        text="Suppress Stream", 
                                        command=self.toggle_stream_suppression)
        self.suppress_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(
            self.control_frame,
            text="Check System Capacity",
            command=self.check_system_capacity
        ).pack(fill=tk.X, pady=2)
        # Add camera list
        ttk.Label(self.control_frame, text="Active Cameras", font=('Helvetica', 10, 'bold')).pack(pady=5)
        self.camera_list = ttk.Treeview(self.control_frame, height=10)
        self.camera_list.pack(fill=tk.X, pady=5)
        self.camera_list["columns"] = ("status",)
        self.camera_list.column("#0", width=150)
        self.camera_list.column("status", width=90)
        self.camera_list.heading("#0", text="Camera")
        self.camera_list.heading("status", text="Status")
        
        # Add separator before resource monitor
        ttk.Separator(self.control_frame, orient='horizontal').pack(
            fill='x', pady=10)
        
        # Add resource monitor
        self.resource_monitor = ResourceMonitor(self.control_frame, self.camera_manager)
        self.resource_monitor.pack(fill='x', pady=5, padx=5)
        
        # Add Recording Settings button
        ttk.Button(
            self.control_frame,
            text="Recording Settings",
            command=self.show_recording_settings
        ).pack(fill=tk.X, pady=2)
    
    def show_add_camera_dialog(self):
        """Show dialog for adding a new camera"""
        dialog = tk.Toplevel(self)
        dialog.title("Add Camera")
        dialog.geometry("300x400")
        
        ttk.Label(dialog, text="Camera Name:").pack(pady=5)
        name_entry = ttk.Entry(dialog)
        name_entry.pack(pady=5)
        
        ttk.Label(dialog, text="Source:").pack(pady=5)
        source_entry = ttk.Entry(dialog)
        source_entry.pack(pady=5)
        
        ttk.Label(dialog, text="FPS:").pack(pady=5)
        fps_entry = ttk.Entry(dialog)
        fps_entry.insert(0, "30")
        fps_entry.pack(pady=5)
        
        # Enable face detection checkbox
        enable_detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dialog, text="Enable Face Detection", 
                       variable=enable_detection_var).pack(pady=5)
        
        def add_camera():
            try:
                source = source_entry.get()
                # Convert to integer if it's a number (for webcam indices)
                if source.isdigit():
                    source = int(source)
                    
                config = CameraConfig(
                    source=source,
                    name=name_entry.get(),
                    fps=int(fps_entry.get()),
                    enable_detection=enable_detection_var.get()
                )
                
                self.add_camera(config)
                dialog.destroy()
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add camera: {str(e)}")
        
        ttk.Button(dialog, text="Add", command=add_camera).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)
    
    def show_remove_camera_dialog(self):
        """Show dialog for removing a camera"""
        if not self.active_cameras:
            messagebox.showinfo("Info", "No active cameras to remove")
            return
            
        dialog = tk.Toplevel(self)
        dialog.title("Remove Camera")
        dialog.geometry("300x200")
        
        ttk.Label(dialog, text="Select camera to remove:").pack(pady=5)
        
        camera_var = tk.StringVar()
        camera_dropdown = ttk.Combobox(dialog, textvariable=camera_var)
        camera_dropdown['values'] = list(self.active_cameras.keys())
        camera_dropdown.pack(pady=5)
        
        def remove_selected():
            camera_name = camera_var.get()
            if camera_name:
                self.remove_camera(camera_name)
            dialog.destroy()
        
        ttk.Button(dialog, text="Remove", command=remove_selected).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack(pady=5)
    
    def add_camera(self, config: CameraConfig):
        """Add camera with improved display setup"""
        if config.name in self.active_cameras:
            messagebox.showerror("Error", f"Camera {config.name} already exists")
            return
        
        try:
            if not self.camera_manager.add_camera(config):
                return
                
            self.active_cameras[config.name] = config
            
            # Create frame with fixed minimum size
            camera_frame = ttk.LabelFrame(self.camera_container, text=config.name)
            camera_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add canvas with minimum size
            canvas = tk.Canvas(camera_frame, bg='black', width=640, height=480)
            canvas.pack(fill=tk.BOTH, expand=True)
            
            self.camera_frames[config.name] = canvas
            
            # Update camera list
            self.camera_list.insert("", "end", text=config.name, values=("Active", "0"))
            
            messagebox.showinfo("Success", f"Added camera: {config.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add camera: {str(e)}")
            if config.name in self.active_cameras:
                del self.active_cameras[config.name]
            if config.name in self.camera_frames:
                self.camera_frames[config.name].master.destroy()
                del self.camera_frames[config.name]
    
    def remove_camera(self, camera_name: str):
        """Remove a camera from the system"""
        try:
            # Remove from camera manager
            self.camera_manager.remove_camera(camera_name)
            
            # Remove frame
            if camera_name in self.camera_frames:
                self.camera_frames[camera_name].master.destroy()
                del self.camera_frames[camera_name]
            
            # Remove from active cameras
            del self.active_cameras[camera_name]
            
            # Update camera list
            for item in self.camera_list.get_children():
                if self.camera_list.item(item)["text"] == camera_name:
                    self.camera_list.delete(item)
                    break
            
            messagebox.showinfo("Success", f"Removed camera: {camera_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove camera: {str(e)}")
    
    def toggle_recording(self):
        """Toggle recording for selected camera"""
        selection = self.camera_list.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a camera to record")
            return
            
        camera_name = self.camera_list.item(selection[0])["text"]
        if camera_name in self.camera_manager.cameras:
            camera = self.camera_manager.cameras[camera_name]
            
            if not camera.recording:
                if camera.start_recording():
                    self.record_button.configure(text="Stop Recording")
                    messagebox.showinfo("Success", f"Started recording camera: {camera_name}")
            else:
                if camera.stop_recording():
                    self.record_button.configure(text="Start Recording")
                    messagebox.showinfo("Success", f"Stopped recording camera: {camera_name}")
    
    def toggle_stream_suppression(self):
        """Toggle stream suppression for selected camera"""
        selection = self.camera_list.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a camera")
            return
            
        camera_name = self.camera_list.item(selection[0])["text"]
        if camera_name in self.camera_manager.cameras:
            camera = self.camera_manager.cameras[camera_name]
            
            if not camera.stream_suppressed:
                camera.suppress_stream()
                self.suppress_button.configure(text="Resume Stream")
                # Clear the camera frame
                if camera_name in self.camera_frames:
                    self.camera_frames[camera_name].delete("all")
            else:
                camera.desuppress_stream()
                self.suppress_button.configure(text="Suppress Stream")
    
    def suppress_and_record(self):
        """Suppress stream and start recording"""
        selection = self.camera_list.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a camera")
            return
            
        camera_name = self.camera_list.item(selection[0])["text"]
        if camera_name in self.camera_manager.cameras:
            camera = self.camera_manager.cameras[camera_name]
            
            try:
                # Start recording if not already recording
                if not camera.recording:
                    if camera.start_recording():
                        # Suppress the stream
                        camera.suppress_stream()
                        self.suppress_button.configure(text="Resume Stream")
                        self.suppress_record_button.configure(text="Stop Record & Resume")
                        # Clear the camera frame
                        if camera_name in self.camera_frames:
                            self.camera_frames[camera_name].delete("all")
                else:
                    # If already recording, stop recording and resume stream
                    camera.stop_recording()
                    camera.desuppress_stream()
                    self.suppress_button.configure(text="Suppress Stream")
                    self.suppress_record_button.configure(text="Suppress & Record")
            
            except Exception as e:
                messagebox.showerror("Error", f"Operation failed: {str(e)}")
    
    def update_loop(self):
        """Optimized update loop for smoother display"""
        try:
            current_time = time.time()
            
            # Update displays immediately for each camera
            for name, camera in self.camera_manager.cameras.items():
                if camera.stream_suppressed:
                    continue
                
                try:
                    frame = camera.get_frame()
                    if frame is None:
                        continue
                    
                    canvas = self.camera_frames.get(name)
                    if not canvas or not canvas.winfo_viewable():
                        continue
                    
                    # Always update frame for live view
                    self._update_camera_display(canvas, frame, 
                        (canvas.winfo_width(), canvas.winfo_height()))
                    
                except Exception as e:
                    logging.error(f"Display error for camera {name}: {e}")
            
            # Update resource monitor less frequently
            if current_time - getattr(self, '_last_resource_update', 0) >= 5.0:
                self.resource_monitor.update_metrics()
                self._last_resource_update = current_time
                self._update_camera_status()
            
        except Exception as e:
            logging.error(f"Error in update loop: {e}")
        finally:
            # Schedule next update sooner for smoother display
            self._update_loop_id = self.after(16, self.update_loop)  # ~60 FPS
    
    def _update_camera_display(self, canvas, frame, dims):
        """Optimized display update with error checking"""
        try:
            if frame is None or dims[0] <= 0 or dims[1] <= 0:
                return
            
            # Convert frame to RGB for display
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            # Calculate aspect ratio
            aspect_ratio = frame.shape[1] / frame.shape[0]
            
            # Calculate display dimensions maintaining aspect ratio
            display_width = dims[0]
            display_height = int(dims[0] / aspect_ratio)
            
            if display_height > dims[1]:
                display_height = dims[1]
                display_width = int(dims[1] * aspect_ratio)
            
            # Resize frame
            frame_resized = cv2.resize(frame_rgb, (display_width, display_height),
                                     interpolation=cv2.INTER_LINEAR)
            
            # Convert to PhotoImage
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)
            
            # Center the image
            x_center = dims[0] // 2
            y_center = dims[1] // 2
            
            # Update canvas
            canvas.delete("all")
            canvas.create_image(x_center, y_center, image=photo, anchor=tk.CENTER)
            canvas.photo = photo  # Keep reference to prevent garbage collection
            
        except Exception as e:
            logging.error(f"Error updating display: {e}")
            # Draw error message on canvas
            canvas.delete("all")
            canvas.create_text(dims[0]//2, dims[1]//2, 
                             text="Error displaying video",
                             fill="red")
    
    def _update_camera_status(self):
        """Update camera status in tree view"""
        for name, camera in self.camera_manager.cameras.items():
            try:
                status = camera.get_status()
                for item in self.camera_list.get_children():
                    if self.camera_list.item(item)["text"] == name:
                        status_text = []
                        if status.get("recording", False):
                            status_text.append("Recording")
                        if camera.stream_suppressed:
                            status_text.append("Suppressed")
                        elif status.get("connected", False):
                            status_text.append("Active")
                        else:
                            status_text.append("Inactive")
                        
                        self.camera_list.item(item, values=(" + ".join(status_text),))
                        break
            except Exception as e:
                logging.error(f"Error updating status for camera {name}: {e}")
    
    def save_camera_config(self):
        """Save camera configurations to file"""
        try:
            config_data = {name: config.to_dict() 
                          for name, config in self.active_cameras.items()}
            
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")]
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(config_data, f, indent=4)
                messagebox.showinfo("Success", "Configuration saved successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def load_camera_config(self):
        """Load camera configurations from file"""
        try:
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json")]
            )
            
            if file_path:
                with open(file_path, 'r') as f:
                    config_data = json.load(f)
                
                # Remove existing cameras
                for name in list(self.active_cameras.keys()):
                    self.remove_camera(name)
                
                # Add cameras from config
                for name, data in config_data.items():
                    config = CameraConfig.from_dict(data)
                    self.add_camera(config)
                
                messagebox.showinfo("Success", "Configuration loaded successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration: {str(e)}")
    
    def on_closing(self):
        """Enhanced closing handler with safe shutdown"""
        if getattr(self, 'shutting_down', False):  # Prevent multiple shutdown attempts
            return
            
        try:
            self.shutting_down = True
            logging.info("Initiating application shutdown...")
            
            # Show shutdown progress
            try:
                progress = tk.Toplevel(self)
                progress.title("Shutting Down")
                progress.transient(self)
                progress.grab_set()
                progress.geometry("300x100")
                progress.resizable(False, False)
                
                label = ttk.Label(progress, text="Shutting down cameras...\nPlease wait...")
                label.pack(pady=20)
                
                self.update_idletasks()
            except:
                pass

            def shutdown():
                try:
                    # Cancel all pending tasks first
                    for after_id in self.tk.eval('after info').split():
                        try:
                            self.after_cancel(after_id)
                        except:
                            pass

                    # Stop all cameras and recordings
                    if hasattr(self, 'camera_manager'):
                        for camera in list(self.camera_manager.cameras.values()):
                            try:
                                if camera.recording:
                                    camera.stop_recording()
                                camera.stop_event.set()
                                self._clear_camera_queues(camera)
                            except Exception as e:
                                logging.error(f"Error stopping camera: {e}")

                    # Wait for all camera threads to finish
                    if hasattr(self, 'camera_manager'):
                        for camera in list(self.camera_manager.cameras.values()):
                            try:
                                if camera.thread.is_alive():
                                    camera.thread.join(timeout=2.0)
                            except Exception as e:
                                logging.error(f"Error joining camera thread: {e}")

                    # Release all camera resources
                    if hasattr(self, 'camera_manager'):
                        for camera in list(self.camera_manager.cameras.values()):
                            try:
                                if hasattr(camera, 'cap') and camera.cap:
                                    camera.cap.release()
                            except Exception as e:
                                logging.error(f"Error releasing camera: {e}")

                    # Clear all queues
                    if hasattr(self, 'camera_manager'):
                        for camera in list(self.camera_manager.cameras.values()):
                            self._clear_camera_queues(camera)

                    # Clear UI
                    try:
                        for widget in self.winfo_children():
                            widget.destroy()
                    except:
                        pass

                    # Clear camera manager
                    if hasattr(self, 'camera_manager'):
                        self.camera_manager.cameras.clear()

                    logging.info("Cleanup completed, exiting application")
                    
                except Exception as e:
                    logging.error(f"Error during shutdown: {e}")
                finally:
                    # Force application exit
                    try:
                        self.quit()
                        self.destroy()
                    except:
                        pass
                    finally:
                        os._exit(0)

            # Run shutdown in separate thread
            threading.Thread(target=shutdown, daemon=False).start()
            
        except Exception as e:
            logging.error(f"Error in shutdown: {e}")
            os._exit(1)

    def _clear_camera_queues(self, camera):
        """Safely clear all queues for a camera"""
        try:
            queues = []
            # Safely get all queue attributes
            for attr in ['frame_queue', 'process_queue', 'recording_queue']:
                if hasattr(camera, attr):
                    queue_obj = getattr(camera, attr)
                    if queue_obj:
                        queues.append(queue_obj)

            # Clear each queue
            for q in queues:
                try:
                    while True:
                        q.get_nowait()
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.debug(f"Error clearing queue: {e}")
                
        except Exception as e:
            logging.error(f"Error in _clear_camera_queues: {e}")

    def check_system_capacity(self):
        """Check and display system capacity for cameras"""
        analyzer = SystemResourceAnalyzer()
        analysis = analyzer.analyze_capacity()
        
        if analysis:
            message = (
                f"System Analysis:\n\n"
                f"Current Cameras: {analysis['current_cameras']}\n"
                f"Safe Additional Cameras: {analysis['safe_additional_cameras']}\n"
                f"Current CPU Usage: {analysis['cpu_usage']:.1f}%\n"
                f"Current Memory Usage: {analysis['memory_usage']:.1f}%\n\n"
                f"Per Camera Usage:\n"
                f"CPU: {analysis['cpu_per_camera']:.1f}%\n"
                f"Memory: {analysis['memory_per_camera']:.1f}MB\n\n"
                f"Recommendation:\n{analysis['recommendation']}"
            )
            
            messagebox.showinfo("System Capacity Analysis", message)

    def _check_system_requirements(self):
        """Check system requirements including codec availability"""
        try:
            # Create test camera to check codecs
            test_camera = CameraStream(CameraConfig(source=0, name="test"))
            available_codecs = test_camera.check_codec_availability()
            
            if not available_codecs:
                messagebox.showwarning(
                    "Codec Warning",
                    "No video codecs found. Please install required codecs:\n\n"
                    "For Ubuntu/Debian:\n"
                    "sudo apt-get update\n"
                    "sudo apt-get install -y ubuntu-restricted-extras\n"
                    "sudo apt-get install -y ffmpeg\n\n"
                    "For Fedora:\n"
                    "sudo dnf install ffmpeg\n"
                    "sudo dnf install gstreamer1-plugins-base gstreamer1-plugins-good\n\n"
                    "For Windows:\n"
                    "Install K-Lite Codec Pack"
                )
            else:
                logging.info(f"Available video codecs: {', '.join(available_codecs)}")
            
        except Exception as e:
            logging.error(f"Error checking system requirements: {e}")

    def show_recording_settings(self):
        """Show recording settings dialog"""
        RecordingSettingsDialog(self, self.camera_manager)

    def initialize_camera(self, camera_id, source, camera_type='IP'):
        """Initialize camera with enhanced error handling and connection retries"""
        try:
            logging.info(f"Initializing camera {camera_id} with source {source}")
            
            if camera_type == 'IP':
                # Ensure source has protocol
                if not any(source.startswith(p) for p in ('http://', 'https://', 'rtsp://', 'rtmp://')):
                    source = f'http://{source}'
                
                # Parse URL
                parsed = urllib.parse.urlparse(source)
                ip_address = parsed.hostname
                port = parsed.port
                path = parsed.path or '/video'
                
                # Try multiple connection methods
                connection_methods = [
                    (f"http://{ip_address}:{port or 80}{path}", cv2.CAP_FFMPEG),
                    (f"rtsp://{ip_address}:{port or 554}{path}", cv2.CAP_FFMPEG),
                    (f"http://{ip_address}:4747/video", cv2.CAP_FFMPEG),  # DroidCam
                    (f"http://{ip_address}:4747/videofeed", cv2.CAP_FFMPEG),  # DroidCam alternate
                ]
                
                last_error = None
                for url, api_preference in connection_methods:
                    try:
                        logging.info(f"Trying connection to {url}")
                        cap = cv2.VideoCapture(url, api_preference)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        if cap.isOpened():
                            # Try to read a frame with timeout
                            start_time = time.time()
                            while time.time() - start_time < 5:  # 5 second timeout
                                ret, frame = cap.read()
                                if ret and frame is not None:
                                    logging.info(f"Successfully connected to {url}")
                                    return cap
                            cap.release()
                    except Exception as e:
                        last_error = e
                        logging.warning(f"Failed to connect to {url}: {e}")
                        if cap:
                            cap.release()
                        continue
                
                raise ValueError(f"Failed to connect to IP camera: {last_error}")
                
            elif camera_type == 'USB':
                # Handle USB camera initialization
                # ... (rest of the USB camera code remains unchanged)
                pass
                
        except Exception as e:
            logging.error(f"Error initializing camera {camera_id}: {str(e)}")
            raise

    def test_camera_connection(self, source, camera_type='IP'):
        """Test camera connection before adding"""
        try:
            if camera_type == 'IP':
                # Validate IP camera URL format
                if '://' not in source:
                    source = f'http://{source}'
                
                # Try different connection methods
                methods = [
                    (lambda: cv2.VideoCapture(source), "Direct URL"),
                    (lambda: cv2.VideoCapture(source.replace('http://', 'rtsp://')), "RTSP"),
                    (lambda: cv2.VideoCapture(source.replace('http://', 'rtmp://')), "RTMP"),
                ]
                
                for method, desc in methods:
                    try:
                        logging.info(f"Trying {desc}: {source}")
                        cap = method()
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                cap.release()
                                return True, f"Connection successful using {desc}"
                        cap.release()
                    except Exception as e:
                        logging.warning(f"{desc} connection failed: {e}")
                
                # Try common IP camera ports
                try:
                    ip = source.split('/')[2].split(':')[0]
                    for port in [80, 8080, 4747]:
                        direct_url = f"http://{ip}:{port}/video"
                        logging.info(f"Trying direct IP: {direct_url}")
                        cap = cv2.VideoCapture(direct_url)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                cap.release()
                                return True, f"Connection successful using port {port}"
                        cap.release()
                except Exception as e:
                    logging.warning(f"Direct IP connection failed: {e}")
                
                return False, "Could not connect to camera using any method"
                
            elif camera_type == 'USB':
                success = False
                message = "Could not connect to USB camera"
                
                try:
                    # Try as numeric index
                    index = int(source)
                    cap = cv2.VideoCapture(index)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            success = True
                            message = f"Successfully connected to USB camera at index {index}"
                    cap.release()
                    
                    # If failed, try other indices
                    if not success:
                        for idx in range(5):
                            if idx != index:
                                cap = cv2.VideoCapture(idx)
                                if cap.isOpened():
                                    ret, frame = cap.read()
                                    if ret:
                                        success = True
                                        message = f"Found camera at index {idx} instead of {index}"
                                        break
                                cap.release()
                
                except ValueError:
                    # Try as path
                    cap = cv2.VideoCapture(source)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret:
                            success = True
                            message = "Successfully connected to USB camera"
                    cap.release()
                
                return success, message
                
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

class GPUAccelerator:
    """Handles GPU acceleration capabilities"""
    def __init__(self):
        self.cuda_available = False
        self.opencl_available = False
        self.gpu_name = None
        self.gpu_memory = None
        self._init_gpu()
        
    def _init_gpu(self):
        """Initialize GPU capabilities"""
        try:
            # Check CUDA availability
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cv2.cuda.setDevice(0)
                self.cuda_available = True
                self.gpu_name = cv2.cuda.getDevice()
                logging.info(f"CUDA GPU available: Device {self.gpu_name}")
                
            # Check OpenCL availability
            if cv2.ocl.haveOpenCL():
                cv2.ocl.setUseOpenCL(True)
                self.opencl_available = True
                platform = cv2.ocl.Platform().name()
                logging.info(f"OpenCL available: {platform}")
                
        except Exception as e:
            logging.warning(f"GPU initialization error: {e}")
            
    def process_frame(self, frame: np.ndarray, operation: str = 'resize', **kwargs) -> np.ndarray:
        """Process frame using available GPU acceleration"""
        try:
            if self.cuda_available:
                return self._cuda_process(frame, operation, **kwargs)
            elif self.opencl_available:
                return self._opencl_process(frame, operation, **kwargs)
            return None  # Indicate GPU processing not available
        except Exception as e:
            logging.error(f"GPU processing error: {e}")
            return None
            
    def _cuda_process(self, frame: np.ndarray, operation: str, **kwargs) -> np.ndarray:
        """Process frame using CUDA"""
        try:
            # Upload frame to GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            if operation == 'resize':
                width = kwargs.get('width', frame.shape[1])
                height = kwargs.get('height', frame.shape[0])
                gpu_frame = cv2.cuda.resize(gpu_frame, (width, height))
            elif operation == 'convert_color':
                conversion = kwargs.get('conversion', cv2.COLOR_BGR2RGB)
                gpu_frame = cv2.cuda.cvtColor(gpu_frame, conversion)
            elif operation == 'detect_faces':
                # Use GPU-accelerated cascade classifier if available
                cascade = cv2.cuda.CascadeClassifier_create(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                gpu_gray = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(gpu_gray).download()
                return faces
                
            # Download result from GPU
            return gpu_frame.download()
            
        except Exception as e:
            logging.error(f"CUDA processing error: {e}")
            return None
            
    def _opencl_process(self, frame: np.ndarray, operation: str, **kwargs) -> np.ndarray:
        """Process frame using OpenCL"""
        try:
            # Convert frame to UMat for OpenCL processing
            gpu_frame = cv2.UMat(frame)
            
            if operation == 'resize':
                width = kwargs.get('width', frame.shape[1])
                height = kwargs.get('height', frame.shape[0])
                gpu_frame = cv2.resize(gpu_frame, (width, height))
            elif operation == 'convert_color':
                conversion = kwargs.get('conversion', cv2.COLOR_BGR2RGB)
                gpu_frame = cv2.cvtColor(gpu_frame, conversion)
                
            # Get result back from GPU
            return gpu_frame.get()
            
        except Exception as e:
            logging.error(f"OpenCL processing error: {e}")
            return None

class RecordingSettingsDialog(tk.Toplevel):
    """Dialog for adjusting recording settings"""
    def __init__(self, parent, camera_manager):
        super().__init__(parent)
        self.camera_manager = camera_manager
        self.title("Recording Settings")
        self.geometry("400x500")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Segment Duration Settings
        ttk.Label(main_frame, text="Segment Duration (minutes):", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.duration_var = tk.StringVar(value="15")
        duration_frame = ttk.Frame(main_frame)
        duration_frame.pack(fill=tk.X, pady=(0, 10))
        duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var, width=10)
        duration_entry.pack(side=tk.LEFT)
        ttk.Label(duration_frame, text="minutes").pack(side=tk.LEFT, padx=5)
        
        # Segment Size Settings
        ttk.Label(main_frame, text="Maximum Segment Size:", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.size_var = tk.StringVar(value="2")
        size_frame = ttk.Frame(main_frame)
        size_frame.pack(fill=tk.X, pady=(0, 10))
        size_entry = ttk.Entry(size_frame, textvariable=self.size_var, width=10)
        size_entry.pack(side=tk.LEFT)
        ttk.Label(size_frame, text="GB").pack(side=tk.LEFT, padx=5)
        
        # Encoding Speed Settings
        ttk.Label(main_frame, text="Encoding Speed:", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.speed_var = tk.StringVar(value="ultrafast")
        speeds = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium']
        speed_combo = ttk.Combobox(main_frame, textvariable=self.speed_var, values=speeds, state='readonly')
        speed_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Resolution Settings with Radio Buttons
        ttk.Label(main_frame, text="Recording Resolution:", font=('Helvetica', 10, 'bold')).pack(anchor='w', pady=(0, 5))
        self.resolution_var = tk.StringVar(value='1280x720')
        resolution_frame = ttk.Frame(main_frame)
        resolution_frame.pack(fill=tk.X, pady=(0, 10))
        
        resolutions = [
            ('Full HD (1920x1080)', '1920x1080'),
            ('HD (1280x720)', '1280x720'),
            ('SD (854x480)', '854x480')
        ]
        
        for text, value in resolutions:
            ttk.Radiobutton(
                resolution_frame,
                text=text,
                value=value,
                variable=self.resolution_var
            ).pack(anchor='w', padx=10)
        
        # Add info labels
        ttk.Label(main_frame, text="Note:", font=('Helvetica', 9, 'bold')).pack(anchor='w', pady=(10, 0))
        ttk.Label(main_frame, text="- Quality (CRF) is automatically adjusted\n"
                                 "- Faster encoding = Lower CPU usage but larger file size\n"
                                 "- Changes apply to new recordings only",
                 justify=tk.LEFT).pack(anchor='w')
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        ttk.Button(button_frame, text="Apply", command=self.apply_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Close", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        
    def apply_settings(self):
        try:
            # Validate inputs
            duration = float(self.duration_var.get())
            size = float(self.size_var.get())
            
            if duration <= 0 or size <= 0:
                raise ValueError("Invalid values")
            
            # Update settings for all cameras
            for camera in self.camera_manager.cameras.values():
                if hasattr(camera, 'recorder') and camera.recorder:
                    camera.recorder.max_segment_duration = duration * 60
                    camera.recorder.max_segment_size = size * 1024 * 1024 * 1024
                    camera.recorder.encoding_preset = self.speed_var.get()
                    
                    # Update resolution
                    width, height = map(int, self.resolution_var.get().split('x'))
                    camera.recording_width = width
                    camera.recording_height = height
            
            messagebox.showinfo("Success", "Settings updated successfully")
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid values")
            
    def reset_defaults(self):
        self.duration_var.set("15")
        self.size_var.set("2")
        self.speed_var.set("ultrafast")
        self.resolution_var.set("1280x720")

# Add this function after imports and before any classes
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    app = None
    shutdown_count = 0
    
    def signal_handler(signum, frame):
        global shutdown_count
        shutdown_count += 1
        
        if shutdown_count == 1:
            logging.info("Initiating graceful shutdown (Press Ctrl+C again to force)")
            if app:
                app.on_closing()
        elif shutdown_count >= 2:
            logging.warning("Forced shutdown initiated")
            os._exit(1)
    
    try:
        # Configure logging with timestamp
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create and run the application
        app = CameraApp()
        app.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Modify the mainloop to handle interrupts
        while True:
            try:
                app.update()
                time.sleep(0.01)  # Small delay to prevent high CPU usage
            except tk.TclError as e:
                if "application has been destroyed" in str(e):
                    break
                raise
            
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        sys.exit(1)
    finally:
        # Ensure all handlers are closed
        if app:
            try:
                app.destroy()
            except:
                pass
        logging.shutdown()
