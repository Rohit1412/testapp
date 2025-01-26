import os
import cv2
from quart import Quart, websocket, render_template, jsonify, request
import asyncio
from multiprocessing import Process, shared_memory, freeze_support, Value
import numpy as np
import time
from datetime import datetime
import logging
from vidgear.gears import CamGear
import socket
from contextlib import closing
import re

# Move app creation to top, after imports
app = Quart(__name__)

# Protocol patterns
PROTOCOL_PATTERNS = {
    'rtsp': r'^rtsp://',
    'rtmp': r'^rtmp://',
    'http': r'^http://',
    'https': r'^https://',
    'udp': r'^udp://',
    'tcp': r'^tcp://',
    'ip': r'^(?:\d{1,3}\.){3}\d{1,3}'
}

def determine_camera_type(source):
    """Determine the type of camera/stream from the source."""
    logging.info(f"Attempting to determine camera type for source: {source}")
    
    # Handle integer or string number for USB cameras
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        try:
            source_int = int(source)
            logging.info(f"Testing USB camera at index {source_int}")
            
            # Test if camera is accessible
            cap = cv2.VideoCapture(source_int)
            if cap.isOpened():
                # Get camera properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = cap.get(cv2.CAP_PROP_FPS)
                logging.info(f"Successfully opened camera. Properties: width={width}, height={height}, fps={fps}")
                
                cap.release()
                return 'usb', source_int  # Return both type and source
            else:
                available_cameras = []
                # Try to find available cameras
                for i in range(10):
                    temp_cap = cv2.VideoCapture(i)
                    if temp_cap.isOpened():
                        available_cameras.append(i)
                        temp_cap.release()
                
                if available_cameras:
                    logging.error(f"Camera index {source} not available. Available cameras: {available_cameras}")
                else:
                    logging.error("No cameras found on system")
                raise ValueError(f"Unable to open USB camera at index {source}")
        except Exception as e:
            logging.error(f"Error accessing USB camera: {str(e)}")
            raise ValueError(f"Error accessing USB camera: {str(e)}")
    
    # Handle URL-based sources
    source_str = str(source).lower()
    for protocol, pattern in PROTOCOL_PATTERNS.items():
        if re.match(pattern, source_str):
            return protocol, source  # Return both type and source
    
    raise ValueError(f"Unsupported camera source: {source}")

class CameraCapture:
    """Factory class to create appropriate camera capture instance"""
    @staticmethod
    def create_capture(source):
        camera_type, source = determine_camera_type(source)  # Unpack both values
        
        if camera_type == 'usb':
            # Convert to int for USB cameras
            source = int(source) if isinstance(source, str) else source
            capture = cv2.VideoCapture(source)
            if not capture.isOpened():
                raise ValueError(f"Failed to open USB camera at index {source}")
            return capture, camera_type  # Return both capture and type
            
        elif camera_type in ['rtmp', 'http', 'https', 'ip']:
            return CamGear(source=source).start(), camera_type
            
        elif camera_type in ['rtsp', 'udp', 'tcp']:
            return cv2.VideoCapture(source), camera_type
            
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")

    @staticmethod
    def read_frame(capture, camera_type):
        """Read a frame from the capture based on camera type"""
        if camera_type in ['rtmp', 'http', 'https', 'ip']:
            return capture.read()
        else:  # OpenCV capture
            ret, frame = capture.read()
            return frame if ret else None

class CameraManager:
    def __init__(self):
        self.camera_id_counter = 0
        self.lock = asyncio.Lock()
    
    async def generate_camera_id(self):
        async with self.lock:
            camera_id = self.camera_id_counter
            self.camera_id_counter += 1
            return camera_id

# Global constants
RECORDING_DIR = 'recordings'

def cleanup_shared_memory(camera_id):
    """Clean up any existing shared memory for a camera ID"""
    try:
        # Try to clean up frame shared memory
        try:
            shm = shared_memory.SharedMemory(name=f"camera_frame_{camera_id}")
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

        # Try to clean up status shared memory
        try:
            shm = shared_memory.SharedMemory(name=f"camera_status_{camera_id}")
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
    except Exception as e:
        logging.warning(f"Error cleaning up shared memory: {e}")

class AsyncCamera:
    def __init__(self, source, max_fps=20, name="default"):
        self.source = source
        self.max_fps = max_fps
        self.name = name
        self.camera_id = None  # Will be set after creation
        
        # Don't create shared memory here - move to setup method
        self.frame_shm = None
        self.status_shm = None
        self.frame_array = None
        self.status_array = None
        self.recording_dir = None
        
        # Add these attributes
        self.recording_process = None
        self.is_recording = False
        self.current_recording = None
        
    def setup(self, camera_id):
        """Initialize shared memory after camera_id is assigned"""
        self.camera_id = camera_id
        
        # Clean up any existing shared memory first
        cleanup_shared_memory(camera_id)
        
        # Create unique shared memory for this camera
        self.frame_shm_name = f"camera_frame_{camera_id}"
        self.status_shm_name = f"camera_status_{camera_id}"
        
        try:
            # Initialize frame shape and shared memory
            self.frame_shape = (480, 640, 3)
            self.frame_size = np.prod(self.frame_shape) * np.dtype(np.uint8).itemsize
            self.frame_shm = shared_memory.SharedMemory(
                name=self.frame_shm_name, 
                create=True, 
                size=self.frame_size
            )
            
            # Status: [is_recording, fps, running, width, height]
            self.status_shape = (5,)
            self.status_shm = shared_memory.SharedMemory(
                name=self.status_shm_name,
                create=True,
                size=self.status_shape[0] * np.dtype(np.float32).itemsize
            )
            
            self.frame_array = np.ndarray(
                self.frame_shape, 
                dtype=np.uint8, 
                buffer=self.frame_shm.buf
            )
            self.status_array = np.ndarray(
                self.status_shape,
                dtype=np.float32,
                buffer=self.status_shm.buf
            )
            
            self.recording_dir = os.path.join(RECORDING_DIR, f"{self.name}_{camera_id}")
            os.makedirs(self.recording_dir, exist_ok=True)
        except Exception as e:
            # Clean up if initialization fails
            cleanup_shared_memory(camera_id)
            raise e

    async def start(self):
        self.status_array[2] = 1.0  # Set running flag
        self.capture_process = Process(
            target=self._capture_loop,
            args=(self.source, self.max_fps)
        )
        self.capture_process.start()

    def _capture_loop(self, source, max_fps):
        capture = None
        consecutive_failures = 0
        max_failures = 10
        
        try:
            capture, camera_type = CameraCapture.create_capture(source)  # Get both values
            logging.info(f"Camera initialized: type={camera_type}, source={source}")
            
            last_frame_time = time.time()
            frame_interval = 1.0 / max_fps

            while self.status_array[2] > 0:
                current_time = time.time()
                if current_time - last_frame_time >= frame_interval:
                    frame = CameraCapture.read_frame(capture, camera_type)
                    
                    if frame is not None:
                        consecutive_failures = 0
                        if frame.shape != self.frame_shape:
                            frame = cv2.resize(frame, 
                                            (self.frame_shape[1], self.frame_shape[0]))
                        self.frame_array[:] = frame
                        self.status_array[1] = 1.0 / (current_time - last_frame_time)
                        last_frame_time = current_time
                    else:
                        consecutive_failures += 1
                        logging.warning(f"Failed to read frame. Attempt {consecutive_failures}/{max_failures}")
                        if consecutive_failures >= max_failures:
                            raise ValueError("Too many consecutive failures reading frames")
                        time.sleep(0.1)

        except Exception as e:
            logging.error(f"Error in capture loop: {str(e)}")
            self.status_array[2] = 0.0  # Signal process to stop
        finally:
            if capture is not None:
                if camera_type in ['rtmp', 'http', 'https', 'ip']:
                    capture.stop()
                else:
                    capture.release()

    async def start_recording(self):
        """Start recording the camera feed"""
        if self.is_recording:
            return False, "Already recording"
            
        try:
            # Create date-based subdirectory
            date_dir = datetime.now().strftime('%Y-%m-%d')
            timestamp = datetime.now().strftime('cam_%H-%M-%S.mp4')
            
            # Create full directory path
            recording_subdir = os.path.join(self.recording_dir, date_dir)
            os.makedirs(recording_subdir, exist_ok=True)
            
            # Full path for the output file
            output_path = os.path.join(recording_subdir, timestamp)
            
            # Create shared memory for recording status
            self.recording_status_shm = shared_memory.SharedMemory(
                create=True,
                size=np.dtype(np.bool_).itemsize,
                name=f"recording_status_{self.camera_id}"
            )
            self.recording_status = np.ndarray(
                (1,), dtype=np.bool_, buffer=self.recording_status_shm.buf
            )
            self.recording_status[0] = True
            
            # Start recording process
            self.recording_process = Process(
                target=self._record_video,
                args=(
                    output_path,
                    self.frame_shm_name,
                    self.frame_shape,
                    f"recording_status_{self.camera_id}"
                )
            )
            self.recording_process.start()
            
            # Update status
            self.is_recording = True
            self.status_array[0] = 1.0
            self.current_recording = output_path
            
            return True, f"Recording started: {output_path}"
        except Exception as e:
            logging.error(f"Failed to start recording: {str(e)}")
            if hasattr(self, 'recording_status_shm'):
                self.recording_status_shm.close()
                self.recording_status_shm.unlink()
            return False, str(e)

    def _record_video(self, output_path, frame_shm_name, frame_shape, recording_status_name):
        """Recording process function"""
        frame_shm = None
        recording_status_shm = None
        out = None
        
        try:
            # Reconnect to shared memory in the new process
            frame_shm = shared_memory.SharedMemory(name=frame_shm_name)
            recording_status_shm = shared_memory.SharedMemory(name=recording_status_name)
            
            frame_array = np.ndarray(frame_shape, dtype=np.uint8, buffer=frame_shm.buf)
            recording_status = np.ndarray((1,), dtype=np.bool_, buffer=recording_status_shm.buf)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.max_fps, 
                                (frame_shape[1], frame_shape[0]))

            while recording_status[0]:  # Check recording status
                try:
                    # Make a copy of the frame to avoid any race conditions
                    frame = np.copy(frame_array)
                    if frame is not None and frame.size > 0:
                        out.write(frame)
                    time.sleep(1.0 / self.max_fps)
                except Exception as e:
                    logging.error(f"Error writing frame: {str(e)}")
                    break
                    
        except Exception as e:
            logging.error(f"Recording process error: {str(e)}")
        finally:
            # Clean up resources
            if out is not None:
                out.release()
            if frame_shm is not None:
                frame_shm.close()
            if recording_status_shm is not None:
                recording_status_shm.close()

    async def stop_recording(self):
        """Stop the current recording"""
        if not self.is_recording:
            return False, "Not recording"
            
        try:
            # Signal recording process to stop
            if hasattr(self, 'recording_status'):
                self.recording_status[0] = False
            
            if self.recording_process:
                # Give the process a moment to finish cleanly
                await asyncio.sleep(0.5)
                
                # Terminate if still running
                if self.recording_process.is_alive():
                    self.recording_process.terminate()
                    self.recording_process.join(timeout=2.0)
                
                self.recording_process = None
            
            # Clean up recording status shared memory
            if hasattr(self, 'recording_status_shm'):
                self.recording_status_shm.close()
                self.recording_status_shm.unlink()
                delattr(self, 'recording_status_shm')
                delattr(self, 'recording_status')
            
            self.is_recording = False
            recorded_file = self.current_recording
            self.current_recording = None
            self.status_array[0] = 0.0
            
            return True, f"Recording stopped: {recorded_file}"
        except Exception as e:
            logging.error(f"Failed to stop recording: {str(e)}")
            return False, str(e)

    async def stop(self):
        """Stop the camera and clean up resources"""
        try:
            # Stop recording if active
            if self.is_recording:
                await self.stop_recording()
            
            # Stop the capture process
            if hasattr(self, 'status_array') and self.status_array is not None:
                self.status_array[2] = 0.0
            
            if hasattr(self, 'capture_process'):
                self.capture_process.terminate()
                self.capture_process.join(timeout=2.0)
            
            # Clean up shared memory
            if hasattr(self, 'frame_shm'):
                self.frame_shm.close()
                self.frame_shm.unlink()
            if hasattr(self, 'status_shm'):
                self.status_shm.close()
                self.status_shm.unlink()
                
            return True
        except Exception as e:
            logging.error(f"Error stopping camera: {str(e)}")
            return False

@app.websocket('/stream/<int:camera_id>')
async def stream(camera_id):
    camera = camera_registry.get_camera(camera_id)
    if not camera:
        return
    
    try:
        while camera.status_array[2] > 0:
            frame = np.copy(camera.frame_array)
            _, buffer = cv2.imencode('.jpg', frame)
            await websocket.send(buffer.tobytes())
            await asyncio.sleep(1/30)
    except Exception as e:
        logging.error(f"Streaming error for camera {camera_id}: {e}")

class CameraRegistry:
    def __init__(self):
        self._counter = Value('i', 0)
        self._cameras = {}  # Regular dict for camera instances
        self._lock = asyncio.Lock()

    async def add_camera(self, camera):
        async with self._lock:
            camera_id = self._counter.value
            self._counter.value += 1
            self._cameras[camera_id] = camera
            return camera_id

    async def remove_camera(self, camera_id):
        async with self._lock:
            if camera_id in self._cameras:
                camera = self._cameras.pop(camera_id)
                await camera.stop()
                return True
            return False

    def get_camera(self, camera_id):
        return self._cameras.get(camera_id)

    def list_cameras(self):
        return list(self._cameras.items())

# Replace global variables with registry
camera_registry = CameraRegistry()

@app.route('/add_camera', methods=['POST'])
async def add_camera():
    try:
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'message': 'Request must be JSON'
            }), 400

        data = await request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Empty request body'
            }), 400

        source = data.get('source')
        if source is None:
            return jsonify({
                'status': 'error',
                'message': 'Camera source is required'
            }), 400

        name = data.get('name', 'default')
        max_fps = data.get('max_fps', 20)
        
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        # Create camera without ID first
        camera = AsyncCamera(source, max_fps, name)
        # Get ID from registry
        camera_id = await camera_registry.add_camera(camera)
        # Setup camera with assigned ID
        camera.setup(camera_id)
        # Start the camera
        await camera.start()
        
        return jsonify({
            'status': 'success',
            'camera_id': camera_id,
            'message': f'Camera {name} added successfully'
        })
    except Exception as e:
        logging.error(f"Failed to add camera: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/remove_camera/<int:camera_id>', methods=['POST'])
async def remove_camera(camera_id):
    """Remove a camera from the registry"""
    try:
        if await camera_registry.remove_camera(camera_id):
            return jsonify({
                'status': 'success',
                'message': f'Camera {camera_id} removed successfully'
            })
        return jsonify({
            'status': 'error',
            'message': f'Camera {camera_id} not found'
        }), 404
    except Exception as e:
        logging.error(f"Error removing camera: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/list_cameras', methods=['GET'])
async def list_cameras():
    cameras = camera_registry.list_cameras()
    return jsonify({
        'status': 'success',
        'cameras': [{
            'id': cid,
            'name': camera.name,
            'source': camera.source,
            'fps': float(camera.status_array[1]),
            'is_recording': bool(camera.status_array[0])
        } for cid, camera in cameras]
    })

@app.route('/')
async def index():
    return await render_template('index.html')

@app.route('/start_recording', methods=['POST'])
async def start_recording_route():
    """Start recording for a camera specified in request body"""
    try:
        data = await request.get_json()
        camera_id = data.get('camera_id')
        
        if camera_id is None:
            return jsonify({
                'status': 'error',
                'message': 'Camera ID is required'
            }), 400
            
        camera = camera_registry.get_camera(camera_id)
        if not camera:
            return jsonify({
                'status': 'error',
                'message': f'Camera {camera_id} not found'
            }), 404
            
        success, message = await camera.start_recording()
        return jsonify({
            'status': 'success' if success else 'error',
            'message': message
        }), 200 if success else 500
        
    except Exception as e:
        logging.error(f"Error in start_recording: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/stop_recording', methods=['POST'])
async def stop_recording_route():
    """Stop recording for a camera specified in request body"""
    try:
        data = await request.get_json()
        camera_id = data.get('camera_id')
        
        if camera_id is None:
            return jsonify({
                'status': 'error',
                'message': 'Camera ID is required'
            }), 400
            
        camera = camera_registry.get_camera(camera_id)
        if not camera:
            return jsonify({
                'status': 'error',
                'message': f'Camera {camera_id} not found'
            }), 404
            
        success, message = await camera.stop_recording()
        return jsonify({
            'status': 'success' if success else 'error',
            'message': message
        }), 200 if success else 500
        
    except Exception as e:
        logging.error(f"Error in stop_recording: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/recordings/<int:camera_id>', methods=['GET'])
async def list_recordings(camera_id):
    camera = camera_registry.get_camera(camera_id)
    if camera:
        recordings = []
        try:
            for file in os.listdir(camera.recording_dir):
                if file.endswith('.mp4'):
                    file_path = os.path.join(camera.recording_dir, file)
                    recordings.append({
                        'filename': file,
                        'size': os.path.getsize(file_path),
                        'created': datetime.fromtimestamp(
                            os.path.getctime(file_path)
                        ).isoformat()
                    })
            return jsonify({
                'status': 'success',
                'recordings': recordings
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error listing recordings: {str(e)}'
            }), 500
    return jsonify({
        'status': 'error',
        'message': 'Camera not found'
    }), 404

@app.route('/check_cameras', methods=['GET'])
async def check_cameras():
    """Endpoint to check available cameras"""
    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            properties = {
                'index': i,
                'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'fps': cap.get(cv2.CAP_PROP_FPS)
            }
            available_cameras.append(properties)
            cap.release()
    
    return jsonify({
        'status': 'success',
        'available_cameras': available_cameras
    })

def create_app():
    # Simplify create_app to avoid manager issues
    logging.basicConfig(level=logging.WARNING)
    os.makedirs(RECORDING_DIR, exist_ok=True)
    return app

def find_free_port(start_port=5000):
    """Find a free port starting from start_port"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        port = start_port
        while port < 65535:
            try:
                s.bind(('', port))
                return port
            except OSError:
                port += 1
        raise RuntimeError("No free ports available")

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "localhost"

def main():
    try:
        port = find_free_port()
        app = create_app()
        local_ip = get_local_ip()
        print(f"Starting server on http://{local_ip}:{port}")
        app.run(host='0.0.0.0', port=port)  # Changed host to '0.0.0.0' to allow external access
    except Exception as e:
        print(f"Failed to start server: {e}")
        import sys
        sys.exit(1)

# Optional: Add a cleanup function to ensure proper shutdown
def cleanup_on_exit():
    # Clean up all cameras
    for camera_id in list(camera_registry._cameras.keys()):
        try:
            asyncio.run(camera_registry.remove_camera(camera_id))
        except Exception as e:
            logging.error(f"Error cleaning up camera {camera_id}: {e}")
        
        # Clean up shared memory
        cleanup_shared_memory(camera_id)


if __name__ == '__main__':
    import atexit
    atexit.register(cleanup_on_exit)
    freeze_support()
    main()