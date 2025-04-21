import sys
import os
import time
import threading
import queue
import cv2
import numpy as np
import random
import ctypes
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import pyds

# Global abort flag
abort_flag = False

# YOLO labels and colors
yolo_labels = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
bbox_colors = [[random.randint(64, 255) for _ in range(3)] for _ in range(len(yolo_labels))]

class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        self.avg_fps = 0
        self.lock = threading.Lock()
        
    def update(self):
        with self.lock:
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed >= 1.0:  # Update FPS every second
                self.current_fps = self.frame_count / elapsed
                self.start_time = time.time()
                self.frame_count = 0
                
    def get_fps(self):
        with self.lock:
            return self.current_fps
            
    def update_avg_fps(self, total_frames, total_time):
        with self.lock:
            self.avg_fps = total_frames / total_time if total_time > 0 else 0

fps_counter = FPSCounter()

def put_text_with_fringe(img, text, origin, color=(0, 255, 0), scale=1.0, thickness=1, font=cv2.FONT_HERSHEY_PLAIN):
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, (0, 0, 0), thickness)
    cv2.putText(img, text, (origin[0], origin[1] + h + baseline), font, scale, color, max(1, thickness//2))
    return img

def nvds_to_bboxes(batch_meta):
    """Convert DeepStream metadata to our bbox format"""
    results = []
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    # Extract bounding box coordinates
                    rect_params = obj_meta.rect_params
                    top = int(rect_params.top)
                    left = int(rect_params.left)
                    width = int(rect_params.width)
                    height = int(rect_params.height)
                    x0, y0 = left, top
                    x1, y1 = left + width, top + height
                    
                    # Get class info
                    class_id = obj_meta.class_id
                    confidence = obj_meta.confidence
                    
                    # Add to results
                    results.append({
                        "box": [x0, y0, x1, y1],
                        "confidence": confidence,
                        "class_id": class_id,
                    })
                    
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            l_frame = l_frame.next
        except StopIteration:
            break
            
    return results

def render_result(img, bboxes, fps, avg_fps):
    """Render detection results with FPS info"""
    result_image = img.copy()
    for bbox in bboxes:
        x0, y0, x1, y1 = bbox["box"]
        conf = bbox["confidence"]
        clsid = bbox["class_id"]
        
        # Ensure class id is valid
        if clsid < 0 or clsid >= len(yolo_labels):
            continue
            
        class_label = yolo_labels[clsid]
        color = bbox_colors[clsid]

        cv2.rectangle(result_image, (x0, y0), (x1, y1), color, 2)
        text = f"{class_label} {conf:.2f}"
        t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, fontScale=1, thickness=1)[0]
        cv2.rectangle(result_image, (x0, y0), (x0 + t_size[0], y0 - t_size[1]), color, -1)
        cv2.putText(result_image, text, (x0, y0), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # Draw instant FPS on top left
    result_image = put_text_with_fringe(result_image, f"FPS: {fps:.2f}", (10, 10), (0, 255, 0), 2, 2)
    
    # Draw avg FPS on top right
    avg_text = f"Avg FPS: {avg_fps:.2f}"
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(avg_text, font, font_scale, thickness)
    x_pos = result_image.shape[1] - text_width - 10
    y_pos = 10
    result_image = put_text_with_fringe(result_image, avg_text, (x_pos, y_pos), (0, 255, 255), font_scale, thickness)
    
    return result_image

def surface_to_array(surface):
    """Convert NvBufSurface to numpy array"""
    import pyds
    import numpy as np
    
    # Get surface info
    surface_info = surface.surfaceList[0]
    
    # Calculate memory size and map memory
    memory_type = surface.memType
    memory_layout = surface_info.layout
    
    # For RGBA data (display format)
    if memory_layout == 3:  # NVBUF_LAYOUT_RGBA
        n_channels = 4
    else:
        n_channels = 3  # Assuming RGB or BGR
        
    # Get dimensions
    width = surface_info.width
    height = surface_info.height
    pitch = surface_info.pitch
    
    # Map memory to cpu and get array
    if memory_type == 0:  # NVBUF_MEM_CUDA_DEVICE
        # Use pyds.get_nvds_buf_surface_gpu() for CUDA memory
        frame_array = pyds.get_nvds_buf_surface_gpu(hash(surface))
    else:
        # For CPU-accessible memory
        byte_array = surface.surfaceList[0].mappedAddr
        if not byte_array:
            return None
            
        # Create a numpy array from mapped memory
        array = np.ctypeslib.as_array(ctypes.cast(byte_array, 
                                              ctypes.POINTER(ctypes.c_uint8)), 
                                  shape=(height, pitch))
                                  
        # Extract actual image data (pitch might be larger than width*n_channels)
        frame_array = array[:, :width*n_channels].reshape(height, width, n_channels)
    
    # Convert to BGR format for OpenCV
    if frame_array is not None and frame_array.shape[2] == 4:  # RGBA format
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
        
    return frame_array


Gst.init(None)

# TEMINAL COMMAND: python3 scripts/deepstream_yolov8_app_async.py videos/head-pose-face-detection-female-and-male.mp4
class SimpleDeepStreamPipeline:
    def __init__(self, input_uri):
        # Create the pipeline
        self.pipeline = Gst.Pipeline.new("simple-deepstream-pipeline")
        
        # Source element
        self.source_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
        self.source_bin.set_property("uri", input_uri)
        print("Source element created and URI set.")

        # Queue element
        self.queue = Gst.ElementFactory.make("queue", "queue")
        print("Queue element created.")

        # NVIDIA video converter
        self.nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
        print("Video converter element created.")

        # Force video to NV12 format
        self.caps1 = Gst.ElementFactory.make("capsfilter", "caps1")
        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=(string)NV12")
        self.caps1.set_property("caps", caps1)
        print("First caps filter created and set.")

        # Second NVIDIA converter to get from NVMM to CPU memory
        self.nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv2")
        print("Second video converter created.")

        # Caps to force BGRx format (CPU accessible)
        self.caps2 = Gst.ElementFactory.make("capsfilter", "caps2")
        caps2 = Gst.Caps.from_string("video/x-raw, format=(string)BGRx")
        self.caps2.set_property("caps", caps2)
        print("Second caps filter created and set.")

        # Appsink element to pull processed frames
        self.appsink = Gst.ElementFactory.make("appsink", "appsink")
        self.appsink.set_property("emit-signals", True)
        self.appsink.set_property("sync", False)
        self.appsink.set_property("max-buffers", 1)
        self.appsink.set_property("drop", True)
        print("Appsink element created.")

                # Add queues for multi-threaded processing
        self.frame_queue = queue.Queue(maxsize=16)     # Queue for raw frames
        self.display_queue = queue.Queue(maxsize=16)   # Queue for frames ready to display
        print("Queues created.")

        # Performance tracking variables
        self.total_frames = 0
        self.start_time_avg = time.time()
        print("Performance tracking variables created.")
        
        # Display throttling variables
        self.last_display_time = 0
        self.display_interval = 1.0/30.0  # 30 FPS maximum display rate
        print("Display throttling variables created.")
        
        # Initialize threading events
        self.stop_event = threading.Event()
        print("Threading events created.")
    
        # Check all elements are created
        elements = [self.source_bin, self.queue, self.nvvidconv, self.caps1, 
                    self.nvvidconv2, self.caps2, self.appsink]
        for element in elements:
            if not element:
                sys.stderr.write(f"Unable to create element: {element}\n")
                sys.exit(1)

        # Add elements to pipeline
        for element in elements:
            self.pipeline.add(element)
        print("All elements added to the pipeline.")

        # Link elements (except source_bin which will be linked on pad-added)
        self.queue.link(self.nvvidconv)
        self.nvvidconv.link(self.caps1)
        self.caps1.link(self.nvvidconv2)
        self.nvvidconv2.link(self.caps2)
        self.caps2.link(self.appsink)
        print("Elements linked.")

        # Connect the pad-added signal for dynamic linking
        self.source_bin.connect("pad-added", self.on_pad_added)
        print("Pad-added signal connected.")

        # Set up bus for pipeline messages
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.on_bus_message)
        print("Bus message handler connected.")

        # Connect the new-sample signal for frame processing
        self.appsink.connect("new-sample", self.on_new_sample)
        print("New-sample signal connected.")

    def on_pad_added(self, src, new_pad):
        print("New pad added:", new_pad.get_name())
        caps = new_pad.get_current_caps()
        struct = caps.get_structure(0)
        name = struct.get_name()
        
        if name.startswith("video/x-raw"):
            sink_pad = self.queue.get_static_pad("sink")
            if not sink_pad.is_linked():
                new_pad.link(sink_pad)
                print("New pad linked to queue.")
        else:
            print(f"Ignoring non-video pad: {name}")

    def on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("End-of-stream")
            self.pipeline.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, {debug}")
            self.pipeline.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending_state = message.parse_state_changed()
                print(f"Pipeline state changed from {old_state.value_nick} to {new_state.value_nick}")

    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.ERROR
        
        buffer = sample.get_buffer()
        caps = sample.get_caps()
        struct = caps.get_structure(0)
        width = struct.get_value('width')
        height = struct.get_value('height')
        
        # Map the buffer for reading
        result, map_info = buffer.map(Gst.MapFlags.READ)
        if result:
            # Create a numpy array from the buffer data
            frame_array = np.ndarray(
                shape=(height, width, 4),
                dtype=np.uint8,
                buffer=map_info.data
            )
            
            # Convert from BGRx to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_BGRA2BGR)
            
            # Put frame in queue for processing thread
            timestamp = time.time()
            try:
                self.frame_queue.put((frame_bgr.copy(), timestamp), block=False)
            except queue.Full:
                # Skip frame if queue is full
                pass
            
            # Free the buffer
            buffer.unmap(map_info)
        
        return Gst.FlowReturn.OK

    def process_frames_thread(self):
        """Process frames from the frame queue"""
        while not self.stop_event.is_set():
            try:
                # Get frame from queue
                frame, timestamp = self.frame_queue.get(timeout=0.1)
                
                if frame is None:
                    continue
                
                # Update FPS counter
                fps_counter.update()
                current_fps = fps_counter.get_fps()
                
                # Calculate average FPS
                self.total_frames += 1
                elapsed_avg = time.time() - self.start_time_avg
                avg_fps = self.total_frames / elapsed_avg if elapsed_avg > 0 else 0.0
                
                # Add FPS display to the frame (left side)
                processed_frame = put_text_with_fringe(
                    frame.copy(), 
                    f"FPS: {current_fps:.2f}", 
                    (10, 10), 
                    (0, 255, 0), 
                    2, 
                    2
                )
                
                # Add average FPS to the right side
                avg_text = f"Avg FPS: {avg_fps:.2f}"
                font = cv2.FONT_HERSHEY_PLAIN
                font_scale = 2
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(avg_text, font, font_scale, thickness)
                x_pos = frame.shape[1] - text_width - 10
                y_pos = 10
                processed_frame = put_text_with_fringe(
                    processed_frame, 
                    avg_text, 
                    (x_pos, y_pos), 
                    (0, 255, 255), 
                    font_scale, 
                    thickness
                )
                
                # Print frame information to terminal periodically
                if self.total_frames % 30 == 0:
                    print(f"Frame #{self.total_frames}, Current FPS: {current_fps:.2f}, Average FPS: {avg_fps:.2f}")
                
                # Put processed frame in display queue
                try:
                    self.display_queue.put((processed_frame, timestamp), block=False)
                except queue.Full:
                    # Skip frame if display queue is full
                    pass
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process thread: {e}")        

    def display_frames_thread(self):
        """Display frames with throttling"""
        while not self.stop_event.is_set():
            try:
                # Get processed frame from queue
                frame, timestamp = self.display_queue.get(timeout=0.1)
                
                if frame is None:
                    continue
                
                # Implement display throttling
                current_time = time.time()
                if current_time - self.last_display_time >= self.display_interval:
                    # Display the frame
                    cv2.imshow("DeepStream Output", frame)
                    key = cv2.waitKey(1)
                    if key in [27, ord('q'), ord('Q')]:  # ESC or q key
                        self.stop_event.set()
                        break
                    
                    self.last_display_time = current_time
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in display thread: {e}")

    def start(self):
        # Start processing threads
        self.process_thread = threading.Thread(target=self.process_frames_thread, daemon=True)
        self.display_thread = threading.Thread(target=self.display_frames_thread, daemon=True)
        
        self.process_thread.start()
        self.display_thread.start()
        
        # Set pipeline to PLAYING state
        self.pipeline.set_state(Gst.State.PLAYING)
        print("Pipeline and processing threads started.")

    def stop(self):
        # signal threads to stop
        self.stop_event.set()

        # Stop the pipeline
        self.pipeline.set_state(Gst.State.NULL)
        print("Pipeline stopped.")

        # Wait for threads to finish
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)
        if hasattr(self, 'display_thread'):
            self.display_thread.join(timeout=1.0)
        
        # Clear the queues
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        while not self.display_queue.empty():
            self.display_queue.get_nowait()

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <H264/MP4 file path or RTSP URI>")
        sys.exit(1)
    
    input_uri = sys.argv[1]
    if not input_uri.startswith("rtsp://") and not input_uri.startswith("file://"):
        # Add file:// prefix for local files if not present
        if os.path.exists(input_uri):
            input_uri = f"file://{os.path.abspath(input_uri)}"
    
    # Create the simplified pipeline
    pipeline = SimpleDeepStreamPipeline(input_uri)
    
    # Create a window for displaying the frame
    cv2.namedWindow("DeepStream Output", cv2.WINDOW_NORMAL)
   
    # Start the pipeline
    pipeline.start()
    
    try:
        # Run the main loop
        loop = GLib.MainLoop()
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the pipeline
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()