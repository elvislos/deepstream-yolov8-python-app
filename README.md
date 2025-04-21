# Deepstream-yolov8-python-app

A comprehensive implementation of YOLOv8 object detection using both NVIDIA DeepStream and Hailo NPU acceleration. This project provides multiple approaches to video processing and object detection, including synchronous and asynchronous implementations.

## Project Structure

The project contains several implementations for video processing and object detection:

DeepStream Implementations

1. `scripts/deepstream_yolov8_detector_async_multithread.py`

   - Full YOLOv8 object detection pipeline using DeepStream
   - Asynchronous processing with multi-threading
   - Features bounding box detection, labels, and FPS counter
   - Optimized for real-time performance

2. `scripts/deepstream_video_stream_async.py`

   - Lightweight video streaming implementation
   - Asynchronous processing
   - FPS monitoring without object detection
   - Useful for testing video pipeline performance

3. `scripts/deepstream_video_stream_sync.py`

   - Synchronous (single-threaded) video streaming
   - Basic implementation for learning purposes
   - Simpler code structure for understanding DeepStream basics

### Hailo NPU Implementation

4. `scripts/hailo_yolov8s_detector_multithread.py`
   - YOLOv8s object detection using Hailo NPU
   - Multi-threaded design with separate threads for:
     - Video capture
     - Neural network inference
     - Post-processing
     - Rendering
   - Hardware-accelerated using Hailo NPU
   - Optimized for 640x640 input resolution

## Features

- Real-time object detection
- Multiple acceleration options (DeepStream/Hailo)
- FPS monitoring and display
- Support for video files and RTSP streams
- COCO dataset class labels (80 classes)
- Configurable confidence thresholds
- Multi-threaded processing for better performance

## Requirements

- NVIDIA GPU with DeepStream support
- Hailo NPU (for Hailo implementation)
- Python 3.x
- OpenCV
- NVIDIA DeepStream SDK
- Hailo Runtime Library (for Hailo implementation)

## Usage

### DeepStream Object Detection (Async Multi-threaded)

```bash
python3 scripts/deepstream_yolov8_detector_async_multithread.py <input_video_path> <config_file>
```

### DeepStream Video Streaming (Async)

```bash
python3 scripts/deepstream_video_stream_async.py <input_video_path>
```

### DeepStream Video Streaming (Sync)

```bash
python3 scripts/deepstream_video_stream_sync.py <input_video_path>
```

### Hailo NPU Object Detection

```bash
python3 scripts/hailo_yolov8s_detector_multithread.py
```

# Important Note on Performance Measurements

This project includes a Hailo NPU implementation that was originally developed by a colleague, demonstrating impressive performance metrics (500 FPS). While attempting to recreate their inference calculation structure in our DeepStream implementation (150 FPS), we discovered significant differences in how performance is measured and achieved. Here's why the numbers differ so dramatically:

## Performance Comparison Analysis

### 1. Hardware Architecture

- **Hailo NPU**: Dedicated neural processing unit specifically optimized for ML inference
- **Jetson AGX**: General-purpose GPU architecture handling multiple tasks

### 2. FPS Calculation Methodology

```python
# Hailo Implementation (Measures inference only):
fps = 1/((etime-stime)/100)  # Pure inference time
put_text_with_fringe(rendered_img, f'NPU {fps:.2f} FPS', (0,0))

# DeepStream Implementation (Measures full pipeline):
elapsed_avg = time.time() - self.start_time_avg  # Includes capture, processing, display
avg_fps = self.total_frames / elapsed_avg
```

### 3. Threading Architecture

```python
# Hailo's Efficient Multi-threading Approach:
th_capt = threading.Thread(target=capt, daemon=True)        # Capture
th_infer = threading.Thread(target=infer, daemon=True)      # Inference
th_postprocess = threading.Thread(target=postprocess)       # Post-processing
th_render = threading.Thread(target=render, daemon=True)    # Display
```

- Hailo separates each operation into dedicated threads
- DeepStream combines processing steps in fewer threads

### 4. Display Optimization

```python
# Hailo's Display Throttling:
if curr_time - last_submission_time > 1/30:  # 30 FPS display limit
    last_submission_time = curr_time
    render_img_q.put(rendered_img)
```

- Hailo runs inference at full speed while limiting display to 30 FPS
- This approach provides more accurate inference speed measurements

### 5. Implementation Stack

- DeepStream: Uses GStreamer pipeline with multiple elements, adding complexity and overhead
- Hailo: Direct hardware implementation with minimal software layers

## Understanding the Performance Gap

The reported performance difference (500 FPS vs 150 FPS) stems from measuring different aspects:

- **Hailo (500 FPS)**:

  - Measures pure neural network execution speed
  - Uses specialized hardware for single-task optimization
  - Separates display refresh rate from inference speed
  - Employs efficient multi-threading

- **DeepStream (150 FPS)**:

  - Measures complete video processing pipeline
  - Runs on versatile but less specialized hardware
  - Includes capture, inference, processing, and display overhead
  - Uses a more complex software stack
