
import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
import os
import numpy as np
import glob
def main():
    # Initialize GStreamer
    Gst.init(None)
    


    # Get all images from calibration_images directory
    calibration_images = glob.glob('./calibration_images/*.jpg')
    if not calibration_images:
        sys.stderr.write("No calibration images found in ./calibration_images directory.\n")
        sys.exit(1)
    print(f"Found {len(calibration_images)} calibration images.")
    
    # Create Pipeline
    pipeline = Gst.Pipeline()

    # Create elements
    source = Gst.ElementFactory.make("multifilesrc", "file-source")
    decoder = Gst.ElementFactory.make("jpegdec", "jpeg-decoder")
    converter = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv")
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
    sink = Gst.ElementFactory.make("fakesink", "fakesink")
    
    if not all([pipeline, source, decoder, converter, streammux, pgie, sink]):
        sys.stderr.write(" One element could not be created. Exiting.\n")
        sys.exit(1)
    
    # Set properties
    source.set_property('location', './calibration_images/%*.jpg')  # Will match all jpg files
    source.set_property('loop', False)  # Don't loop through images
    streammux.set_property('width', 640)  # YOLOv8 input size
    streammux.set_property('height', 640)  # YOLOv8 input size
    streammux.set_property('batch-size', 1)
    pgie.set_property('config-file-path', "config_infer_primary_yolov8.txt")
    
    # Add elements to pipeline
    pipeline.add(source)
    pipeline.add(decoder)
    pipeline.add(converter)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(sink)
    
    # Link elements
    source.link(decoder)
    decoder.link(converter)
    
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = converter.get_static_pad("src")
    srcpad.link(sinkpad)
    
    streammux.link(pgie)
    pgie.link(sink)
    
    print("Starting calibration process...")
    # Start playing
    pipeline.set_state(Gst.State.PLAYING)
    
    try:
        loop = GObject.MainLoop()
        
        def on_eos(bus, message):
            print("Finished processing all calibration images")
            loop.quit()
            return True
            
        # Listen for EOS (End of Stream)
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message::eos", on_eos)
        
        print("Processing calibration images...")
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Clean up
        print("Cleaning up...")
        pipeline.set_state(Gst.State.NULL)
        print("Calibration complete. Check for calib.table file.")

if __name__ == '__main__':
    main()