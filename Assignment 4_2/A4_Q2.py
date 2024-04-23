import cv2
import depthai as dai
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

def detect_object(frame):
    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for the color of the object (e.g., red)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    
    # Threshold the HSV image to get only red colors
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area to get the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Draw bounding box on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Calculate center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        
        return frame, (center_x, center_y), (w, h)
    
    return frame, None, None

def estimate_distance(object_center, image_width):
    # Assuming fixed object width and focal length (for simplicity)
    object_width = 10  # Width of object in centimeters
    focal_length = 100  # Focal length in pixels (example value)

    # Calculate distance using simple formula: distance = (focal_length * object_width) / object_width_pixels
    if object_center:
        object_width_pixels = object_center[0] * 2  # Assuming object covers half of the image width
        distance_cm = (focal_length * object_width) / object_width_pixels
        return distance_cm
    return None

def stereo_camera():
    # Pipeline
    pipeline = dai.Pipeline()

    # Define a ColorCamera node
    cam_rgb = pipeline.createColorCamera()
    cam_rgb.setPreviewSize(300, 300)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    # Define a NeuralNetwork node to run object detection
    nn = pipeline.createMobileNetDetectionNetwork()
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath("mobilenet-ssd_openvino_2021.2_6shave.blob")
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    cam_rgb.preview.link(nn.input)

    # Define outputs
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    nn.passthrough.link(xout_rgb.input)

    # Connect to the device
    with dai.Device(pipeline) as device:
        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()

            # Perform object detection
            frame, object_center, object_dimensions = detect_object(frame)

            # Estimate distance
            distance = estimate_distance(object_center, frame.shape[1])

            # Display distance on frame
            if distance is not None:
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Convert frame to JPEG format for streaming
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stereo_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)