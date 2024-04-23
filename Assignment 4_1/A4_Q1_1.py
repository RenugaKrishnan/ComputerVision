import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load camera matrices and distortion coefficients for left and right cameras
camera_matrix_left = np.loadtxt('left_camera_matrix.txt')
camera_matrix_right = np.loadtxt('right_camera_matrix.txt')

# Load rotation and translation matrices for stereo rectification
R_left = np.loadtxt('left_rotation_matrix.txt')
T_left = np.loadtxt('left_translation_vector.txt')
R_right = np.loadtxt('right_rotation_matrix.txt')
T_right = np.loadtxt('right_translation_vector.txt')

# Initialize the stereo camera
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

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

def estimate_distance(object_center, disparity_map):
    # Calculate depth from disparity map using stereo vision
    if object_center:
        center_x, center_y = object_center
        # Assuming disparity map is already obtained from stereo camera calibration
        depth = disparity_map[int(center_y), int(center_x)]
        # Convert disparity to depth using stereo calibration parameters
        depth /= 16.0  # Scale factor for OpenCV disparity map
        # Stereo rectification
        points_3d = cv2.triangulatePoints(R_left, R_right, np.array([[center_x], [center_y]]), np.array([[center_x], [center_y]]))
        depth = points_3d[2][0] / points_3d[3][0]  # Divide by homogeneous coordinate
        return depth
    return None

def stereo_camera():
    camera_left = cv2.VideoCapture(0)
    camera_right = cv2.VideoCapture(1)

    while True:
        # Capture frames from both left and right cameras
        ret1, frame_left = camera_left.read()
        ret2, frame_right = camera_right.read()

        # Perform object detection on the left frame
        frame_left, object_center, object_dimensions = detect_object(frame_left)

        # Calculate disparity map from stereo images
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        disparity_map = stereo.compute(gray_left, gray_right)

        # Estimate distance to object using disparity map
        distance = estimate_distance(object_center, disparity_map)

        # Display distance on left frame
        if distance is not None:
            cv2.putText(frame_left, f"Distance: {distance:.2f} cm", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert frame to JPEG format for streaming
        ret, jpeg = cv2.imencode('.jpg', frame_left)
        frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_left.release()
    camera_right.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(stereo_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
