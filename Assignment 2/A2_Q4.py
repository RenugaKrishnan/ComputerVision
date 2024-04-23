from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_integral_image', methods=['POST'])
def calculate_integral_image():
    # Load the video
    video_path = "SampleVideo.mp4"
    if not os.path.exists(video_path):
        return render_template('error.html', message="Video file not found")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return render_template('error.html', message="Unable to open video file")

    # Read the first frame from the video
    ret, frame = cap.read()
    if not ret:
        return render_template('error.html', message="Unable to read frame from video")

    gray_clr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_clr.shape

    integral_image = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            integral_image[i][j] = int(gray_clr[i][j])

    for i in range(1, width):
        integral_image[0][i] += integral_image[0][i - 1]

    for j in range(1, height):
        integral_image[j][0] += integral_image[j - 1][0]

    for i in range(1, height):
        for j in range(1, width):
            integral_image[i][j] = integral_image[i - 1][j] + integral_image[i][j - 1] - integral_image[i - 1][j - 1] + gray_clr[i][j]

    np.savetxt('integral_matrix.txt', integral_image, fmt='%d')
    plt.plot(integral_image)
    plt.savefig("static/integral_image.jpg")  # Save image to 'static' folder
    plt.close()
    
    return render_template('result.html', image_path="static/integral_image.jpg")

if __name__ == '__main__':
    app.run(debug=True)
