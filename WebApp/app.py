from flask import Flask, render_template
from utils import camera_matrix_calculation, integral_image_feed, image_stitching

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera_matrix_calculation')
def camera_matrix():
    return camera_matrix_calculation.calculate()

@app.route('/integral_image_feed')
def integral_image():
    return integral_image_feed.compute()

@app.route('/image_stitching')
def image_stitch():
    return image_stitching.stitch()

if __name__ == '__main__':
    app.run(debug=True)
