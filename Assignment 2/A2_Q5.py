from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

def find_homography(src_pts, dst_pts):
    """
    Estimate homography matrix using RANSAC algorithm.
    """
    assert len(src_pts) >= 4 and len(dst_pts) >= 4, "At least 4 corresponding points are required."

    # Compute homography matrix
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Ensure homography matrix is of the correct type and size
    if homography is not None and homography.shape == (3, 3):
        if homography.dtype != np.float32:
            homography = homography.astype(np.float32)
        return homography
    else:
        return None

def image_stitch(img1, img2):
    # Convert images to Gray
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # Initialize BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to select good matches
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)
    matches = np.asarray(good)

    if len(matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find Homography matrix using RANSAC
        homography = find_homography(src_pts, dst_pts)

        if homography is not None:
            # Warp image1 onto image2
            warped_img1 = cv2.warpPerspective(img1, homography, (img2.shape[1] + img1.shape[1], img2.shape[0]))

            # Combine images
            stitched_image = warped_img1.copy()
            stitched_image[0:img2.shape[0], 0:img2.shape[1]] = img2

            return stitched_image
        else:
            raise AssertionError("Failed to estimate homography matrix.")
    else:
        raise AssertionError("Can’t find enough keypoints for stitching.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file1' not in request.files or 'file2' not in request.files:
        return redirect(url_for('index'))

    file1 = request.files['file1']
    file2 = request.files['file2']

    if file1.filename == '' or file2.filename == '':
        return redirect(url_for('index'))

    img1 = cv2.imdecode(np.fromstring(file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.fromstring(file2.read(), np.uint8), cv2.IMREAD_COLOR)

    try:
        stitched = image_stitch(img1, img2)
        if stitched is not None:
            cv2.imwrite("static/stitched_image.jpg", stitched)
            return render_template('result.html')
        else:
            return render_template('error.html', message="Stitching failed. Please try again.")
    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)