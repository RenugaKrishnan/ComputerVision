from flask import Flask, render_template
import numpy as np
from math import sqrt

app = Flask(__name__)

@app.route('/')
def home():
    # Load rotational matrix
    rot_matrix = np.loadtxt('rgb_rotation_matrix.txt')

    # Load translation vector
    tra_v = np.loadtxt('rgb_translation_vector.txt')

    # Load camera intrinsic matrix
    c_mtx = np.loadtxt('rgb_camera_matrix.txt')

    # Convert camera intrinsic matrix to homogeneous form
    int_mtx = np.append(np.append(c_mtx, [[0],[0],[1]], axis=1), [np.array([0,0,0,1])], axis=0)

    # Combine rotation matrix with translation vector to form the extrinsic matrix
    ext_mtx = np.hstack((rot_matrix, np.reshape(tra_v, (3, 1))))

    # Add the bottom row [0, 0, 0, 1] for homogeneous coordinates
    ext_mtx = np.vstack((ext_mtx, [0, 0, 0, 1]))

    # Calculate camera matrix
    camera_matrix = np.dot(int_mtx, ext_mtx)

    # Calculate the inverse of the camera matrix
    inverse_mat = -np.linalg.inv(camera_matrix)

    # Define the points in homogeneous coordinates [X, Y, Z, 1]
    project_points1 = np.array([[5],[10],[30],[1]])
    project_points2 = np.array([[100],[90],[30],[1]])

    # Transform points from camera frame to real-world frame
    real_dim_p1 = inverse_mat.dot(project_points1)
    real_dim_p2 = inverse_mat.dot(project_points2)

    # Calculate the distance between the two points
    dist = sqrt((real_dim_p2[0][0]-real_dim_p1[0][0])**2 +
                (real_dim_p2[1][0]-real_dim_p1[1][0])**2 +
                (real_dim_p2[2][0]-real_dim_p1[2][0])**2 )

    return render_template('index.html', real_dim_p1=real_dim_p1, real_dim_p2=real_dim_p2, dist=dist)

if __name__ == '__main__':
    app.run(debug=True)
