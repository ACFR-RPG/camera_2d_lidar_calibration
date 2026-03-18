
import os
import argparse
import numpy as np
import cv2


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        print(os.path.join(folder,filename))
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def main():
    parser = argparse.ArgumentParser(description="Calibrate image intrinsics from a collection of checkerboard images.")
    parser.add_argument("image_dir", help="Image directory.")
    args = parser.parse_args()

    image_dir = args.image_dir

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

    objp = np.zeros((11*8, 3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:11].T.reshape(-1,2)*0.019
    # Arrays to store object points and image points from all the images.

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = load_images_from_folder(image_dir)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8,11), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (8,11), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(mtx)
    print(dist)

# Intrinsics: 
# [[519.26845842   0.         331.11197675]
#  [  0.         518.89359517 229.43433605]
#  [  0.           0.           1.        ]]
# Distortion: 
# [[ 0.11418155  0.19343114 -0.00268067  0.00371577 -1.09539701]]


if __name__ == '__main__':
    main()