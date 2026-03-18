
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

    # Object Points (objp) refers to the known points on an object, in this case a checkerboard
    # The known points are the checkerboard vertices (inside, excluding the out most corners)
    # Depending on your checkerboard layout, the parameters below willc change
    # TODO: Update the checkerboard parameters based on your own printed out board
    checkerboard_width = 11
    checkerboard_height = 8
    checkerboard_size = 0.19
    objp = np.zeros((checkerboard_width*checkerboard_height, 3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard_height, 0:checkerboard_width].T.reshape(-1,2)*checkerboard_size

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = load_images_from_folder(image_dir)

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (checkerboard_height, checkerboard_width), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            # Search in the grayscale image for the corners, refined sub pixel coorindates
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners for introspection - do the corners/vertices drawn back onto the image match with the checkerboard in the camera view?
            cv2.drawChessboardCorners(img, (checkerboard_height, checkerboard_width), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print(mtx)
    print(dist)

# Example Intrinsic: 
# [[519.26845842   0.         331.11197675]
#  [  0.         518.89359517 229.43433605]
#  [  0.           0.           1.        ]]
# Example Distortion: 
# [[ 0.11418155  0.19343114 -0.00268067  0.00371577 -1.09539701]]


if __name__ == '__main__':
    main()