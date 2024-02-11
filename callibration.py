# import cv2
# import numpy as np
# import pandas as pd
#
# # Load teh CSV file
#
# data = pd.read_csv('calibration.csv')
#
# # Extract pixel coordinates
# img_points = data[['x_px', 'y_px']].values.astype(float)
#
# # Create world coordinates (assuming Z is fixed at 0)
# world_points = data[['x_cm', 'y_cm']].assign(Z_cm=0).values.astype(float)
#
#
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_points, img_points, (640, 480), None, None, criteria=criteria)
import cv2
import numpy as np
import pandas as pd

# Load the CSV file
data = pd.read_csv('calibration.csv')

# Extract pixel coordinates
img_points = data[['x_px', 'y_px']].values.astype(float)

# Create world coordinates (assuming Z is fixed at 0)
world_points = data[['x_cm', 'y_cm']].assign(Z_cm=0).values.astype(float)

# Create a list of world coordinates for each image
world_points_list = [world_points] * len(img_points)

# Create a list of image coordinates for each image
img_points_list = [img_points] * len(world_points)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Perform intrinsic calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([world_points_list], [img_points_list], (640, 480), None, None, criteria=criteria)
