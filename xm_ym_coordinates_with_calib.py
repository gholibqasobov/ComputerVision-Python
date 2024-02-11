import cv2
import pickle
import numpy as np

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            "Workplace Coordinates"
            # x coordinate
            ref_x_cm = 25
            ref_x_px = 255
            x_cm = (cX * ref_x_cm) / ref_x_px - ref_x_cm

            # y coordinate

            ref_y_cm = 5
            ref_y_px = 63

            y_cm = (cY * ref_y_cm) / ref_y_px - ref_y_cm

            cv2.putText(image, 'ID: ' + str(markerID) + ' - ' + '({:.2f}, {:.2f})'.format(x_cm, y_cm) + str(
                (topLeft[0], topLeft[1])), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
                        thickness=2)


            # cv2.putText(image, str(markerID) + ' - ' + str((cX, cY)), (topLeft[0], topLeft[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0),
            #             thickness=2)
            # cv.putText(img, 'Hello', (255, 255), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (255, 0, 0), thickness=2)
            print("[Inference] ArUco marker ID: {}".format(markerID))

    return image

# Load calibration parameters
with open('C:\Projects\CV Robot Arm Control\CameraCalibration\calibration.pkl', 'rb') as f:
    calibration_data = pickle.load(f)

camera_matrix = calibration_data[0]
dist_coefficients = calibration_data[1]

# Define ArUco dictionary
aruco_type = "DICT_4X4_100"
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
arucoParams = cv2.aruco.DetectorParameters()

# Initialize the webcam
cap = cv2.VideoCapture("http://192.168.111.206:8080/video")

# Set the desired frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    # Capture a frame from the webcam
    ret, img = cap.read()

    # Resize the frame for better visualization (adjust the width as needed)
    width = 1000
    height = int(width * (img.shape[0] / img.shape[1]))
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    # Undistort the image using calibration parameters
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coefficients)

    # Detect ArUco markers in the undistorted image
    corners, ids, rejected = cv2.aruco.detectMarkers(undistorted_img, arucoDict, parameters=arucoParams)

    # Your existing code for displaying ArUco markers
    detected_markers = aruco_display(corners, ids, rejected, img)

    # ... (same as in the second file)
    # Define the size of the ArUco marker in meters
    aruco_marker_size = 0.04  # Change this value based on the actual size of your markers

    # Define the 3D coordinates of the ArUco marker corners in its local coordinate system
    object_points = np.array([
        [0, 0, 0],
        [aruco_marker_size, 0, 0],
        [aruco_marker_size, aruco_marker_size, 0],
        [0, aruco_marker_size, 0]
    ], dtype=np.float32)

    # Convert pixel coordinates to real-world coordinates
    if len(corners) > 0:
        # Assuming you have pixel coordinates of the first detected marker in 'pixel_x' and 'pixel_y'
        # pixel_coordinates = np.array([corners[0][0][0], corners[0][0][1]], dtype=np.float32).reshape(-1, 1, 2)
        # _, rvec, tvec = cv2.solvePnP(object_points, pixel_coordinates, camera_matrix, dist_coefficients)
        # real_world_coordinates, _ = cv2.projectPoints(pixel_coordinates, rvec, tvec, camera_matrix, dist_coefficients)

        # Assuming you have pixel coordinates of all four corners of the detected marker
        pixel_coordinates = corners[0][0].astype(float).reshape(-1, 1, 2)
        _, rvec, tvec = cv2.solvePnP(object_points, pixel_coordinates, camera_matrix, dist_coefficients)
        real_world_coordinates, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coefficients)
        cv2.putText(detected_markers, str(pixel_coordinates[0][0]), (255, 255), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (255, 0, 0), thickness=2)
        # Print or use the real-world coordinates as needed
        print("Real-world coordinates:", real_world_coordinates)



    # Display the modified image with ArUco markers
    cv2.imshow("Image", detected_markers)

    # Check for key press to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()
cap.release()
