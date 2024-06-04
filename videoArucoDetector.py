import cv2
from cv2 import aruco
import numpy as np
import time

def detect_aruco_markers_in_video(video_path, output_path='output.mp4', dictionary=aruco.DICT_6X6_250, display_width=400, save_video=False):
    """
    Detect ArUco markers in a video and display the original and processed frames side by side,
    with lines connecting the center bottom of the screen to the centers of detected markers.
    Optionally save the processed video to a specified output file.

    Parameters:
    - video_path: str, path to the input video file.
    - output_path: str, path to the output video file (used only if save_video is True).
    - dictionary: aruco dictionary type (default is aruco.DICT_6X6_250).
    - display_width: int, the width to which the video frames should be resized (default is 800 pixels).
    - save_video: bool, whether to save the processed video (default is False).
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    # Get the original video's properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the new dimensions
    aspect_ratio = frame_height / frame_width
    new_width = display_width
    new_height = int(new_width * aspect_ratio)
    
    # Initialize the VideoWriter object if save_video is True
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (2 * new_width, new_height))  # Size for concatenated frames

    # Load the specified dictionary
    aruco_dict = aruco.getPredefinedDictionary(dictionary)
    
    # Initialize the detector parameters using default values
    parameters = aruco.DetectorParameters()

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the markers
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        
        # Draw detected markers on a copy of the original frame
        processed_frame = frame.copy()
        if ids is not None:
            processed_frame = aruco.drawDetectedMarkers(processed_frame, corners, ids)

            # Calculate the center of each detected marker
            centers = []
            for corner in corners:
                center = np.mean(corner[0], axis=0).astype(int)
                centers.append(center)
            
            # Define the start point at the center bottom of the frame
            height, width, _ = frame.shape
            start_point = (width // 2, height)

            # Draw lines connecting the start point to the centers
            if centers:
                cv2.line(processed_frame, start_point, tuple(centers[0]), (0, 255, 0), 2)
                for i in range(len(centers) - 1):
                    cv2.line(processed_frame, tuple(centers[i]), tuple(centers[i + 1]), (0, 255, 0), 2)

        # Resize the frames to fit the display width
        resized_frame = cv2.resize(frame, (new_width, new_height))
        resized_processed_frame = cv2.resize(processed_frame, (new_width, new_height))

        # Concatenate the original and processed frames side by side
        combined_frame = np.hstack((resized_frame, resized_processed_frame))

        # Write the combined frame to the output video if save_video is True
        if save_video:
            out.write(combined_frame)

        # Display the combined frame
        cv2.imshow('Original and Detected ArUco markers', combined_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

##        time.sleep(0.06)

    # Release the video capture and writer, and close windows
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Path to the input video
    video_path = 'farm2.mp4'
    # Path to the output video
    output_path = 'processed_farm2.mp4'
    detect_aruco_markers_in_video(video_path, output_path, save_video=True)
