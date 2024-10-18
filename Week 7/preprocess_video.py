import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim


def save_frame(frame, count, output_dir):
    # Create the full path for saving the frame
    filename = os.path.join(output_dir, f'screenshot_{count:03d}.png')
    cv2.imwrite(filename, frame)
    print(f"Saved {filename}")


def is_frame_significantly_different(frame1, frame2, ssim_threshold=0.5):
    # Convert frames to grayscale for SSIM comparison
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two frames
    score, _ = ssim(gray_frame1, gray_frame2, full=True)

    # Return True if the SSIM score is below the threshold (indicating significant difference)
    return score < ssim_threshold


def capture_screenshots(video_path, custom_dir_name):
    # Create the output directory under DemoOutput
    output_dir = os.path.join('DemoOutput', custom_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()

    if not ret:
        print("Unable to read the video")
        return

    count = 0
    save_frame(prev_frame, count, output_dir)  # Save the first frame
    count += 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if the current frame is significantly different from the previous frame
        if is_frame_significantly_different(prev_frame, frame):
            save_frame(frame, count, output_dir)
            count += 1
            prev_frame = frame

    cap.release()
    cv2.destroyAllWindows()



# Example usage
video_path = 'DemoOutput/Demo outputs/1918.mp4'
dir_name = '1918'

capture_screenshots(video_path, dir_name)
