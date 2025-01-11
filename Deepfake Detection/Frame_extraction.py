#import all the necessary libraries
import cv2
import tempfile

#function to extract frames from the video
def extract_frames(video_file):
    frames = []
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(video_file.read())
    cap = cv2.VideoCapture(temp_video.name)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames