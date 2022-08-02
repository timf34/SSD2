import matplotlib.pyplot as plt
import PIL as pil
import cv2
import numpy as np

from typing import List

video_file = r'rl-video-step-0-to-step-200.mp4'


def parse_video_to_frames(video_path: str) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    return frames


def display_frames(frames: List[np.ndarray]):
    """
    This is just to display the frames using matplotlib
    :param frames:
    :return:
    """
    for frame in frames:
        plt.imshow(frame)
        plt.show()


if __name__ == '__main__':
    frames = parse_video_to_frames(video_file)
    display_frames(frames)
