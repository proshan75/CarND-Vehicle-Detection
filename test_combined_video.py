import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from moviepy.editor import VideoFileClip

import cv2
from lesson_functions import *

from test_video import *
from test_adv_lane_video import *

def process_image(img):
    lane_img = process_adv_lane_video_img(img)
    result_img = process_veh_detect_video_img(lane_img)

    return result_img

Output_video = 'output_project_combined_video_scaled.mp4'
Input_video = 'project_video.mp4'

clip1 = VideoFileClip(Input_video)
video_clip = clip1.fl_image(process_image)
video_clip.write_videofile(Output_video, audio=False)

