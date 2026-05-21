import cv2
import numpy as np

from sklearn.metrics import pairwise #for distance calculation - how many fingers are raised


# Global Variables
background = None
accumalated_weight = 0.5        # how sensitive should the ROI react(higher the sensitive), 
#                                 you cannot treat frame as background because there can be small changes like lighting and stuff,  
#                                 hence this ignores the small changes and includes newer frames as background --> used in accum_avg()

ROItop = 20
ROIbottom= 300
ROIright = 300
ROIleft = 600


# function to find the accumalated average of the ROI (accumalated average)
def accumalated_avg(frame, accumalated_weight):

    global background


    #if running first time -> set background as first frame
    if background is None:
        background = frame.copy().astype('float')
        return
    
    cv2.accumulateWeighted(frame, background, accumalated_weight)




# segment the hand from the background
def segment(frame, minThreshold=25):

    # calcualte the difference between background and current frame -> to detect hands
    difference = cv2.absdiff(background.astype('uint8'), frame)

    # get the hand and threshold it( convert hand to a binary black and white image)
    ret, thresholdedImage = cv2.threshold(difference, minThreshold, 255, cv2.THRESH_BINARY)


    #grab contours -> we only want external contours
    image, contours, hierarchy = cv2.findContours(thresholdedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    
    else:
        # assuming hand is the largest object entering the ROI
        hand_segment = max(contours, key=cv2.contourArea)

        return (thresholdedImage, hand_segment)