import cv2
import numpy as np

from sklearn.metrics import pairwise #for distance calculation - how many fingers are raised


# Global Variables
background = None
'''
    accumalated_weight is how sensitive should the ROI react(higher the sensitive),
    you cannot treat frame as background because there can be small changes like lighting and stuff,
    hence this ignores the small changes and includes newer frames as background --> used in accum_avg()
'''
accumalated_weight = 0.5
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
        # assuming hand is the largest object entering the ROI, get the area of the extreme contours
        # a curve line around the hand forming the boundry of the hand
        handSegment = max(contours, key=cv2.contourArea)

        return (thresholdedImage, handSegment)
    



# count fingers - draw the polygon and count using that
def countFingers(thresholdedImage, handSegment):

    # calculating the convex hull (polygon) using the boundry of the hand(contour)
    convHull = cv2.convexHull(handSegment)

    # getting the extremem points, these will result in TUPLES OF X,Y coordinates
    # extreme top point
    top = tuple(convHull[convHull[:,:,1].argmin()[0]])

    # extreme bottom point
    bottom = tuple(convHull[convHull[:,:,1].argmax()[0]])

    # extreme left point
    left = tuple(convHull[convHull[:,:,0].argmin()[0]])

    # extreme right point
    right = tuple(convHull[convHull[:,:,1].argmax()[0]])



    # deriving the center of the hand using these points (concept is center will be mid point of top-bottom left-right)
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    # distance of the left right bottom top points from the center, since these points are tupes we use pairwise thing to calculate
    distance = pairwise.euclidean_distances([cX,cY], [left, right, top, bottom])[0]

    maxDistance = distance.max()

    radius = int(0.8*maxDistance)
    circumference = (2*np.pi*radius)

    # this roi is used to know if the fingertip points are closed or raised -- if the points are outside this circle then its raised
    # same size as the thresholded image
    circular_roi = np.zeros(thresholdedImage[:2], dtype='uint8')
    cv2.circle(circular_roi, (cX,cY), radius, 255, 10)

    # make that circular as mask
    circular_roi = cv2.bitwise_and(thresholdedImage, thresholdedImage, mask = circular_roi)
    image, contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #finger count
    count = 0

    for cnt in contours:

        (x,y,w,h) = cv2.boundingRect(cnt)

        # threshold to cut noise points of wrist
        wristThresh = (cY + (cY*0.25)) > (y+h)

        # threshold to cut noise points which are too far
        tooFar = ((circumference*0.25) > cnt.shape[0])

        if wristThresh and tooFar:
            count += 1

        return count

