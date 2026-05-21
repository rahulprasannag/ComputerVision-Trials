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
accumulated_weight = 0.5
ROItop = 20
ROIbottom= 300
ROIright = 300
ROIleft = 600


# function to find the accumalated average of the ROI (accumalated average)
def accumalated_avg(frame, accumulated_weight):
    
    # Grab the background
    global background
    
    # if running first time -> set background as first frame
    if background is None:
        background = frame.copy().astype("float")
        return None

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(frame, background, accumulated_weight)




# segment the hand from the background
def segment(frame, minThreshold=25):
    global background
    
    # calcualte the difference between background and current frame -> to detect hands
    difference = cv2.absdiff(background.astype("uint8"), frame)

    # get the hand and threshold it( convert hand to a binary black and white image)
    _ , thresholdedImage = cv2.threshold(difference, minThreshold, 255, cv2.THRESH_BINARY)

    # Grab contours -> we only want external contours
    contours, hierarchy = cv2.findContours(thresholdedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        # assuming hand is the largest object entering the ROI, get the area of the extreme contours
        # a curve line around the hand forming the boundry of the hand
        handSegment = max(contours, key=cv2.contourArea)
        
        return (thresholdedImage, handSegment)
    

# count fingers - draw the polygon and count using that
def count_fingers(thresholded, hand_segment):
    
    # calculating the convex hull (polygon) using the boundry of the hand(contour)
    convHull = cv2.convexHull(hand_segment)
    
    # getting the extremem points, these will result in TUPLES OF X,Y coordinates
    # extreme top point
    top    = tuple(convHull[convHull[:, :, 1].argmin()][0])

    # extreme bottom point
    bottom = tuple(convHull[convHull[:, :, 1].argmax()][0])

    # extreme left point
    left   = tuple(convHull[convHull[:, :, 0].argmin()][0])

    # extreme right point
    right  = tuple(convHull[convHull[:, :, 0].argmax()][0])

    # deriving the center of the hand using these points (concept is center will be mid point of top-bottom left-right)
    cX = (left[0] + right[0]) // 2
    cY = (top[1] + bottom[1]) // 2

    # distance of the left right bottom top points from the center, since these points are tupes we use pairwise thing to calculate
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[left, right, top, bottom])[0]
    
    maxDistance = distance.max()
    
    # Create a circle with 90% radius of the max euclidean distance
    radius = int(0.8 * maxDistance)
    circumference = (2 * np.pi * radius)

    # this roi is used to know if the fingertip points are closed or raised -- if the points are outside this circle then its raised
    # same size as the thresholded image
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
    cv2.circle(circular_roi, (cX, cY), radius, 255, 10)
    
    
    # make that circular as mask
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
    contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #finger count
    count = 0

    for cnt in contours:
        
        # Bounding box of countour
        (x, y, w, h) = cv2.boundingRect(cnt)

        # threshold to cut noise points of wrist
        wristThresht = ((cY + (cY * 0.25)) > (y + h))
        
        # threshold to cut noise points which are too far
        tooFar = ((circumference * 0.25) > cnt.shape[0])
        
        
        if  wristThresht and tooFar:
            count += 1

    return count


cam = cv2.VideoCapture(0)

# Intialize a frame count
num_frames = 0



while True:
    # get the current frame
    ret, frame = cam.read()

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    # Grab the ROI from the frame
    roi = frame[ROItop:ROIbottom, ROIright:ROIleft]

    # Apply grayscale and blur to ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # For the first 60 frames we will calculate the average of the background.
    # We will tell the user while this is happening
    if num_frames < 60:
        accumalated_avg(gray, accumulated_weight)
        if num_frames <= 59:
            cv2.putText(frame_copy, "WAIT! GETTING BACKGROUND AVG.", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Finger Count",frame_copy)
            
    else:
        
        # segment the hand region
        hand = segment(gray)

        if hand is not None:
            
            thresholded, hand_segment = hand

            cv2.drawContours(frame_copy, [hand_segment + (ROIright, ROItop)], -1, (255, 0, 0),1)

            # Count the fingers
            fingers = count_fingers(thresholded, hand_segment)
            cv2.putText(frame_copy, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Thesholded", thresholded)

    # Draw ROI Rectangle on frame copy
    cv2.rectangle(frame_copy, (ROIleft, ROItop), (ROIright, ROIbottom), (0,0,255), 5)

    num_frames += 1

    # Display the frame with segmented hand
    cv2.imshow("Finger Count", frame_copy)


    # Close windows with Esc
    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()