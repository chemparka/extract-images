import os
import cv2
import numpy as np
from nms import non_max_suppression_fast

def union(a,b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return (x, y, w, h)

def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return () # or (0,0,0,0) ?
    return (x, y, w, h)

def combineBoxes(boxes):
    noIntersectLoop = False
    noIntersectMain = False
    posIndex = 0
     # keep looping until we have completed a full pass over each rectangle
     # and checked it does not overlap with any other rectangle
    while noIntersectMain == False:
        noIntersectMain = True
        posIndex = 0
         # start with the first rectangle in the list, once the first 
         # rectangle has been unioned with every other rectangle,
         # repeat for the second until done
        while posIndex < len(boxes):
            noIntersectLoop = False
            while noIntersectLoop == False and len(boxes) > 1 and posIndex < len(boxes): #added posIndex < len(boxes) to prevent indexError
                a = boxes[posIndex]
                listBoxes = np.delete(boxes, posIndex, 0)
                index = 0
                for b in listBoxes:
                    #if there is an intersection, the boxes overlap
                    if intersection(a, b): 
                        newBox = union(a,b)
                        listBoxes[index] = newBox
                        boxes = listBoxes
                        noIntersectLoop = False
                        noIntersectMain = False
                        index = index + 1
                    else:                                                               #changed break to else
                        noIntersectLoop = True
                        index = index + 1
            posIndex = posIndex + 1

    return np.array(boxes).astype("int")
            
#loading images
img = cv2.imread("lang-99.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#preprocessing
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
gray = cv2.dilate(gray, kernel, iterations = 1)                     #dilate to remove text
gray = cv2.erode(gray, kernel, iterations = 2)                      #erode to restore dilation
ret, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_TOZERO)        #change white bg to blk
ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)      #invert binary image for easier processing

#try to fill images rectangles and remove noise
gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

#find contours and approximate to square
image, contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

idx = 0
rect = []
for c in contours:
    if cv2.contourArea(c) > 1500:
        rect.append(cv2.boundingRect(c))
        sorts = sorted(rect, key = lambda x: x[1]*3000+x[0])
        combine = combine_boxes(sorts)

for x,y,w,h in combine:
  idx += 1
  cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
  roi = img[y:y+h, x:x+w]                                         #save region of interests
  cv2.imwrite('roi-' + str(idx) + '.png', roi)


cv2.imshow('contours', img)
cv2.waitKey(0)

                   
                   


