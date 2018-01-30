import cv2
import os
import numpy as np
import itertools
import argparse
import logging

log = logging.getLogger()
log.setLevel(logging.DEBUG)
saveLog = logging.FileHandler('log.log')
saveLog.setLevel(logging.DEBUG)
log.addHandler(saveLog)

def loadInput(path):

    imagePaths = []
    if not os.path.isdir(path):
        raise IOError("The folder " + path + " doesn't exist.")
    
    for root, dirs, files in os.walk(path):                
        for filename in (x for x in files if x.endswith('.png')):
            imagePaths.append(os.path.join(root, filename))
        return imagePaths
 
def getName(path):
    base = os.path.basename(path)
    return os.path.splitext(base)[0]

def preProcess(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    gray = cv2.dilate(image, kernel, iterations = 1)                    #dilate to remove text
    gray = cv2.erode(gray, kernel, iterations = 2)                      #erode to restore dilation
    ret, gray = cv2.threshold(gray, 254, 255, cv2.THRESH_TOZERO)        #change white bg to blk
    ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)      #invert binary image for easier processing
    #try to fill images rectangles and remove noise
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return gray

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
  if w<0 or h<0: return () 
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
           
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Scanned image extractor.')
    ap.add_argument('-p', '--path', required=True,
                help="Path to image e.g. .\image.png")
    ap.add_argument('-o', '--output', required=False, default=None,
                help="Path for output images e.g. ..\roi")
    ap.add_argument('-s', '--saveoriginal', required=False, default=None,
                help="Save original document with bounded boxes")
    args = vars(ap.parse_args())
    

    if args["output"] == None:
        output_dir = args["path"]
    else:
        output_dir = os.path.realpath(args["output"])
        if not os.path.isdir(output_dir):
            logging.error(("Output directory %s does not exist" % args["output"]))
        else:
            logging.info("Output will be saved to " + output_dir)
    imgInput = loadInput(args["path"])
    
    for i in imgInput:
        filename = getName(i)
        colour = cv2.imread(i)
        imgSize = colour.shape[1]*colour.shape[0]
        gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
        img = preProcess(gray)

        #find contours and approximate to square
        image, contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        idx = 0
        rect = []
        for c in contours:
            if cv2.contourArea(c) > 1500:
                rect.append(cv2.boundingRect(c))
                sortBoxes = sorted(rect, key = lambda x: x[1]*3000+x[0])
                try:
                    final = combineBoxes(sortBoxes)
                except IndexError:
                    log.debug ("Boxes in " + filename + " can't be combined.")
                    cv2.rectangle(colour, (x,y), (x+w, y+h), (0, 0, 255), 2)
                    if w*h == imgSize:
                        log.debug ("Can't find significant image in " + filename)
                    else:
                        roi = colour[y:y+h, x:x+w]                                   #save region of interests
                        cv2.imwrite(os.path.join(output_dir, filename + '_roi-' + str(idx) + '.png'), roi) 
                                                           

        for x,y,w,h in final:
            if w*h == imgSize:
                log.debug ("Can't find significant image in " + filename)
            elif w < 50:
                log.debug ("Width too narrow for box in " + filename)
            elif h < 50:
                log.debug ("Height too short for box in " + filename)
            else:
                idx += 1
                cv2.rectangle(colour, (x,y), (x+w, y+h), (0, 0, 255), 2)
                roi = colour[y:y+h, x:x+w]                                        #save region of interests
                cv2.imwrite(os.path.join(output_dir, filename + '_roi-' + str(idx) + '.png'), roi)

        if args["saveoriginal"] is not None:
            cv2.imwrite(os.path.join(output_dir, filename + '.png'), colour)
        

    cv2.destroyAllWindows()
