import cv2
import numpy as np
import sys
import math
from scipy.signal import argrelextrema

#adapted from scipy cookbook
def smooth(x,window_len=17,window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len<3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def form_projection_array(img):
    proj = smooth(np.sum(cv2.GaussianBlur(img.copy(), (7, 7), 0), axis=0))
    maxInd = argrelextrema(proj, np.greater)[0]
    return maxInd

# threshold -> canny -> hough transform to take out non-textual features
def line_overlay(gray):
    blur = cv2.GaussianBlur(gray.copy(),(7,7),0)
    ret,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY_INV)
    canny = cv2.Canny(thresh, 50, 200, None, 3)
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 20, None, 100, 5)
    if lines is not None:
        for i in range(len(lines)):
            l = lines[i][0]
            cv2.line(gray, (l[0], l[1]), (l[2], l[3]), (255,255,255), 3, cv2.LINE_AA)


if __name__ == '__main__':
    filename = sys.argv[1]
    image = cv2.imread('../input/'+filename)
    image = cv2.resize(image, (512, 768))                    # Resize image

    # load in gray
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    line_overlay(gray)
        
    #binarize
    ret, thresh = cv2.threshold(gray, 127,255,cv2.THRESH_BINARY_INV)

    #dilation
    img_dilation = cv2.dilate(thresh, np.ones((4, 4), np.uint8)
, iterations=1)
    

    #projection
    projection_indices = form_projection_array(img_dilation)

    #find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1 )

    #sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: -cv2.boundingRect(ctr)[0])

    # draw line alignment
    for ix in projection_indices:
        cv2.line(image, (ix, 0), (ix, image.shape[0]), (0, 255, 0), 1, cv2.LINE_AA)

    # draw bounding boxes
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        # print("Segment No. {} ({}x{}), A={}".format(i, w, h, w*h))
        if (w * h < 100 or w * h > 50000): continue
        if (w < 8 or w > 60): continue

        roi = image[y:y+h, x:x+w]

        cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),1)
        #cv2.putText(image, "#{}".format(i), (x + w + 5, int(y + h / 2)), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0))

    cv2.imshow('marked areas',image)
    cv2.waitKey(0)
    sys.exit(0)
