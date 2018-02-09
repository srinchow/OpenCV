import cv2
import numpy as np

def lane_detection(original):

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)  # grayscale conversion
    hsv = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)  # HSV conversion

    y = np.array([20, 100, 100], dtype='uint8') # lower limit for yellow
    Y = np.array([255, 255, 255], dtype='uint8') #upper limit for yellow
    mask_yellow = cv2.inRange(hsv, y, Y)
    mask_white = cv2.inRange(gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)

    # Appying a gaussian blur
    gauss_gray = cv2.GaussianBlur(mask_yw_image, (1, 1), 0)

    # Appying Canny Edge Detection
    canny_edges = cv2.Canny(gauss_gray, 50, 150)
    img = canny_edges

    #Setting a region of interest
"""
    imshape = img.shape

    lower_left = [0, imshape[0]]
    lower_right = [imshape[1] * 0.85, imshape[0]]
    top_left = [imshape[1] * 0.4, imshape[0] * 0.61]
    top_right = [imshape[1] * 0.6, imshape[0] * 0.61]
    vertices = [np.array([lower_left, top_left, top_right, lower_right], dtype=np.int32)]

    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    img = masked_image
   """ 
    # Appying Hough Line Transform
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 20, np.array([]), 50, 50)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    color = [255, 0, 0]
    thickness = 10
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(original, (x1, y1), (x2, y2), color, thickness)

    return original

def main():
    vid = cv2.VideoCapture('project_video.mp4')

    if (vid.isOpened() == False):
        print("Error opening video file")
    while(vid.isOpened()):
        ret , frame = vid.read()
        if ret == True:
            img = lane_detection(frame)
            cv2.imshow('frame',img)
            if cv2.waitKey(25)& 0xff == ord('q'): # press q to exit
                break
        else:
            break
    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':main()
