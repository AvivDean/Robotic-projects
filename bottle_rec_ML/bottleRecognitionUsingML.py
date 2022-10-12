import cv2
import numpy as np
from matplotlib import pyplot as plt
while True:
    frame = cv2.imread('C:/Users/Roie/Desktop/bottle_HSV/1.jpg')
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([frame],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
        plt.xlabel("RGB Values")
        plt.ylabel("Pixels")
        plt.show()
    blurred_frame = cv2.GaussianBlur(frame,(5,5),0)
    #convert to hsv
    hsv = cv2.cvtColor(blurred_frame,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[180,256],[0,180,0,256])
    #range for lower red mask
    #lower red range
    l_r = np.array([0,120,70])
    #upper red range
    u_r = np.array([10,255,255])
    mask1 = cv2.inRange(hsv,l_r,u_r)
    #range for upper red mask
    l_r = np.array([170,120,70])
    u_r = np.array([180,255,255])
    mask2 = cv2.inRange(hsv,l_r,u_r)
    #final mask
    final_mask = mask1+mask2
    res = cv2.bitwise_and(frame,frame,mask = final_mask)
    edged = cv2.Canny(res, 30, 200)
    contours, hierarchy = cv2.findContours(edged,
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cv2.drawContours(frame,contours,-1,(0,255,0),3)
    print(contours)
    cv2.imshow("frame",frame)
    cv2.imshow("mask",final_mask)
    cv2.imshow("res",res)
    cv2.imshow("edged",edged)
    plt.xlabel("HSV Values")
    plt.ylabel("Pixels")
    plt.hist(frame.ravel(),256,[0,256])
    plt.show()
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
    break
