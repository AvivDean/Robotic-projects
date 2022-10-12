import time
import cv2
import numpy as np

# show_image_enable = True
draw_circle_enable = True
camera_port = 1
kernel = np.ones((5, 5), np.uint8)
img = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
SCREEN_WIDTH = img.get(3)
SCREEN_HIGHT = img.get(4)
CENTER_X = SCREEN_WIDTH / 2
CENTER_Y = SCREEN_HIGHT / 2
BALL_SIZE_MIN = SCREEN_HIGHT / 10
BALL_SIZE_MAX = SCREEN_HIGHT / 3
MIDDLE_TOLERANT = 5
# Filter setting
hmnL = 25.5
hmxL = 53.5
smnL = 134.6
smxL = 225.67
vmnL = 120.36
vmxL = 220.3
while (img.isOpened()):
    ret, bgr_image = img.read()
    orig_image = bgr_image
    bgr_image = cv2.medianBlur(bgr_image, 3)
    # Convert input image to HSV
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image, keep only the green pixels
    thImage = cv2.inRange(hsv_image, (hmnL, smnL, vmnL), (hmxL, smxL,
                                                          vmxL))
    # Combine the above two images
    gausF = cv2.GaussianBlur(thImage, (9, 9), 2, 2)
    circles = cv2.HoughCircles(thImage, cv2.HOUGH_GRADIENT, 1, 120,
                               100, 20, 10, 0)
    circles = np.uint16(np.around(circles))
    # Loop over all detected circles and outline them on the original image
    all_r = np.array([])

    if circles is not None:
        try:

            for i in circles[0, :]:
                # print("i: %s"%i)
                all_r = np.append(all_r, int(round(i[2])))
                closest_ball = all_r.argmax()
                center = (int(round(circles[0][closest_ball][0])),
                          int(round(circles[0][closest_ball][1])))
                radius = int(round(circles[0][closest_ball][2]))
            if draw_circle_enable:
                cv2.circle(orig_image, center, radius, (0, 0, 255), 2)
            time.sleep(0.1)
        except IndexError:
            print('No ball was found, I will keep looking')
            continue
        else:
            pass
    x = 0  # x initial in the middle
    y = 0  # y initial in the middle
    r = 0  # ball radius initial to 0(no balls if r < ball_size)
    for _ in range(10):
        (tmp_x, tmp_y), tmp_r = center, radius
    if tmp_r > BALL_SIZE_MIN:
        x = tmp_x
        y = tmp_y
        r = tmp_r
        break
    print(x, y, r)
    if r < BALL_SIZE_MIN:
        print('The ball is far away')
    elif r < BALL_SIZE_MAX:
        if abs(x - CENTER_X) > MIDDLE_TOLERANT:
            if x < CENTER_X:  # Ball is on left
                print('Ball is on the left')
            else:  # Ball is on right
                print('Ball is on the right')
        if abs(y - CENTER_Y) > MIDDLE_TOLERANT:
            if y < CENTER_Y:
                print('Ball is on top')
            else:
                print('Ball is in the bottom')
    cv2.namedWindow("Gaussian Filter for threshold image",
                    cv2.WINDOW_NORMAL)
    cv2.imshow("Gaussian Filter for threshold image", thImage)
    cv2.namedWindow("Detected green circles on the input image",
                    cv2.WINDOW_NORMAL)
    cv2.imshow("Detected green circles on the input image",
               orig_image)
    print('press "q" to exit the program')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
img.release()
cv2.destroyAllWindows()