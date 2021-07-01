import cv2
import numpy as np


line_pos = 550  # line position
detected_cars = []
cars = 0

# function to take center.
def takeCenter(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

cap = cv2.VideoCapture('video.mp4')

subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, rect = cap.read()
    gray = cv2.cvtColor(rect, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    img_sub = subtractor.apply(blur)
    # it increases the white region in the image or size of foreground object increases, It is also useful in joining broken parts of an object
    dilation = cv2.dilate(img_sub, np.ones((5, 5)))
    # Returns a structuring element of the specified size and shape for morphological operations.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # It is useful in closing small holes inside the foreground objects, or small black points on the object.
    dilated = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    cnts, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # image,start point, end point
    cv2.line(rect, (25, line_pos), (1250, line_pos), (0, 0, 255), 3)

    for cnt in cnts:
        (x, y, w, h) = cv2.boundingRect(cnt)

        if (w >= 80 and h >= 80):
            # for draw rectangle
            cv2.rectangle(rect, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = takeCenter(x, y, w, h)
            cv2.circle(rect, center, 5, (0, 0, 255), -1)
            # for add single item to the existing list
            detected_cars.append(center)

        for (x, y) in detected_cars:
            if y < (line_pos + 6) and y > (line_pos - 6):
                cars += 1
                cv2.line(rect, (25, line_pos), (1250, line_pos), (0, 255, 0), 3)
                # removes a given object from the list
                detected_cars.remove((x, y))

    cv2.putText(rect, "DETECTED CARS : " + str(cars), (50, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow("Video", rect)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
