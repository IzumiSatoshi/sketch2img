import cv2
import time
import timeit


image = cv2.imread("./data/generated_image_3.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (64, 64))
can = cv2.Canny(gray, 100, 200)
cv2.imwrite("output.png", can)
