import cv2

img = cv2.imread("live_testing/12.jpg")
flipped = cv2.flip(img, 1)  # 1 for horizontal flip, 0 for vertical
cv2.imwrite("live_testing/flipped_12.jpg", flipped)