import cv2

def binarize_image(image, threshold): # grayscale -> 이진화
    # Grayscale 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이진화
    _, binary_image = cv2.threshold(gray_image, threshold, 1, cv2.THRESH_BINARY)

    return binary_image