import cv2
import numpy as np

image_gray = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY)
cv2.imshow('Gray Image', image_gray)
cv2.imshow('Binary Image', binary_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np

image_night = cv2.imread('night.jpg')
image_light = cv2.imread('light.jpg')

image_night_gray = cv2.cvtColor(image_night, cv2.COLOR_BGR2GRAY)
image_light_gray = cv2.cvtColor(image_light, cv2.COLOR_BGR2GRAY)

mean = 0
stddev = 15
noise_night = np.random.normal(mean, stddev, image_night_gray.shape).astype(np.uint8)
noise_light = np.random.normal(mean, stddev, image_light_gray.shape).astype(np.uint8)

image_night_noisy = cv2.add(image_night_gray, noise_night)
image_light_noisy = cv2.add(image_light_gray, noise_light)

alpha = 1.5
beta = 0
image_night_contrast = cv2.convertScaleAbs(image_night_noisy, alpha=alpha, beta=beta)
image_light_contrast = cv2.convertScaleAbs(image_light_noisy, alpha=alpha, beta=beta)

image_night_smooth = cv2.GaussianBlur(image_night_noisy, (5, 5), 0)
image_light_smooth = cv2.GaussianBlur(image_light_noisy, (5, 5), 0)

kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
image_night_sharpened = cv2.filter2D(image_night_noisy, -1, kernel_sharpening)
image_light_sharpened = cv2.filter2D(image_light_noisy, -1, kernel_sharpening)

image_night_filtered = cv2.medianBlur(image_night_noisy, 5)
image_light_filtered = cv2.medianBlur(image_light_noisy, 5)

cv2.imshow('Enhanced Night Image - Contrast', image_night_contrast)
cv2.imshow('Enhanced Night Image - Smooth', image_night_smooth)
cv2.imshow('Enhanced Night Image - Sharpened', image_night_sharpened)
cv2.imshow('Enhanced Night Image - Filtered', image_night_filtered)

cv2.imshow('Enhanced Light Image - Contrast', image_light_contrast)
cv2.imshow('Enhanced Light Image - Smooth', image_light_smooth)
cv2.imshow('Enhanced Light Image - Sharpened', image_light_sharpened)
cv2.imshow('Enhanced Light Image - Filtered', image_light_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()

image1 = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('red.jpg', cv2.IMREAD_GRAYSCALE)

image2_resized = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

f_transform1 = np.fft.fft2(image1)
f_transform2 = np.fft.fft2(image2_resized)

magnitude1 = np.abs(f_transform1)
magnitude2 = np.abs(f_transform2)
phase1 = np.angle(f_transform1)
phase2 = np.angle(f_transform2)

f_transform_combined = magnitude2 * np.exp(1j * phase1)

image_combined = np.fft.ifft2(f_transform_combined)
image_combined = np.abs(image_combined)

image_combined = cv2.normalize(image_combined, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('Combined Image', image_combined.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

image_night = cv2.imread('night.jpg')
image_light = cv2.imread('light.jpg')

image_night_gray = cv2.cvtColor(image_night, cv2.COLOR_BGR2GRAY)
image_light_gray = cv2.cvtColor(image_light, cv2.COLOR_BGR2GRAY)

mean = 0
stddev = 15
noise_night = np.random.normal(mean, stddev, image_night_gray.shape).astype(np.uint8)
noise_light = np.random.normal(mean, stddev, image_light_gray.shape).astype(np.uint8)

image_night_noisy = cv2.add(image_night_gray, noise_night)
image_light_noisy = cv2.add(image_light_gray, noise_light)

alpha = 1.5
beta = 0
image_night_contrast = cv2.convertScaleAbs(image_night_noisy, alpha=alpha, beta=beta)
image_light_contrast = cv2.convertScaleAbs(image_light_noisy, alpha=alpha, beta=beta)

image_night_smooth = cv2.GaussianBlur(image_night_noisy, (5, 5), 0)
image_light_smooth = cv2.GaussianBlur(image_light_noisy, (5, 5), 0)

kernel_sharpening = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
image_night_sharpened = cv2.filter2D(image_night_noisy, -1, kernel_sharpening)
image_light_sharpened = cv2.filter2D(image_light_noisy, -1, kernel_sharpening)

image_night_filtered = cv2.medianBlur(image_night_noisy, 5)
image_light_filtered = cv2.medianBlur(image_light_noisy, 5)

cv2.imshow('Enhanced Night Image - Contrast', image_night_contrast)
cv2.imshow('Enhanced Night Image - Smooth', image_night_smooth)
cv2.imshow('Enhanced Night Image - Sharpened', image_night_sharpened)
cv2.imshow('Enhanced Night Image - Filtered', image_night_filtered)

cv2.imshow('Enhanced Light Image - Contrast', image_light_contrast)
cv2.imshow('Enhanced Light Image - Smooth', image_light_smooth)
cv2.imshow('Enhanced Light Image - Sharpened', image_light_sharpened)
cv2.imshow('Enhanced Light Image - Filtered', image_light_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
