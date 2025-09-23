import sys
import cv2
import numpy as np
import os

# gaussian kernel 만드는 함수 
def gaussian_blur_kernel_2d(sigma, height, width):
    h_half = height // 2
    w_half = width // 2

    #initialize kernel filter 
    kernel = np.zeros((height, width), dtype=np.float32)

    for y in range(-h_half, h_half + 1):
        for x in range(-w_half, w_half + 1):
            normal = 1.0 / (2.0 * np.pi * sigma**2)
            exp_term = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
            kernel[y + h_half, x + w_half] = normal * exp_term

    # 합이 1이 되도록 정규화
    kernel /= kernel.sum()
    return kernel


#kernel cross_correlation으로 연산 
def cross_correlation_2d(img, kernel):
    kH,kW=kernel.shape
    pad_h,pad_w=kH//2,kW//2

    if len(img.shape) == 2:  # 그레이스케일 이미지
        img_h, img_w = img.shape
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        result = np.zeros((img_h, img_w), dtype=np.float32)
        
        # Cross-correlation 수행
        for i in range(img_h):
            for j in range(img_w):
                result[i, j] = np.sum(padded_img[i:i+kH, j:j+kW] * kernel)
    
    else:  # RGB 이미지
        img_h, img_w, channels = img.shape
        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
        result = np.zeros((img_h, img_w, channels), dtype=np.float32)
        
        # 각 채널별로 Cross-correlation 수행
        for c in range(channels):
            for i in range(img_h):
                for j in range(img_w):
                    result[i, j, c] = np.sum(padded_img[i:i+kH, j:j+kW, c] * kernel)
    
    return result



#kernel img랑 convolve연산산
def convolve_2d(img, kernel):
    k_flip = kernel[::-1, ::-1]
    return cross_correlation_2d(img, k_flip)

#저역 kernel수행한 이미지 반환 
def low_pass(img, sigma, size):
    kernel = gaussian_blur_kernel_2d(sigma, size, size)
    return convolve_2d(img, kernel)

def high_pass(img, sigma, size):
    lp=low_pass(img,sigma,size)
    hp=img-lp
    return hp

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)
