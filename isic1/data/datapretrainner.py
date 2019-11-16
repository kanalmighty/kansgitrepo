import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
def color_space_demo(image1,image2):
   low = np.array([125,43,46])
   high = np.array([155,255,255])
   mask = cv.inRange(image2, low, high)
   dst = cv.bitwise_and(image2,image2,mask = mask)
   # cv.imshow('mask', mask)
   # cv.imshow('dst' ,dst)

def fill_color_demo(image):
    copy_image = image.copy()
    h ,w = copy_image.shape[:2]
    mask = np.zeros([h+2, w+2],dtype=np.uint8)
    cv.floodFill(copy_image, mask, (30,30), (0, 255, 255), (100,100,100) ,(50,50,50) ,cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("a", copy_image)

def add_gaussion_noise(image):
    h, w, c  = image.shape
    for row in range(h):
        for column in range(w):
            noise = np.random.normal(0, 20, 3)
            print(noise)
            b = image[row, column, 0]
            g = image[row, column, 1]
            r = image[row, column, 2]
            image[row, column, 0] = b + noise[0] if (b + noise[0]) <= 255 else 255
            image[row, column, 1] = b + noise[1] if (b + noise[1]) <= 255 else 255
            image[row, column, 2] = b + noise[2] if (b + noise[1]) <= 255 else 255
    cv.imshow("gs", image)

def gaussion_blur_demo(image):
    dst = cv.GaussianBlur(image,(0,0) ,15)
    cv.imshow("demo", dst)

def blur_demo(image):
    dst = cv.blur(image, (1 ,15))
    cv.imshow("demo", dst)

def median_blur_demo(image):
    dst = cv.medianBlur(image,5)
    cv.imshow("demo", dst)


def contrast_brightness_demo(image, c, b):
    h ,w , c = image.shape
    blank = np.zeros([h, w, c] , dtype=image.dtype)
    dst =  cv.addWeighted(image, c, blank, 1-c, b)
    cv.imshow("demo" ,dst)

def bi_demo(image):
    dst = cv.bilateralFilter(image,0,100,15)
    cv.imshow('dst',dst)


def plot_his(image):
    colors = ['blue', 'green', 'red']
    for i, color in enumerate(colors):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.xlim([0, 256])
    plt.show()

def back_projection(image):
    target_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    target_hist = cv.calcHist([target_hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv.normalize(target_hist, target_hsv, 0, 255, cv.NORM_MINMAX)
    dst = cv.calcBackProject([target_hsv], [0, 1], target_hist, [0, 180, 0, 256], 1)
    cv.imshow("mat", dst)


def def_equal_hist(image):
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    dst = cv.equalizeHist(grey)
    cv.imshow("mat", dst)

def def_hist_2d(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    dst = cv.calcHist([hsv], [0,1], None, [180,256], [0,180,0, 256])
    cv.imshow("mat", dst)
def gloab_threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print(ret)
    cv.imshow("mat", binary)

def local_threshold_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.adaptiveThreshold(gray,  255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    print(ret)
    cv.imshow("mat", binary)

def pyramid_demo(image):
    level = 3
    temp = image.copy()
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("aa", dst)
        print(dst.shape)
        temp = dst.copy()
    return pyramid_images

def lapalian_demo(images):
    lenth = len(images)
    for i in range(lenth-1, -1, -1):
        expanded = cv.pyrUp(images[i], dstsize=images[i-1].shape[:2])
        lab = cv.subtract(images[i-1] - expanded)
        cv.imshow("aa", lab)

def contours_find(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # cv.imshow("mat", binary)
    contours, herichy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        cv.drawContours(image, contours, i, (0, 0, 255), 0)
        print(i)
    cv.imshow("mat", image)

def get_erode(image):
    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(grey, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dst = cv.erode(binary, kernel)
    cv.imshow("mat", dst)





if __name__ == '__main__':
    img1 = cv.imread("D:\\pycharmspace\\datasets\\isic2019\\image\\ISIC_0000000.jpg")
    img2 = cv.imread("D:\\pycharmspace\\datasets\\isic2019\\image\\ISIC_0000004.jpg")
    get_erode(img1)
