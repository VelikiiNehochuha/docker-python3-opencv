import os
import sys

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Load an color image in grayscale

_file = 'owl__8e5e0aca9fbc6da6580d754c056758cd.jpg'
source_name = _file
default_path = '/images/'

img = cv.imread(default_path + source_name, 0)

kernel = np.ones((5,5),np.uint8)
# noize = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
dilate = cv.dilate(img,kernel,iterations = 1)
erode = cv.erode(dilate,kernel,iterations = 1)
closing = cv.morphologyEx(erode, cv.MORPH_CLOSE, kernel)
dilate = cv.dilate(closing,kernel,iterations = 1)
opening = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)
# ret, thresh = cv.threshold(dilate,127,255,0)
 

# ret,thresh = cv.threshold(opening,127,255,0)
ret2,th2 = cv.threshold(opening,235,255,cv.THRESH_BINARY)
# opening[th2 == 127] = 0
# cv.imshow('res', opening)
image, contours, hierarchy = cv.findContours(th2,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    print(contour)
    # get rectangle bounding contour
    [x, y, w, h] = cv.boundingRect(contour)
    if w < 30 or h < 30:
        continue
    cnt = contours[4]
    img = cv.drawContours(img, [contour], 0, (70,255,70), 3)
    # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 0)
cv.imshow('res', img)

cv.imwrite(
    '/test/test.jpg',
    img)
cv.waitKey(0)
sys.exit(0)
blur = cv.GaussianBlur(img,(5,5),0)
ret2,th2 = cv.threshold(img,0,240,cv2.THRESH_BINARY)
erode = cv.erode(th2,kernel,iterations = 1)
# ret,thresh = cv.threshold(dilate,0,10,cv.THRESH_BINARY)
image, contours, hierarchy = cv.findContours(erode,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
for contour in contours:
    print(contour)
    # get rectangle bounding contour
    [x, y, w, h] = cv.boundingRect(contour)
    if h < 30 or w < 30:
        continue
        # draw rectangle around contour on original image
    else:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 0)
# img2 = img.copy()
# h,w = img.shape[:2]
# mask = np.zeros((h,w), np.uint8)

# gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# equ = cv.equalizeHist(gray)

# black = np.where(equ>10)
# img2[black[0], black[1], :] = [255, 255, 255]

# # Transform to gray colorspace and make a thershold then dilate the thershold
# gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
# _, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# kernel = np.ones((15,15),np.uint8)
# dilation = cv.dilate(thresh,kernel,iterations = 1)

# # Search for contours and select the biggest one and draw it on mask
# _, contours, hierarchy = cv.findContours(dilation,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
# cnt = max(contours, key=cv.contourArea)
# cv.drawContours(mask, [cnt], 0, 255, -1)

# # Perform a bitwise operation
# res = cv.bitwise_and(img, img, mask=mask)
# k = cv.waitKey(0)
# sys.exit(0)

# tmp_dir = "/test/" + source_name.replace('.jpg', '')
# try:
#     os.mkdir(tmp_dir)
# except:
#     pass
# print(contours)
# for contour in contours:
#     # get rectangle bounding contour
#     [x, y, w, h] = cv.boundingRect(contour)
#     if h < 10 or w < 10:
#         continue
#     # draw rectangle around contour on original image
#     cv.rectangle(imgray, (x, y), (x + w, y + h), (255, 0, 255), 0)
    # base_name = '{}____x{}_y{}_w{}_h{}.jpg'.format(
    #     source_name.replace('.jpg', ''),
    #     x, y, w, h)
    # crop_img = img[y:y+h, x:x+w]
    # print(x, y, w, h)
    # if x == 0 and y == 344 and w == 116 and h == 40:
    #     pass
    #     cv.imwrite(
    #         tmp_dir + '/' + 'alpha.jpg',
    #         crop_img)
    # else:
    #     cv.imwrite(
    #         tmp_dir + '/' + base_name,
    #         crop_img)

# img = cv.imread('home.jpg')
# color = ('b','g','r')
# for i,col in enumerate(color):
#     histr = cv.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr,color = col)
#     plt.xlim([0,256])
# plt.show()
# plt.hist(img.ravel(),256,[0,256]); plt.show()
# cv.imshow('res', imgray)
# k = cv.waitKey(0)


# for root, dirs, files in os.walk(default_path):
#     for _file in files:
#         source_name = _file
#         img = cv.imread(default_path + source_name, 0)
#         ret, thresh = cv.threshold(img,128,255,0)
#         image, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
#         # print(contours)
#         tmp_dir = "/test/" + source_name.replace('.jpg', '')
#         try:
#             os.mkdir(tmp_dir)
#         except:
#             pass
#         for contour in contours:
#             # get rectangle bounding contour
#             [x, y, w, h] = cv.boundingRect(contour)
#             if h < 30 or w < 30:
#                 continue
#             # draw rectangle around contour on original image
#             # cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 0)
#             base_name = '{}____x{}_y{}_w{}_h{}.jpg'.format(
#                 source_name.replace('.jpg', ''),
#                 x, y, w, h)
#             crop_img = img[y:y+h, x:x+w]
#             print(x, y, w, h)
#             if x == 0 and y == 344 and w == 116 and h == 40:
#                 pass
#                 cv.imwrite(
#                     tmp_dir + '/' + 'alpha.jpg',
#                     crop_img)
#             else:
#                 cv.imwrite(
#                     tmp_dir + '/' + base_name,
#                     crop_img)
        

