# -*- coding: utf-8 -*-

# import sys  
# reload(sys)   
# sys.setdefaultencoding('utf8')
import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from imageio import imread
import pdb


def weightedAverage(pixel):
    return 0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]


def exponential_euclidean(canal, sigma):
    return np.exp(-(canal - 0.5)**2 / (2 * sigma**2))


def show(color_array):
    """ Function to show image"""
    plt.imshow(color_array)
    plt.show()
    plt.axis('off')


def show_gray(gray_array):
    """ Function to show grayscale image"""
    fig = plt.figure()
    plt.imshow(gray_array, cmap=plt.cm.Greys_r)
    plt.show()
    plt.axis('off')


    '''
    이미지 처리를 위한 클래스
    fmt  = 폴더,            path = 파일 명
    crop = 자를건지 여부,    n    = 얼마나 자를지
    '''
class Image(object):
    """Class for Image"""
    def __init__(self, fmt, path, crop=False, n=0):
        self.path = os.path.join("image_set", fmt, str(path))
        self.fmt = fmt
        self.array = imread(self.path)
        # print(self.array)
        # 픽셀 값 0~255를 255로 나눈후 표현한 실수값
        self.array = self.array.astype(np.float32) / 255
        # print(self.array)
        if crop:
            self.crop_image(n)
        self.shape = self.array.shape
        # print(self.shape) # (800, 1200, 3)
        # print(self.shape[0])

    def crop_image(self, n):
        resolution = 2**n
        (height, width, _) = self.array.shape
        (max_height, max_width) = (resolution * (height // resolution),
                                   resolution * (width // resolution))
        (begin_height, begin_width) = ((height - max_height) // 2,
                                       (width - max_width) // 2)
        self.array = self.array[begin_height:max_height + begin_height,
                                begin_width:max_width + begin_width]

    @property
    def grayScale(self):
        """Grayscale image"""
        rgb = self.array
        
        """
        np.dot은 행렬의 곱
        왜 0.299,0.587,0.114를 곱할까...
        인간의 눈은 영상의 밝기를 29.9%의 빨강, 58.7%의 초록, 11.4%의 파란색으로 느낀다고 함
        따라서 RGB -> YCbCr로 바꿀 때에 Y(흑백 값)을 구하기 위해
        출처! : https://dynaforce.tistory.com/53
        """
        self._grayScale = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
        # print('*'*100)
        # print(rgb[..., :3])
        # print('*'*100)
        # print('-'*100)
        # print(self._grayScale)
        # print('-'*100)
        return self._grayScale

    def saturation(self):
        """Function that returns the Saturation map"""
        red_canal = self.array[:, :, 0]
        green_canal = self.array[:, :, 1]
        blue_canal = self.array[:, :, 2]
        # print(red_canal)
        # print(green_canal)
        # print(blue_canal)
        mean = (red_canal + green_canal + blue_canal) / 3.0
        '''
        np.sqrt는 제곱근을 구함
        각각의 값에서 평균 값을 빼서 제곱근하는 표준편차를 구하는 식
        '''
        saturation = np.sqrt(((red_canal - mean)**2 + (green_canal - mean)**2 +
                              (blue_canal - mean)**2) / 3)
        return saturation

    '''
    대비정도를 구해서 외각선 구하는 함수
    '''
    def contrast(self):
        """Function that returns the Constrast numpy array"""
        grey = self.grayScale
        # shape[0] = width, shape[1] = height
        contrast = np.zeros((self.shape[0], self.shape[1]))
        grey_extended = np.zeros((self.shape[0] + 2, self.shape[1] + 2))
        grey_extended[1:self.shape[0] + 1, 1:self.shape[1] + 1] = grey
            #    kernel = np.array([[ -1, -1, -1 ],
            #                       [ -1, 8, -1 ],
            #                        [ -1, -1, -1 ]])
            
            
        # kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                contrast[row][col] = np.abs(
                    (kernel *
                     grey_extended[row:(row + 3), col:(col + 3)]).sum())
        contrast = (contrast - np.min(contrast))
        contrast = contrast / np.max(contrast)
        return contrast

    '''
    가장자리 검출 알고리즘 soble
    '''
    def sobel(self):
        """Function that returns the Constrast numpy array"""
        grey = self.grayScale
        sobel_h = np.zeros((self.shape[0], self.shape[1]))
        sobel_v = np.zeros((self.shape[0], self.shape[1]))
        grey_extended = np.zeros((self.shape[0] + 2, self.shape[1] + 2)) # 가로 + 2, 세로 + 2 픽셀 담는 어레이 저장
        grey_extended[1:self.shape[0] + 1, 1:self.shape[1] + 1] = grey   # grey arr 대입
        
        kernel1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kernel2 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, -1]])
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                sobel_h[row][col] = np.abs(
                    (kernel1 *
                     grey_extended[row:(row + 3), col:(col + 3)]).sum())
                sobel_v[row][col] = np.abs(
                    (kernel2 *
                     grey_extended[row:(row + 3), col:(col + 3)]).sum())
        return sobel_h, sobel_v

    def exposedness(self):
        """Function that returns the Well-Exposedness map"""
        red_canal = self.array[:, :, 0]
        green_canal = self.array[:, :, 1]
        blue_canal = self.array[:, :, 2]
        sigma = 0.2
        red_exp = exponential_euclidean(red_canal, sigma)
        green_exp = exponential_euclidean(green_canal, sigma)
        blue_exp = exponential_euclidean(blue_canal, sigma)
        return red_exp * green_exp * blue_exp


if __name__ == "__main__":
    # im = Image("jpeg", "grandcanal_mean.jpg")
    # im = Image("mask", "mask_mean.jpg")
    # sat = im.contrast()
    # show_gray(sat)
    im = Image("mask", "mask_over.jpg")
    sat = im.contrast()
    show_gray(sat)
    # im = Image("mask", "mask_under.jpg")
    # sat = im.contrast()
    # show_gray(sat)
    
