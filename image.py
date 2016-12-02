# -*- coding: utf-8 -*-

import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc


def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]

def show(color_array):
    """ Function to show image"""
    plt.imshow(color_array)
    plt.show()
    plt.axis('off')

def show_gray(gray_array):
    """ Function to show grayscale image"""
    plt.imshow(gray_array, cmap=plt.cm.Greys_r)
    plt.show()
    plt.axis('off')

class Image(object):
    """Class for Image"""

    def __init__(self, fmt, path):
        self.path = os.path.join("image_set", fmt, str(path))
        self.fmt = fmt
        self.array = misc.imread(self.path)
        self.shape = self.array.shape

    @property
    def grayScale(self):
        """Grayscale image"""
        grey = np.zeros((self.shape[0], self.shape[1]))
        for row in range(len(self.array)):
            for col in range(len(self.array[row])):
                grey[row][col] = weightedAverage(self.array[row][col])
        self._grayScale = grey
        return self._grayScale

    def get_saturation_image(self):
        red_canal = self.array[:, :, 0]
        green_canal = self.array[:, :, 1]
        blue_canal = self.array[:, :, 2]
        mean = (red_canal + green_canal + blue_canal) / 3
        saturation = np.sqrt(((red_canal - mean)**2 + (green_canal - mean)**2 + (blue_canal - mean)**2)/3)
        return saturation


eagle = Image('', 'eagle.jpg')

sat = eagle.get_saturation_image()

show_gray(sat)
