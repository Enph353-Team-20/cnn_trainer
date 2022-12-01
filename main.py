#!/usr/bin/env python3

import string
import random
from random import randint
import cv2
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw

import math
import re

from collections import Counter
from matplotlib import pyplot as plt

import utils

class SymbolImage():
    def __init__(self, img, sym, o_h):
        self.img = img
        self.o_h = o_h
        self.sym = sym

class Network():
    training_imgs = []

    def __init__(self):
        pass

    def train(self):
        pass

    def import_plates(self, folder_path):
        """Import entire license plates 

        Args:
            folder_path (_type_): _description_
        """
        filenames = list_files(folder_path)
        for fn in filenames:
            plate = cv2.imread(folder_path + '/' + fn, cv2.IMREAD_GRAYSCALE)

            crops = self.crop_plate(plate)
            syms = (fn[1], fn[2], fn[3], fn[4], fn[5])

            for i in range(len(crops)):
                self.training_imgs.append(SymbolImage(
                    crops[i],
                    syms[i],
                    convert_to_one_hot(np.array(remap_sym(syms[i])), 36)
                ))
            

    def crop_plate(self, img):
        """Returns 5 cropped images from a license plate: car identifier, as well as the 4 letters/numbers.

        Args:
            img (arr): np array
        """
        return (
            img[:,:],
            img[:,:],
            img[:,:],
            img[:,:],
            img[:,:]
        )


    

def list_files(path):
    return os.listdir(path)

def remap_sym(sym):
    if sym >= '0' and sym <= '9':
        return ord(sym) - ord('0')
    else:
        return ord(sym) - ord('A') + 10

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def get_one_hot(sym):
    return convert_to_one_hot(np.array(remap_sym(sym)), 36)


if __name__ == "__main__":
    nn = Network()
    nn.import_plates('./training_data')
    for im in nn.training_imgs[0:3]:
        print('Image:')
        print(str(im.sym) + ', ' + str(im.o_h))
        print(im.img)
