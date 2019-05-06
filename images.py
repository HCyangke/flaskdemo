#coding=utf-8
import cv2
import numpy as np
class images(object):
    """docstring for images"""
    def __init__(self):
        super(images, self).__init__()
        self.NoneFalse=cv2.imread('幻灯片01.tiff')
        self.NoneTrue=cv2.imread('幻灯片02.tiff')
        self.FightFalse=cv2.imread('幻灯片03.tiff')
        self.FightTrue=cv2.imread('幻灯片04.tiff')
        self.StallFalse=cv2.imread('幻灯片05.tiff')
        self.StallTrue=cv2.imread('幻灯片06.tiff')
        self.GarbageFalse=cv2.imread('幻灯片07.tiff')
        self.GarbageTrue=cv2.imread('幻灯片08.tiff')
        self.SmashFalse=cv2.imread('幻灯片09.tiff')
        self.SmashTrue=cv2.imread('幻灯片10.tiff')
        self.DripFalse=cv2.imread('幻灯片11.tiff')
        self.DripTrue=cv2.imread('幻灯片12.tiff')
# img=images()
# print(img.NoneFalse.shape)
# image=np.concatenate((img.NoneFalse, img.NoneFalse))
# print(image.shape)
