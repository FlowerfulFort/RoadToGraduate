import os
# from scipy.misc import imread
from imageio import imread
import cv2 
from PIL import Image

from dataset.abstract_image_type import AbstractImageType

Image.MAX_IMAGE_PIXELS = None


class RawImageType(AbstractImageType):
    """
    image provider constructs image of type and then you can work with it
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        # self.im = imread(os.path.join(self.paths['images'], self.fn), mode='RGB')
        self.im = imread(os.path.join(self.paths, self.fn))
        # image_path = os.path.join(self.paths['images'], self.fn)
        # self.im = Image.open(image_path)
        # self.im = cv2.imread(os.path.join(self.paths['images'], self.fn))

    def read_image(self):
        im = self.im[...,:-1] if self.has_alpha else self.im
        return self.finalyze(im)

    def read_mask(self):
        path = os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn))
        mask = imread(path, mode='L')
        # mask = cv2.imread(path, mode='L')
        #image = Image.open(path)
        #mask = image.convert('L')
        return self.finalyze(mask)

    def read_alpha(self):
        return self.finalyze(self.im[...,-1])

    def finalyze(self, data):
        #return self.reflect_border_finalyzed(data, 22)
        #return self.reflect_border(data)
        return data
    
