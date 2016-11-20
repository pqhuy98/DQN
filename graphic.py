import scipy
import numpy as np
import matplotlib.pyplot as plt

def transform(img,h=32,w=32) :
    res = scipy.misc.imresize(img,(h,w),'bilinear')
    res = rgb2gray(res)
    return res


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def show_grey(img,name="") :
    plt.figure()
    plt.title(name)
    plt.imshow(img,cmap='Greys_r')
    plt.show()

def show_rgb(img,name="") :
    plt.figure()
    plt.title(name)
    plt.imshow(-img)
    plt.show()
    