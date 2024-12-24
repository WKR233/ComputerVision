'''
This is the code for project 1 question 2
Question 2: Verify the 1/f power law observation in natural images in Set A
'''
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import cv2
path = "./image_set/setA/"
colorlist = ['red', 'blue', 'black', 'green']
linetype = ['-', '-', '-', '-']
labellist = ["natural_scene_1.jpg", "natural_scene_2.jpg",
                 "natural_scene_3.jpg", "natural_scene_4.jpg"]

img_list = [cv2.imread(os.path.join(path,labellist[i]), cv2.IMREAD_GRAYSCALE) for i in range(4)]
def fft(img):
    ''' 
    Conduct FFT to the image and move the dc component to the center of the spectrum
    Tips: dc component is the one without frequency. Google it!
    Parameters:
        1. img: the original image
    Return:
        1. fshift: image after fft and dc shift
    '''
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) # Need to be changed
    # TODO: Add your code here

    return fshift

def amplitude(fshift):
    '''
    Parameters:
        1. fshift: image after fft and dc shift
    Return:
        1. A: the amplitude of each complex number
    '''

    A = np.abs(fshift) # Need to be changed
    # TODO: Add your code here

    return A

def xy2r(x, y, centerx, centery):
    ''' 
    change the x,y coordinate to r coordinate
    '''
    rho = math.sqrt((x - centerx)**2 + (y - centery)**2)
    return rho

def cart2porl(A,img):
    ''' 
    Parameters: 
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. f: the frequency list 
        2. A2_f: the amplitude of each frequency
    Tips: 
        1. Use the function xy2r to get the r coordinate!
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)
    # build the r coordinate
    basic_f = 1
    max_r = min(centerx,centery)
    # the frequency coordinate
    f = np.arange(0,max_r + 1,basic_f)

    # the following process is to do the sampling for each frequency of f
    A_f = [[] for _ in range(len(f))]
    rmax = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r = round(xy2r(i,j,centerx,centery))
            if r<=max_r:
                A_f[r].append(A[i][j])
    for i in range(len(f)):
        A_f[i] = np.average(A_f[i])
    # TODO: Add your code here

    return f, A_f


def get_S_f0(A,img):
    ''' 
    Parameters:
        1. A: the amplitude of each complex number
        2. img: the original image
    Return:
        1. S_f0: the S(f0) list
        2. f0: frequency list
    '''
    centerx = int(img.shape[0] / 2)
    centery = int(img.shape[1] / 2)
    A_2 = np.square(A)
    f, A2_f = cart2porl(A_2, img)
    f0 = np.arange(0,100,1)

    S_f0 = np.ones(len(f0))
    for freq in f0:
        S_f0[freq] = 2 * math.pi * np.sum([A2_f[x]*x for x in range(freq,max(2*freq+1,f0[-1]))])
    # TODO: Add your code here

    return S_f0, f0
    
def main():
    plt.figure(1)
    # q1

    for i in range(4):
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        f, A_f = cart2porl(A,img_list[i])
        plt.plot(np.log(f[1:190]),np.log(A_f[1:190]), color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("1/f law")
    plt.savefig("./pro2_result/f1_law.jpg", bbox_inches='tight', pad_inches=0.0)

    # q2
    plt.figure(2)
    for i in range(4):
        fshift = fft(img_list[i])
        A = amplitude(fshift)
        S_f0, f0 = get_S_f0(A,img_list[i])
        plt.plot(f0[10:],S_f0[10:], color=colorlist[i], linestyle=linetype[i], label=labellist[i])
    plt.legend()
    plt.title("S(f0)")
    plt.savefig("./pro2_result/S_f0.jpg", bbox_inches='tight', pad_inches=0.0)
if __name__ == '__main__':
    main()
