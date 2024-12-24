''' 
This is the main file of Part 1: Julesz Ensemble
'''

from numpy.ctypeslib import ndpointer
import numpy as np
from filters import get_filters
import cv2
from torch.nn.functional import conv2d, pad
import torch
from gibbs import gibbs_sample


def conv(image, filter):
    ''' 
    Computes the filter response on an image.
    Notice: Choose your padding method!
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape (x, x)
    Return:
        filtered_image: numpy array of shape (H, W)
    '''

    # TODO
    (H, W) = image.shape
    (x, _) = filter.shape
    halflength = (x - 1) / 2
    filtered_image = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            for _i in range(x):
                for _j in range(x):
                    delta_x = _i - halflength
                    delta_y = _j - halflength
                    if i + delta_x < 0:
                        new_x = i + delta_x + H
                    elif i + delta_x > H - 1:
                        new_x = i + delta_x - H
                    else:
                        new_x = i + delta_x

                    if j + delta_y < 0:
                        new_y = j + delta_y + W
                    elif j + delta_y > W - 1:
                        new_y = j + delta_y - W
                    else:
                        new_y = j + delta_y
                    
                    filtered_image[i][j] += filter[_i][_j] * image[int(new_x)][int(new_y)]

    return filtered_image 

def get_histogram(filtered_image,bins_num, max_response, min_response, img_H, img_W):
    ''' 
    Computes the normalized filter response histogram on an image.
    Parameters:
        1. filtered_image: numpy array of shape (H, W)
        2. bins_num: int, the number of bins
        3. max_response: int, the maximum response of the filter
        4. min_response: int, the minimum response of the filter
    Return:
        1. histogram: histogram (numpy array)
    '''

    # TODO
    histogram = list(np.histogram(filtered_image, bins = bins_num, range = (min_response, max_response))[0])
    histogram[0] += len(np.where(filtered_image < min_response)[0])
    histogram[-1] += len(np.where(filtered_image > max_response)[0])

    histogram = histogram / np.sum(histogram)

    return histogram

def julesz(img_size = 64, img_name = "fur_obs.jpg", save_img = True):
    ''' 
    The main method
    Parameters:
        1. img_size: int, the size of the image
        2. img_name: str, the name of the image
        3. save_img: bool, whether to save intermediate results, for autograder
    '''
    max_intensity = 255

    # get filters
    F_list = get_filters()
    F_list = [filter.astype(np.float32) for filter in F_list]

    # selected filter list, initially set as empty
    filter_list = []
    filter_indexs = []
    iteration_errors = []


    # size of image
    img_H  = img_W = img_size


    # read image
    img_ori = cv2.resize(cv2.imread(f'images/{img_name}', cv2.IMREAD_GRAYSCALE), (img_H, img_W))
    img_ori = (img_ori).astype(np.float32)
    img_ori = img_ori * 7 // max_intensity 

    # store the original image
    if save_img:
        cv2.imwrite(f"results/{img_name.split('.')[0]}/original.jpg", (img_ori / 7 * 255))

    # synthesize image from random noise
    img_syn = np.random.randint(0,8,img_ori.shape).astype(np.float32)

    # TODO
    bins_num = 7
    filtered_oris = []
    bounds = []
    hists_ori = []
    weight = np.array([8, 4, 2, 1, 2, 4, 8])
    T = 0.01
    sweep = 50
    for filter in F_list:
        filtered_image = conv(img_ori, filter)
        bounds.append((filtered_image.max(), filtered_image.min()))
        filtered_oris.append(filtered_image)
        hists_ori.append(get_histogram(filtered_image, bins_num, filtered_image.max(), filtered_image.min(), img_H, img_W))
    hists_ori = np.array(hists_ori)
    bounds = np.array(bounds)

    max_error = 0 # TODO
    threshold = 0.1 # TODO
    max_error_idx = 0

    round = 0
    print("---- Julesz Ensemble Synthesis ----")
    while 1: # Not empty

        # TODO
        print("loop")
        hists_syn = []
        index = 0
        for f in F_list:
            if index not in filter_indexs:
                filtered_image = conv(img_syn, f)
                hist_syn = get_histogram(filtered_image, bins_num, bounds[index][0], bounds[index][1], img_H, img_W)
                hists_syn.append(hist_syn)
            index += 1
        hists_syn = np.array(hists_syn)

        # now we got the hists_syn, next is to compute its error
        loop_error = np.zeros(shape = len(hists_syn))
        print(f"In this iteration, filter number = {len(hists_syn)}")
        for i in range(len(loop_error)):
            loop_error[i] = np.abs(hists_syn[i] - hists_ori[i]) @ weight
        max_error_idx = np.argmax(loop_error)
        max_error = loop_error[max_error_idx]
        iteration_errors.append(loop_error)
        print(f"In this iteration, error = {loop_error}")
        if max_error < threshold:
            break

        if max_error_idx not in filter_indexs: #avoid duplicate
            filter_list.append(F_list[max_error_idx])
            filter_indexs.append(max_error_idx)
        img_syn, hist_syn = gibbs_sample(img_syn, hists_syn[filter_indexs], img_ori, hists_ori[filter_indexs], filter_list, sweep, bounds[filter_indexs], T, weight, bins_num)

        # save the synthesized image
        synthetic = img_syn / 7 * 255
        if save_img:
            cv2.imwrite(f"results/{img_name.split('.')[0]}/synthesized_{round}.jpg", synthetic)
        round += 1
    with open("errors.txt", "w") as f:
        for error in iteration_errors:
            f.write(str(error))
    return img_syn  # return for testing
    
if __name__ == '__main__':
    julesz()
