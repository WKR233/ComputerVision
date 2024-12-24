''' 
This is file for part 1 
It defines the Gibbs sampler and we use cython for acceleration
'''
from tqdm import tqdm
import numpy as np
import random

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

def gibbs_sample(img_syn, hists_syn,
                 img_ori, hists_ori,
                 filter_list, sweep, bounds,
                 T, weight, num_bins):
    '''
    The gibbs sampler for synthesizing a texture image using annealing scheme
    Parameters:
        1. img_syn: the synthesized image, numpy array, shape: [H,W]
        2. hists_syn: the histograms of the synthesized image, numpy array, shape: [num_chosen_filters,num_bins]
        3. img_ori: the original image, numpy array, shape: [H,W]
        4. hists_ori: the histograms of the original image, numpy arrays, shape: [num_chosen_filters,num_bins]
        5. filter_list: the list of selected filters
        6. sweep: the number of sweeps
        7. bounds: the bounds of the responses of img_ori, a array of numpy arrays in shape [num_chosen_filters,2], bounds[x][0] max response, bounds[x][1] min response
        8. T: the initial temperature
        9. weight: the weight of the error, a numpy array in the shape of [num_bins]
        10. num_bins: the number of bins of histogram, a scalar
    Return:
        img_syn: the synthesized image, a numpy array in shape [H,W]
    '''

    H,W = (img_syn.shape[0],img_syn.shape[1])
    num_chosen_filters = len(filter_list)
    print(" ---- GIBBS SAMPLING ---- ")
    for s in tqdm(range(sweep)):
        print(f"round {s}")
        for pos_h in range(H):
            for pos_w in range(W):
                pos = [pos_h,pos_w]
                print(pos)
                img_syn,hists_syn = pos_gibbs_sample_update(img_syn,hists_syn,img_ori,hists_ori,filter_list,bounds,weight,pos,num_bins,T)
        max_error = (np.abs(hists_syn-hists_ori) @ weight).max()
        print(f'Gibbs iteration {s+1}: error = {(np.abs(hists_syn-hists_ori) @ weight).mean()} max_error: {max_error}')
        T = T * 0.3
        if max_error < 0.1:
            print(f"Gibbs iteration {s+1}: max_error: {max_error} < 0.1, stop!")
            break
    return img_syn, hists_syn


def pos_gibbs_sample_update(img_syn, hists_syn,
                            img_ori, hists_ori,
                            filter_list, bounds,
                            weight, pos, 
                            num_bins, T):
    '''
    The gibbs sampler for synthesizing a value of single pixel
    Parameters:
        1. img_syn: the synthesized image, a numpy array in shape [H,W]
        2. hists_syn: the histograms of the synthesized image, a list of numpy arrays in shape [num_chosen_filters,num_bins]
        3. img_ori: the original image, a numpy array in shape [H,W]
        4. hists_ori: the histograms of the original image, a list of numpy arrays in shape [num_chosen_filters,num_bins]
        5. filter_list: the list of filters, a list of numpy arrays 
        6. bounds: the bounds of the responses of img_ori, a list of numpy arrays in shape [num_chosen_filters,2], in the form of (max_response, min_response)
        7. weight: the weight of the error, a numpy array in the shape of [num_bins]
        8. pos: the position of the pixel, a list of two scalars
        9. num_bins: the number of bins of histogram, a scalar
        10. T: current temperture of the annealing scheme
    Return:
        img_syn: the synthesized image, a numpy array in shape [H,W]
        hist_syn: the histograms of the synthesized image, a list of numpy arrays in shape [num_chosen_filters,num_bins]
    '''
    H = img_syn.shape[0]
    W = img_syn.shape[1]
    pos_h = pos[0]
    pos_w = pos[1]
    energy = 0

    # calculate the conditional probability: p(I(i,j) = intensity | I(x,y),\forall (x,y) \neq (i,j))
    # perturb (i,j) pixel's intensity

    # TODO
    hists_list = [] # store all the hists for different pixel intensity
    img = img_syn.copy()
    for i in range(8): #eight intensity
        hists_pixel = []
        for f in filter_list:
            index = filter_list.index(f)
            img[pos_h][pos_w] = i #change its intensity
            img_tmp = conv(img, f)
            hist = get_histogram(img_tmp, bins_num=num_bins, max_response=bounds[index][0], min_response=bounds[index][1], img_H=H, img_W=W)
            hists_pixel.append(hist)
        hists_list.append(hists_pixel)
    hists_list = np.array(hists_list)

    # calculate the energy
    # TODO
    energy = np.sum(np.abs(hists_ori - hists_list) @ weight, axis=1)
    energy = energy - energy.min()
    
    probs = np.exp(-energy/T)
    eps = 1e-10
    probs = probs + eps
    # normalize the probs
    probs = probs / probs.sum()
    # sample the intensity change the synthesized image
    try:
        # inverse_cdf
        # TODO
        sample = random.random()
        accum = 0
        for i in range(len(probs)):
            if accum <= sample and sample < accum + probs[i]:
                pixel_intensity = i
                break
            accum += probs[i]
    except:
        raise ValueError(f'probs = {probs}')

    # update the histograms
    # TODO
    img_syn[pos_h][pos_h] = pixel_intensity
    hists_syn = hists_list[pixel_intensity]

    return img_syn,hists_syn


