'''
This is the main file for the project 2's first method Gibss Sampler
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
import os
import matplotlib.pyplot as plt
from torch.nn.functional import conv2d, pad


def cal_pot(gradient, norm):
    ''' 
    The function to calculate the potential energy based on the selected norm
    Parameters:
        gradient: the gradient of the image, can be nabla_x or nabla_y, numpy array of size:(img_height,img_width, )
        norm: L1 or L2
    Return:
        A term of the potential energy
    '''
    if norm == "L1":
        return abs(gradient)
    elif norm == "L2":
        return gradient**2 
    else:
        raise ValueError("The norm is not supported!")




def gibbs_sampler(img, loc, energy, beta, norm):
    ''' 
    The function to perform the gibbs sampler for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. energy: a scale
        4. beta: annealing temperature
        5. norm: L1 or L2
    Return:
        img: the updated image
    '''
    
    energy_list = np.zeros((255,1))
    # get the size of the image
    img_height, img_width = img.shape


    # original pixel value
    original_pixel = img[loc[0], loc[1]]

    
    for i in range(255):
        energy_list[i][0] = i
    if loc[0] == 0:
        neighborhood1 = img[img_height - 1, loc[1]]
    else:
        neighborhood1 = img[loc[0] - 1, loc[1]]
    
    if loc[0] == img_height - 1:
        neighborhood2 = img[0, loc[1]]
    else:
        neighborhood2 = img[loc[0] + 1, loc[1]]

    if loc[1] == 0:
        neighborhood3 = img[loc[0], img_width - 1]
    else:
        neighborhood3 = img[loc[0], loc[1] - 1]

    if loc[1] == img_width - 1:
        neighborhood4 = img[loc[0], 0]
    else:
        neighborhood4 = img[loc[0], loc[1] + 1]

    if norm == "L1":
        energy_list = np.abs(energy_list - neighborhood1) + np.abs(energy_list - neighborhood2) + np.abs(energy_list - neighborhood3) + np.abs(energy_list - neighborhood4)
    else:
        energy_list = np.power(energy_list - neighborhood1, 2) + np.power(energy_list - neighborhood2, 2) + np.power(energy_list - neighborhood3, 2) + np.power(energy_list - neighborhood4, 2)

    # normalize the energy
    energy_list = energy_list - energy_list.min()
    # calculate the conditional probability
    probs = np.exp(-energy_list * beta)
    # normalize the probs
    probs = probs / probs.sum()

    try:
        # inverse_cdf and updating the img
        rand_num = np.random.rand(1)
        accu_probs = np.zeros(255)
        accu_probs[0] = probs[0][0]
        for i in range(1, 255):
            accu_probs[i] = accu_probs[i - 1] + probs[i][0]

        for i in range(255):
            if accu_probs[i] < rand_num and accu_probs[i + 1] > rand_num:
                img[loc[0], loc[1]] = i
                break
    except:
        raise ValueError(f'probs = {probs}')
    return img

def conv(image, filter):
    ''' 
    Computes the filter response on an image.
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape, can be [[-1,1]] or [[1],[-1]] or [[1,-1]] or [[-1],[1]] ....
    Return:
        filtered_image: numpy array of shape (H, W)
    '''

    filtered_image = image
    if len(filter) == 1: # [[-1, 1]] or [[1, -1]]
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if j != image.shape[1] - 1:
                    filtered_image[i][j] = image[i][j]*filter[0][0] + image[i][j+1]*filter[0][1]
                else:
                    filtered_image[i][j] = image[i][j]*filter[0][0] + image[i][0]*filter[0][1]
    else: # [[1], [-1]] or [[-1], [1]]
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if i != image.shape[0] - 1:
                    filtered_image[i][j] = image[i][j]*filter[0][0] + image[i+1][j]*filter[1][0]
                else:
                    filtered_image[i][j] = image[i][j]*filter[0][0] + image[0][j]*filter[1][0]

    return filtered_image

def main():
    # read the distorted image and mask image
    name = "stone"
    size = "small"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)

    # calculate initial energy
    red_channel = distort[:,:,2].copy()
    energy = 0

    #calculate nabla_x
    filtered_img = conv(red_channel, np.array([[-1,1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))

    # calculate nabla_y
    filtered_img = conv(red_channel, np.array([[-1],[1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))

    # calculate number of masks
    img_height, img_width = filtered_img.shape
    cnt = 0
    for i in range(img_height):
        for j in range(img_width):
            if mask[i, j, 2] == 255:
                cnt = cnt + 1
    losses = []

    norm = "L2"
    beta = 0.1
    img_height, img_width, _ = distort.shape

    sweep = 100
    for s in tqdm(range(sweep)):
        for i in range(img_height):
            for j in range(img_width):
                # only change the channel red
                if mask[i,j,2] == 255:
                    distort[:,:,2] = gibbs_sampler(distort[:,:,2], [i,j], energy, beta, norm)
        beta = beta * 1.01
        # per pixel error
        loss = np.sum(np.power(distort[:, :, 2] - ori[:, :, 2], 2))/cnt
        print(f"s = {s}, loss = {loss}")
        losses.append(loss)


        save_path = f"./result/{name}/L1_{size}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(f"{save_path}/gibbs_{s}.bmp", distort)

    plt.plot([i for i in range(sweep)], losses)
    plt.xlabel("sweep")
    plt.ylabel("per pixel error")
    plt.savefig(f"{save_path}/loss.jpg")

if __name__ == "__main__":
    main()
