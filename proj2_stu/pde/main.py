'''
This is the main file for the project 2's Second method PDE
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d, pad
import matplotlib.pyplot as plt
import os




def pde(img, loc, beta):
    ''' 
    The function to perform the pde update for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. beta: learning rate
    Return:
        img: the updated image
    '''

    # TODO
    image_height, image_width = img.shape
    delta = 0
    now_pixel = img[loc[0], loc[1]]

    if loc[0] == 0:
        delta += 0.25*(img[image_height - 1, loc[1]] - now_pixel)
    else:
        delta += 0.25*(img[loc[0] - 1, loc[1]] - now_pixel)

    if loc[0] == image_height - 1:
        delta += 0.25*(img[[0], loc[1]] - now_pixel)
    else:
        delta += 0.25*(img[loc[0] + 1, loc[1]] - now_pixel)

    if loc[1] == 0:
        delta += 0.25*(img[loc[0], image_width - 1] - now_pixel)
    else:
        delta += 0.25*(img[loc[0], loc[1] - 1] - now_pixel)
    
    if loc[1] == image_width - 1:
        delta += 0.25*(img[loc[0], 0] - now_pixel)
    else:
        delta += 0.25*(img[loc[0], loc[1] + 1] - now_pixel)

    img[loc[0], loc[1]] += int(beta*delta)

    return img


def main():
    # read the distorted image and mask image
    name = "stone"
    size = "big"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)

    losses = []

    beta = 1
    img_height, img_width, _ = distort.shape

    sweep = 100
    for s in tqdm(range(sweep)):
        for i in range(img_height):
            for j in range(img_width):
                # only change the channel red
                if mask[i,j,2] == 255:
                    distort[:,:,2] = pde(distort[:,:,2], [i,j], beta)

        cnt = 0
        for i in range(img_height):
            for j in range(img_width):
                if mask[i, j, 2] == 255:
                    cnt = cnt + 1


        loss = np.sum(np.power(distort[:, :, 2] - ori[:, :, 2], 2))/cnt
        print(f"s = {s}, loss = {loss}")
        losses.append(loss)

        if s % 10 == 0:
            save_path = f"./result/{name}/{size}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(f"{save_path}/pde_{s}.bmp", distort)

    plt.plot([i for i in range(sweep)], losses, )
    plt.xlabel("sweep")
    plt.ylabel("per pixel error")
    plt.savefig(f"{save_path}/loss.jpg")


if __name__ == "__main__":
    main()







        

