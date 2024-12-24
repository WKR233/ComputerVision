'''
Thanks Yuran Xiang for the help of this problem
-----------------------------------------------
This is the code for project 1 question 3
A 2D scale invariant world
'''
import numpy as np
import cv2
r_min = 1
def inverse_cdf(x):
    ''' 
    Parameters:
        1. x: the random number sampled from uniform distribution
    Return:
        1. y: the random number sampled from the cubic law power
    '''
    y = np.sqrt(1/(1-x)) # Need to be changed
    # TODO: Add your code here
    return y
def GenLength(N):
    ''' 
    Function for generating the length of the line
    Parameters:
        1. N: the number of lines
    Return:
        1. random_length: N*1 array, the length of the line, sampled from sample_r
    Tips:
        1. Using inverse transform sampling. Google it!
    '''
    # sample a random number from uniform distribution
    U = np.random.random(N)
    random_length = inverse_cdf(U)
    return random_length

def DrawLine(points,rad,length,pixel,N):
    ''' 
    Function for drawing lines on a image
    Parameters:
        1. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        2. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        3. length: N*1 array, the length of the line, sampled from sample_r
        4. pixel: the size of the image
        5. N: the number of lines
    Return:
        1. bg: the image with lines
    '''
    # background
    bg = 255*np.ones((pixel,pixel)).astype('uint8')

    # TODO: Add your code here
    for i in range(N):
        x1 = int(points[i])
        y1 = int(points[i + N])
        orientation = rad[i]
        len = length[i]
        if len < 1:
            continue
        x2 = int(x1 + len * np.cos(orientation))
        y2 = int(y1 + len * np.sin(orientation))
        if x2 < 0:
            x2 = 0
        if y2 < 0:
            y2 = 0
        if x2 >= pixel:
            x2 = pixel - 1
        if y2 >= pixel:
            y2 = pixel - 1
        cv2.line(bg, (x1, y1), (x2, y2), 0, 1)
    cv2.imwrite('./pro3_result/'+str(pixel)+'.png', bg)
    return bg

def solve_q1(N = 5000,pixel = 1024):
    ''' 
    Code for solving question 1
    Parameters:
        1. N: the number of lines
        2. pixel: the size of the image
    '''
    # Generating length
    length = GenLength(N)

    # Generating starting points uniformly
    points_N = np.random.choice(pixel*pixel, N, replace = False) # Need to be changed
    points = np.zeros(2*N)
    count = 0
    for number in points_N:
        points[count] = number % pixel
        points[count + N] = number // pixel
        count = count + 1
    # TODO: Add your code here

    # Generating orientation, range from 0 to 2\pi
    rad = []
    for i in range(N):
        r = 2 * np.pi * np.random.rand()
        rad.append(r)
    # TODO: Add your code here

    image = DrawLine(points,rad,length,pixel,N)
    return image,points,rad,length

def DownSampling(img,points,rad,length,pixel,N,rate):
    ''' 
    Function for down sampling the image
    Parameters:
        1. img: the image with lines
        2. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        3. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        4. length: N*1 array, the length of the line
        5. pixel: the size of the image
        6. rate: the rate of down sampling
    Return:
        1. image: the down sampled image
    Tips:
        1. You can use Drawline for drawing lines after downsampling the components
    '''
    image = DrawLine(points//rate,rad,length/rate,pixel//rate,N) # Need to be changed    
    # TODO: Add your code here

    return image

def crop(image1,image2,image3):
    ''' 
    Function for cropping the image
    Parameters:
        1. image1, image2, image3: I1, I2, I3
    '''
    pixel = 128
    for i in range(2):
        x1 = np.random.choice(image1.shape[0] - pixel)
        y1 = np.random.choice(image1.shape[1] - pixel)
        x2 = np.random.choice(image2.shape[0] - pixel)
        y2 = np.random.choice(image2.shape[1] - pixel)
        x3 = np.random.choice(image3.shape[0] - pixel)
        y3 = np.random.choice(image3.shape[1] - pixel)
        I1 = image1[x1:x1+pixel, y1:y1+pixel]
        I2 = image2[x2:x2+pixel, y2:y2+pixel]
        I3 = image3[x3:x3+pixel, y3:y3+pixel]
        cv2.imwrite('./pro3_result/crop/'+str(image1.shape[0])+'crop'+str(i)+'.png', I1)
        cv2.imwrite('./pro3_result/crop/'+str(image2.shape[0])+'crop'+str(i)+'.png', I2)
        cv2.imwrite('./pro3_result/crop/'+str(image3.shape[0])+'crop'+str(i)+'.png', I3)
    # TODO: Add your code here
    return


def main():
    N = 10000
    pixel = 1024
    image_1024, points, rad, length = solve_q1(N,pixel)
    # 512 * 512
    image_512 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 2)
    # 256 * 256
    image_256 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 4)
    crop(image_1024,image_512,image_256)
if __name__ == '__main__':
    main()
