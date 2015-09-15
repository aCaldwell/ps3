"""Problem Set 3: Window-based Stereo Matching."""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2

import os

input_dir = "input"  # read images from os.path.join(input_dir, <filename>)
output_dir = "output"  # write images to os.path.join(output_dir, <filename>)


def save_image(img, img_name):
    """ Save the image to the output directory

    Params:
    -------
        img: the image to save
        img_name: the name of the image file

    """
    cv2.imwrite(os.path.join(output_dir, img_name), img)

def find_best_match(patch, strip):
    """ Find the best x value for the patch in strip

    Params:
    -------
        patch: the patch to look for
        strip: the strip to look for the patch in

    Returns:
    --------
        best_x: the best x location that matches

    """

    best_x = 0
    w_h, w_w = window_size = patch.shape
    strip_w, strip_h = strip.shape
    shape = (strip_w - w_h+1, strip_h - w_w+1) + window_size
    strides = (strip.strides*2)
    y = as_strided(strip, shape=shape, strides=strides)
    ssd = np.sum(np.square(np.subtract(patch,y)),axis=(2,3))
    if ssd.size > 0:
        best_x = np.argmin(ssd)
    return best_x


def match_strips(L_strip, R_strip, b):
    """ match each two stripes together

    Params:
    -------
        L_strip: strip from the left image
        R_strip: strip from the right image

    Returns:
    --------
        disparity: the disparity for these two strips
    """
    steps = L_strip.shape[1]
    disparity = np.zeros(L_strip.shape)
    for x in range(steps-1):
        patch_left = L_strip[:, x:(x+b-1)]
        x_right = find_best_match(patch_left, R_strip)
        disparity[0, x + 1] = (x_right-x)
    return disparity


def disparity_ssd(L, R, ksize=13):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))

    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L
    ksize: the x and y shape of the kernel

    Returns: Disparity map, same size as L, R
    """
    L_h, L_w = L_size = L.shape
    R_h, R_w = R_size = R.shape

    if L_size != R_size:
        return -1

    b = ksize+1
    y = 0
    border = (b-2)/2

    padded_r = cv2.copyMakeBorder(R, border,border,border,border,cv2.BORDER_CONSTANT,value=0)
    padded_l = cv2.copyMakeBorder(L, border,border,border,border,cv2.BORDER_CONSTANT,value=0)
    disparity = np.zeros((L_h, L_w))
    while y <= L_h-1:
        _y = y+border
        L_strip = padded_l[_y:_y+b-1,:]
        R_strip = padded_r[_y:_y+b-1,:]
        matches = match_strips(L_strip, R_strip, b)
        distance_to_bottom = L_h-y
        if disparity[y:y+matches.shape[0],:].shape[0] < b-1:
            disparity[y:y+matches.shape[0],:] = matches[:distance_to_bottom,border:-border]
        else:
            disparity[y:y+matches.shape[0],:] = matches[:,border:-border]
        y += 1

    return disparity


def normalize(img):

    img -= img.min()
    img /= img.max()
    img *= 255
    img = np.uint8(img)
    return img



def disparity_ncorr(L, R):
    """Compute disparity map D(y, x) such that: L(y, x) = R(y, x + D(y, x))

    Params:
    L: Grayscale left image, in range [0.0, 1.0]
    R: Grayscale right image, same size as L

    Returns: Disparity map, same size as L, R
    """

    pass  # TODO: Your code here


def main():
    """Run code/call functions to solve problems."""


    # 1-a
    # Read images
    L = cv2.imread(os.path.join('input', 'pair0-L.png'), 0) * (1 / 255.0)  # grayscale, scale to [0.0, 1.0]
    R = cv2.imread(os.path.join('input', 'pair0-R.png'), 0) * (1 / 255.0)

    # Compute disparity (using method disparity_ssd defined in disparity_ssd.py)
    D_L = disparity_ssd(L, R)  # TODO: implemenet disparity_ssd()
    D_R = disparity_ssd(R, L)

    # TODO: Save output images (D_L as output/ps3-1-a-1.png and D_R as output/ps3-1-a-2.png)
    # Note: They may need to be scaled/shifted before saving to show results properly
    print D_L, D_L.max(), D_L.dtype
    D_L = normalize(D_L)
    print D_L, D_L.max(), D_L.dtype
    normalize(D_R)
    save_image(D_L, 'ps3-1-a-1.png')
    save_image(D_R, 'ps3-1-a-2.png')

    # 2
    # TODO: Apply disparity_ssd() to pair1-L.png and pair1-R.png (in both directions)

    L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1/255.0)
    R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1/255.0)

    D_L = disparity_ssd(L, R, 7)
    D_R = disparity_ssd(R, L, 7)

    np.clip(D_L, 0, 90)
    normalize(D_L)
    normalize(D_R)
    save_image(D_L, 'ps3-2-a-1.png')
    save_image(D_R, 'ps3-2-a-2.png')

    # 3
    # TODO: Apply disparity_ssd() to noisy versions of pair1 images
    # TODO: Boost contrast in one image and apply again

    # 4
    # TODO: Implement disparity_ncorr() and apply to pair1 images (original, noisy and contrast-boosted)

    # 5
    # TODO: Apply stereo matching to pair2 images, try pre-processing the images for best results


if __name__ == "__main__":
    main()
