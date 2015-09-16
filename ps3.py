"""Problem Set 3: Window-based Stereo Matching."""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2

import os

from IPython import embed

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


def rolling_window(stripe, w_size=3):
    """ Returns a stride array of the stripe

    Params:
    -------
        stripe: the strip to stride
        w_size: the size of the strides

    Return:
    -------
        y: the stride of the stripe
    """

    shape = (stripe.shape[0]+(w_size/2-1), w_size, w_size)
    strides = (stripe.itemsize, stripe.strides[0], stripe.strides[1])
    y = as_strided(stripe, shape=shape, strides=strides)
    return y


def find_best_match(patch, strip, patches):
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
    y = rolling_window(strip, patch.shape[1])
    print "scanlines are equal: ",np.array_equal(y,patches)
    best_x = np.argmin(np.sum(np.square(np.subtract(y,patch)),axis=(1,2)))
    return best_x


def match_strips(L_strip, R_strip, ksize=3):
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
    disparity = np.zeros((1,L_strip.shape[1]))
    print "stripes are equal: ", np.array_equal(L_strip, R_strip)
    patches = rolling_window(L_strip, ksize)
    for x in range(patches.shape[0]):
        patch_left = patches[x]
        x_right = find_best_match(patch_left, R_strip, patches)
        disparity[0, x] = (x_right-x)
    return disparity


def disparity_ssd(L, R, ksize=3):
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

    y = 0
    border = (ksize-1)/2

    padded_l = cv2.copyMakeBorder(L, border,border,border,border,cv2.BORDER_CONSTANT,value=0)
    padded_r = cv2.copyMakeBorder(R, border,border,border,border,cv2.BORDER_CONSTANT,value=0)
    disparity = np.zeros((L_h, L_w))
    for y in range(border, L_h):
        L_strip = padded_l[y-border:y+border+1,:]
        R_strip = padded_r[y-border:y+border+1,:]
        if np.array_equal(L_strip, R_strip) == False:
            matches = match_strips(L_strip, R_strip, ksize)
            disparity[y,:] = matches[:,border:-border]

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
    D_L = disparity_ssd(L, R, 7)  # TODO: implemenet disparity_ssd()
    D_R = disparity_ssd(R, L, 7)

    # TODO: Save output images (D_L as output/ps3-1-a-1.png and D_R as output/ps3-1-a-2.png)
    # Note: They may need to be scaled/shifted before saving to show results properly
    D_L = normalize(D_L)
    normalize(D_R)
    save_image(D_L, 'ps3-1-a-1.png')
    save_image(D_R, 'ps3-1-a-2.png')

    # 2
    # TODO: Apply disparity_ssd() to pair1-L.png and pair1-R.png (in both directions)

    #L = cv2.imread(os.path.join('input', 'pair1-L.png'), 0) * (1/255.0)
    #R = cv2.imread(os.path.join('input', 'pair1-R.png'), 0) * (1/255.0)

    #D_L = disparity_ssd(L, R, 7)
    #D_R = disparity_ssd(R, L, 7)

    #np.clip(D_L, -90, 0) * -1
    #normalize(D_L)
    #normalize(D_R)
    #save_image(D_L, 'ps3-2-a-1.png')
    #save_image(D_R, 'ps3-2-a-2.png')

    # 3
    # TODO: Apply disparity_ssd() to noisy versions of pair1 images
    # TODO: Boost contrast in one image and apply again

    # 4
    # TODO: Implement disparity_ncorr() and apply to pair1 images (original, noisy and contrast-boosted)

    # 5
    # TODO: Apply stereo matching to pair2 images, try pre-processing the images for best results


if __name__ == "__main__":
    main()
