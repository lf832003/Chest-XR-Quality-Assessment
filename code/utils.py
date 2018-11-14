import pydicom

import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_float
from skimage.exposure import equalize_hist, rescale_intensity
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import disk, binary_opening, remove_small_holes, remove_small_objects
from skimage.transform import resize

def load_data(dicompath, target_shape = (256, 256)):
    # read dicom file and adjust the format to meet the requirment of prediction 
    ds = pydicom.read_file(dicompath)
    img = img_as_float(ds.pixel_array)
    img = resize(img, target_shape)
    img = equalize_hist(img)

    img -= img.mean()
    img /= img.std()
    img = rescale_intensity(img)
    img = np.expand_dims(img, axis = 0)
    img = np.expand_dims(img, axis = -1)

    return img, ds.pixel_array

def post_process(mask, image_shape, scale = 0.02):
    # post process the mask, remove small objects and holes, resize the mask to original shape of dicom input
    mask = mask > 0.5
    mask = remove_small_objects(mask, scale * np.prod(mask.shape))
    mask = remove_small_holes(mask, scale * np.prod(mask.shape))

    return resize(mask, image_shape, order = 3)

def calculate_spine_angle(psl, psr):
    # calculate the spine angle (gree line direction in the graph)
    if np.abs(psr[1] - psl[1]) < 1.0e-6:
        kpx = 0.0
        kpy = 1.0
    elif np.abs(psr[0] - psl[0]) < 1.0e-6:
        kpx = 1.0
        kpy = 0.0
    else:
        kpx = -(psr[1] - psl[1]) / (psr[0] - psl[0])        
        kpy = 1.0
        
    return np.arctan2(kpy, kpx) * 180 / np.pi, kpx, kpy

def im2bw(image):
    # gray scale image to binary image
    thresh = threshold_otsu(image)
    return (image > thresh).astype(np.uint8)

def find_centroid_of_body(image, sigma = 5, disk_rad = 10, morph_scale = 0.05):
    # find the weighted centroid of body (green cross in the graph)
    img_gaussian = gaussian(image, sigma = sigma)
    img_bw = im2bw(img_gaussian)
    selem = disk(disk_rad)
    img_morph = binary_opening(img_bw, selem)
    img_morph = remove_small_holes(img_morph.astype(np.bool), morph_scale * np.prod(image.shape))
    img_morph = remove_small_objects(img_morph, morph_scale * np.prod(image.shape)).astype(np.uint8)
    img_label = label(img_morph)
    props = regionprops(img_label, intensity_image = image)
    max_idx = np.argmax([prop.area for prop in props])
    yc, xc = props[max_idx].weighted_centroid
    
    return (xc, yc)

def find_centroid_of_lungs(mask, image = None):
    # find the weighted centroid of lungs based on lung segmentation (red star in the graph)
    mask_label = label(mask)
    props = regionprops(mask_label, intensity_image = image)
    yc1, xc1 = props[0].weighted_centroid
    yc2, xc2 = props[1].weighted_centroid
    if xc1 < xc2:
        return (xc1, yc1), (xc2, yc2)
    else:
        return (xc2, yc2), (xc1, yc1)
