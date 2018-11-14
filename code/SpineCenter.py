import os

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from IO import readfileslist, read_dicom_header, writexlsx
from predict import run_inference_for_single_image
from utils import calculate_spine_angle, load_data, find_centroid_of_body, find_centroid_of_lungs

from keras.models import load_model
from skimage.measure import find_contours

# The thresholds set to determine whether the spine is tilted or the body is offset from the center of image
TILT_ANGLE = 2 # 2 degree
OFFSET_X = 10 # 10 mm
OFFSET_Y = 10 # 10 mm

flags = tf.app.flags
flags.DEFINE_string('dicom_path', '', 'Path to dicom folder')
flags.DEFINE_string('model_path', '', 'Path to model (.hdf5)')
flags.DEFINE_string('save_pic_path', '', 'Path to folder which stores output images')
flags.DEFINE_string('save_xlsx_path', '', 'Path to the xlsx file')
FLAGS = flags.FLAGS

# def single_spine_center_detect():
def draw_image(image, contours, ps, psl, psr, kpx, kpy, angle, save_path = [], length = 200, my_dpi = 96):
    # draw image
    plt.figure(figsize = (image.shape[1] / my_dpi, image.shape[0] / my_dpi), dpi = my_dpi)
    plt.imshow(image, cmap = 'gray')
    plt.plot(psl[0], psl[1], 'r*', markersize = 16) # centroid of right lung    
    plt.plot(psr[0], psr[1], 'r*', markersize = 16) # centroid of left lung
    plt.plot([psl[0], psr[0]], [psl[1], psr[1]], 'r', linewidth = 2.5)
    
    plt.plot(ps[0], ps[1], 'gx', markersize = 16) # centroid of body

    x = [(psl[0] + psr[0]) / 2.0 + kpx / np.sqrt(kpx ** 2 + kpy ** 2) * l for l in np.arange(-length / 2.0, length / 2.0)]
    y = [(psl[1] + psr[1]) / 2.0 + kpy / np.sqrt(kpx ** 2 + kpy ** 2) * l for l in np.arange(-length / 2.0, length / 2.0)]    
    plt.plot(x, y, 'g', linewidth = 2.5)
    
    for contour in contours:
        plt.plot(contour[:, 1], contour[:, 0], 'b', linewidth = 1.5)
        
    plt.plot(image.shape[1] / 2.0, image.shape[0] / 2.0, 'mo', markersize = 16)
    plt.plot([0, image.shape[1] - 1], [image.shape[0] / 2.0, image.shape[0] / 2.0], 'm--', linewidth = 1.5) # center line of image
    plt.plot([image.shape[1] / 2.0, image.shape[1] / 2.0], [0, image.shape[0] - 1], 'm--', linewidth = 1.5)
    plt.title(os.path.basename(save_path)[0:-4]) # depend on the filename of dicom file
    # plt.title('Spine tilt angle {:.2f} (degree)'.format(90 - angle))
    
    plt.axis([0, image.shape[1] - 1,image.shape[0] - 1, 0])
    plt.savefig(save_path)        
        
    return None

def is_spine_tilt(angle, offset_x, offset_y):
    # Need to be changed based on the real clinical standard
    if angle > TILT_ANGLE:
        return 'Yes'
    if offset_x > OFFSET_X or offset_y > OFFSET_Y:
        return 'Yes'
    else:
        return 'No'

def multi_spine_center_detect(inputfolder, savefolder, xlsxpath):
    # main function
    # inputfolder, path to the folder which contains the dicom file, may have multiple dicom files
    # savefolder, path to the folder whcih stores the output image with several lines and markers
    # xlsxpath, path to the xlsx path which stores the infromation shown below:
    # File_path | Offset_col (mm) | Offset_row (mm) | Tilt_angle (degree) | Is_offset | Pic_path
    dcmlist = readfileslist(inputfolder, '.dcm') # read dicom file path from the input folder   
    unet = load_model('../model/trained_model.hdf5') # load model, change the path when necessary, FLAGS.model_path
    row_spacing, col_spacing = read_dicom_header(dcmlist[0]) # get imager pixel spacing
    
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
        
    tilt_angle = []
    offset_x = []
    offset_y = []
    results = []
    picpath = []
    
    for dcm in dcmlist:        
        img_test, img_original = load_data(dcm) # load dicom file and transfer to required format for prediction
        
        mask = run_inference_for_single_image(unet, img_test, img_original.shape) # make prediction           
        
        ps = find_centroid_of_body(img_original) # find centroid of body (green cross)
        psl, psr = find_centroid_of_lungs(mask, img_original) # find centroid of luns (red stars)
        
        angle, kpx, kpy = calculate_spine_angle(psl, psr) # calculate the spine direction (green line)
        contours = find_contours(mask, 0.5) # find contours of lungs
        
        tilt_angle.append(90 - angle) # calculate the tilt angle
        offset_x.append(col_spacing * (ps[0] - img_original.shape[1] / 2.0)) # calculate the col offset (mm)
        offset_y.append(row_spacing * (ps[1] - img_original.shape[0] / 2.0)) # calculate the row offset (mm)
        results.append(is_spine_tilt(tilt_angle[-1], offset_x[-1], offset_y[-1])) # determine whether the spine is tilt of offset from the center
        
        if os.path.isabs(savefolder):
            picpath.append(os.path.join(savefolder, os.path.basename(dcm)[0:-4] + '_' + str(dcmlist.index(dcm)) + '.png'))
        else:
            picpath.append(os.path.join(os.getcwd(), savefolder, os.path.basename(dcm)[0:-4] + '_' + str(dcmlist.index(dcm)) + '.png'))
        # picpathlist.append(picpath) # only work for xlsx files       
        draw_image(img_original, contours, ps, psl, psr, kpx, kpy, angle, picpath[-1]) # draw image with above information
        
    df = pd.DataFrame({'File_path': dcmlist, 'Tilt_angle': tilt_angle, 'Offset_col': offset_x, 'Offset_row': offset_y, 'Is_offset': results, 'Pic_path': picpath}) 
    writexlsx(df, xlsxpath) # save the pandas dataframe to xlsx file
       
    return None       

def main(_):
    multi_spine_center_detect(FLAGS.dicom_path,
                              FLAGS.save_pic_path, 
                              FLAGS.save_xlsx_path)

if __name__ == '__main__':
    tf.app.run() 
