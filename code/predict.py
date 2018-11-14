# import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np
# import pydicom

from utils import load_data, post_process

# from skimage import img_as_float
# from skimage.exposure import equalize_hist, rescale_intensity
# from skimage.measure import find_contours
# from skimage.morphology import remove_small_objects, remove_small_holes
# from skimage.transform import resize

from keras.models import load_model

# flags = tf.app.flags
# flags.DEFINE_string('model_path', '', 'Path to trained U-net')
# flags.DEFINE_string('dicom_path', '', 'Path to dicom path')
# flags.DEFINE_string('save_path', '', 'Path to stored pic')
# FLAGS = flags.FLAGS

def run_inference_for_single_image(model, img, img_original_shape):
    # Run inference for one image
    # model, U-net, obtained form keras load_model function
    # img, numpy array obtained from load_data function
    # img_original_shape, the shape of original dicom file
    mask = np.squeeze(model.predict(img))
    mask = post_process(mask, img_original_shape)

    return mask

def run_inference_for_multiple_images(model, images, image_original_shape):
    # Run inference for multiple images
    mask_list = []
    
    for idx, image in enumerate(images):
        mask = run_inference_for_single_image(model, image, image_original_shape[idx])
        mask_list.append(mask)
    
    return mask_list

# def main(_):
#     unet = load_model(FLAGS.model_path)
#     img_test, img_original = load_data(FLAGS.dicom_path)
#     mask = run_inference_for_single_image(unet, img_test)
#     mask = post_process(mask, img_original.shape)
#     contours = find_contours(mask, 0.5)

#     plt.imshow(img_original, cmap = 'gray')
#     for contour in contours:
#         plt.plot(contour[:, 1], contour[:, 0], 'r')
#     plt.savefig(FLAGS.save_path)

#     return None
    
# if __name__ == '__main__':
#     tf.app.run()
