import numpy as np

roi_ioc_x_start = 471
roi_ioc_y_start = 602

ptycho_roi_x_start = 54
ptycho_roi_y_start = 42

roi_x = 256
roi_y = 256

img = np.load('test_image.npy')

img_crop = img[roi_ioc_y_start+ptycho_roi_y_start:roi_ioc_y_start+ptycho_roi_y_start+roi_y,\
               roi_ioc_x_start+ptycho_roi_x_start:roi_ioc_x_start+ptycho_roi_x_start+roi_x]