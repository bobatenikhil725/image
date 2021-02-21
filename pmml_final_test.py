# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 18:15:50 2021

@author: bobate
"""

#importing libraries
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tf.config.experimental.set_memory_growth(physical_devices[0], True)

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
import cv2
import numpy as np
from sklearn.metrics import mean_squared_error 

# load an image
def load_image(filename, size=(512,512)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

# loading pretrained model 
model = load_model('model_010780.h5')
# loading  source image and resizing
src_image = load_image('D:/COLLEGE/1st Sem/ProbMeth(875)/PROJECT/Results/683_composite1.jpg')
resize_src_image=cv2.resize(src_image[0], (512,512))
print('Loaded', src_image.shape)
# generate image from source
gen_image = model.predict(src_image)
# scale from [-1,1] to [0,1]
gen_image = (gen_image + 1) / 2.0
gen_image1 = (gen_image*255).astype(np.uint8)
gen_image1 = cv2.cvtColor(gen_image1[0], cv2.COLOR_BGR2RGB)
# plot the image
pyplot.imshow(gen_image[0])
pyplot.axis('off')
pyplot.show()
psnr=cv2.PSNR(resize_src_image, gen_image[0])
print('psnr={}'.format (psnr))
MSE = np.square(np.subtract(resize_src_image, gen_image[0])).mean() 
print('mse={}'.format (MSE))
#Writing the output image
cv2.imwrite('D:/COLLEGE/1st Sem/ProbMeth(875)/PROJECT/{}{}'.format(2,('_out'+'.jpg')),gen_image1)

