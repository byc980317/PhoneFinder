import keras
import sys
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import tensorflow as tf
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your trained model
model_path = os.path.join('.', 'snapshots', 'resnet50_inference_csv_09.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
#model = models.convert_model(model)

#print(model.summary())

# load label to names mapping for visualization purposes

# load image
image = read_image_bgr(sys.argv[1])
y_scale, x_scale = image.shape[0], image.shape[1]

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

# correct for image scale
boxes /= scale

# visualize detections
NMS = False
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.4 or NMS == True:
        continue
    NMS = True
    print((box[0]+box[2])/(2*x_scale), (box[1]+box[3])/(2*y_scale))
    color = label_color(label)