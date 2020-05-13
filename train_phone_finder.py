from boxTocsv import xml_to_csv
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import sys
import os
import numpy as np

def calc_dist(pair1, pair2):
    return ((float(pair1[0])-float(pair2[0]))**2 + (float(pair1[1])-float(pair2[1]))**2)**0.5

# get labeled image data and transfer them into csv
data_path = sys.argv[1]
xml_df = xml_to_csv(data_path)
xml_df.to_csv('phone_train.csv', index=None, header=False)

# the file of training model is in keras_retinanet/bin

model_path = os.path.join('./snapshots', 'resnet50_inference_csv_09.h5')
# uncomment this line below if we want to compare the result with pre-trained model
# model_path = os.path.join('./snapshots', 'resnet50_coco_best_v2.1.0.h5')
model = models.load_model(model_path, backbone_name='resnet50')

# get the labeled result
res_dict = {}
with open('./data/labels.txt', 'r+') as file:
    while True:
        line = file.readline().strip().split(' ')
        if line == ['']:
            break
        res_dict[line[0]] = (line[1], line[2])

# file counter
count = 0
correct = 0
for filename in os.listdir('./data'):
    if filename.endswith('.jpg'):
        # preprocess_image
        image = read_image_bgr(os.path.join('data', filename))
        y_scale, x_scale = image.shape[0], image.shape[1]

        image = preprocess_image(image)
        image, scale = resize_image(image)

        # predict the bounding box
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale

        # NMS switcher
        NMS = False
        x, y = None, None
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores too low will be supressed
            if score < 0.4 or NMS == True:
                continue
            NMS = True

            # our result
            x, y = (box[0] + box[2]) / (2 * x_scale), (box[1] + box[3]) / (2 * y_scale)

        if x != None and y != None and calc_dist((x,y),res_dict[filename]) <= 0.05:
            correct += 1
        count += 1
print('Train accuracy is: ', correct / count)