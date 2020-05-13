Project Description:
The project is targeted to do a phone detection task using Deep Neural Network. Given some images that contain a phone inside,
I need to find its accurate location of the center of the phone. The model is based on RetinaNet to achieve both accuracy and efficiency.
The pre-trained model is trained on COCO dataset and available on github, but it only has 42.64% of accuracy on our data,
then I used Transfer Learning to enhance the performance.

My approach is using a labelling software to label the bounding boxes of the phone manually and transfer them into
csv file for training. This part of code is in boxTocsv.py. Then, I use resnet50_coco_best_v2.1.0.h5 in snapshots folder
as the pre-trained model and only train the weight of last several layers. Due to the limitation of my computer, I just
train 10 epochs, where each of them has 10 steps. Since this task mostly deals with regression loss, I will select the
model with lowest regression loss. The result on training set is 99.22%.

Setup instruction:
1. Install Numpy: pip install numpy --user
2. In this repository, execute pip install . --user
Alternatively, run python setup.py build_ext --inplace (Basic Requirement: Microsoft Visul Studio C++ 14.0.0)

Usage:
Prediction:
python find_phone.py ./data/image.jpg
The result will be (x, y) where x and y are horizontal and vertical distance away from the origin (top-left corner).

Training:
python keras_retinanet/bin/train.py --weights snapshots/resnet50_coco_best_v2.1.0.h5 --workers 0 --freeze-backbone csv phone_train.csv class_map.csv

Future Improvement:
1. Preparing more labeled images for training
2. Try other models such as RCNN