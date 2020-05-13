<h2> Background </h2>

The project is targeted to do a phone detection task using Deep Neural Network. Given some images that contain a phone inside,
it will find its accurate location of the center of the phone. The model is based on RetinaNet to achieve both accuracy and efficiency.
The pre-trained model is trained on COCO dataset and available on https://github.com/fizyr/keras-retinanet/releases, but it only has 42.64% of accuracy on our data, then I used Transfer Learning to enhance the performance. If returned value has a value difference smaller than 0.01 from its correct label, we treat is as a correct prediction. The result on training set is 99.22%.

Some examples of my data are:

![demo-image1](/data/0.jpg) ![demo-image2](/data/1.jpg)
<h2> Setup </h2>
Install Numpy: pip install numpy --user
In this repository, execute pip install . --user
Alternatively, run python setup.py build_ext --inplace (Basic Requirement: Microsoft Visul Studio C++ 14.0.0)

<h2> Usage: </h2>
Training:
Create a snapshots folder
Download resnet50_coco_best_v2.1.0.h5 from https://github.com/fizyr/keras-retinanet/releases
python keras_retinanet/bin/train.py --weights snapshots/resnet50_coco_best_v2.1.0.h5 --workers 0 --freeze-backbone csv phone_train.csv class_map.csv

Prediction:
python find_phone.py 'Path-To-Image'
The result will be (x, y) where x and y are horizontal and vertical distance away from the origin (top-left corner).

Here are some sample results:

![test-image1](/data/10.jpg)

The result for the image above (0.48851349499760843, 0.4447669748879649)

![test-image2](/data/100.jpg)

The result for the image above (0.8161838453643177, 0.8618901726658359)
