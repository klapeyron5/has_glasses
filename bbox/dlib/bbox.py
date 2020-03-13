import dlib

import dlib.cuda as cuda;
print(cuda.get_num_devices());
print(dlib.DLIB_USE_CUDA)
dlib.DLIB_USE_CUDA = True
print(dlib.DLIB_USE_CUDA)

import numpy as np
import cv2
from klapeyron_py_utils.timings.timings import get_time

cnn_face_detector = dlib.cnn_face_detection_model_v1('../../res/mmod_human_face_detector.dat')


def get_bb_dlib_cnn(imgs, batch_size=10):
    # assert len(imgs.shape) == 4
    dets = cnn_face_detector(imgs, 1, batch_size=batch_size)
    for det in dets:
        for bb in det:
            r = bb.rect
            # bboxes.append([r.left(), r.top(), r.right(), r.bottom()])
            # img = cv2.rectangle(img, (r.left(), r.top()), (r.right(), r.bottom()), [0, 255, 0], thickness=3)

test_input_path = 'D:\has_glasses_data\example_data_glasses/data/with_glasses/0.jpg'
test_input_path = 'D:\has_glasses_data\MeGlass\data/7134850@N05_identity_2@8276957582_3.jpg'

img = dlib.load_rgb_image(test_input_path)
# a = cnn_face_detector(img, 1)
# a = get_bb_dlib_cnn([img], 1)
imgs = [img]*1

print(get_time(get_bb_dlib_cnn, n_avg=10, **{'imgs': imgs, 'batch_size': 1}))

# get_bb_dlib_cnn(imgs)

# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('./test.jpg', img)
pass

# 38 sec on [img] bs=1