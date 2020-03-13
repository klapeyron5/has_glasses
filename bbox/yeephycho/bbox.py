"""
https://github.com/yeephycho/tensorflow-face-detection
"""

import tensorflow as tf
import numpy as np
from klapeyron_py_utils.timings.timings import get_time
from klapeyron_py_utils.image.io_image import read_img_as_bytes_ndarray
import cv2
import os


dir_path = os.path.dirname(__file__)


class FDet:
    def __init__(self):
        m = tf.saved_model.load(os.path.join(dir_path, 'export'))
        self.m = m.prune('image_tensor:0', ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0'])

    def get_bbox(self, img, thr=0.4):
        sh = img.shape
        assert len(sh)==3
        img = img.reshape((1,)+sh)
        (bboxes, scores, classes) = self.m(tf.convert_to_tensor(img))
        bboxes = bboxes.numpy()[0]
        scores = scores.numpy()[0]
        classes = classes.numpy()[0]
        if len(bboxes) > 0:
            ids = np.where(classes==1)[0]
            scores = scores[ids]
            id = np.argmax(scores)
            score = scores[id]
            if score >= thr:
                bbox = bboxes[id]
                assert all([0 <= x <= 1 for x in bbox])
                h, w, _ = sh
                t, l, b, r = bbox
                t = int(round(h*t))
                l = int(round(w*l))
                b = int(round(h*b))
                r = int(round(w*r))
                return [l,t,r,b]
        return [-1,-1,-1,-1]


def ut_0():
    det = FDet()

    x = np.random.randint(0,256,size=(128, 128, 3), dtype=np.uint8)
    bb = det.get_bbox(x)

    x = np.zeros(shape=(128, 128, 3), dtype=np.uint8)
    bb = det.get_bbox(x)

    test_input_path = '../../data/example_data_glasses/with_glasses/0.jpg'
    _, img = read_img_as_bytes_ndarray(test_input_path)
    l,t,r,b = det.get_bbox(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.rectangle(img, (l, t), (r, b), [0, 255, 0], thickness=3)
    cv2.imwrite('./test_yeephycho.jpg', img)
    pass


def ut_1():
    det = FDet()
    import os
    import shutil
    example_data_path = 'D:\has_glasses_data\example_data_glasses/data'
    dist_vis_dir = './vis_example_data'
    if os.path.isdir(dist_vis_dir):
        shutil.rmtree(dist_vis_dir)
    os.mkdir(dist_vis_dir)
    assert os.path.isdir(dist_vis_dir)
    for pardir, dirs, files in os.walk(example_data_path):
        for file in files:
            real_file = os.path.join(pardir, file)
            dst_file = os.path.relpath(real_file, example_data_path)
            dst_file = dst_file.replace(os.path.sep, '_')
            # dst_file = dst_file.replace('.', '')
            dst_file = os.path.join(dist_vis_dir, dst_file)
            _, img = read_img_as_bytes_ndarray(real_file)
            l,t,r,b = det.get_bbox(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.rectangle(img, (l, t), (r, b), [0, 255, 0], thickness=3)
            cv2.imwrite(dst_file, img)
            assert os.path.isfile(dst_file)

ut_1()
