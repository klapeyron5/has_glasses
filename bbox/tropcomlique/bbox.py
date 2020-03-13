
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
        self.m = m.prune('import/image_tensor:0', ['import/boxes:0', 'import/scores:0', 'import/num_boxes:0'])

    def get_bbox(self, img, thr=0.4):
        sh = img.shape
        assert len(sh)==3
        img = img.reshape((1,)+sh)
        bboxes, scores, num_boxes = self.m(tf.convert_to_tensor(img))
        bboxes = bboxes.numpy()[0]
        scores = scores.numpy()[0]
        num_boxes = num_boxes.numpy()[0]
        bboxes = bboxes[:num_boxes]
        scores = scores[:num_boxes]
        if len(bboxes) > 0:
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


def ut_1():
    det = FDet()
    import os
    import shutil
    example_data_path = 'D:\has_glasses_data\SoF/data'
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
# ut_1()