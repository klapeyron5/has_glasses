import dlib
import cv2
from klapeyron_py_utils.image.io_image import read_img_as_bytes_ndarray

# detector = dlib.get_frontal_face_detector()
#
#
# def get_bb_dlib_cnn(img):
#     # assert len(imgs.shape) == 4
#     dets = detector(img, 1)
#     for det in dets:
#         for bb in det:
#             r = bb.rect
#             # bboxes.append([r.left(), r.top(), r.right(), r.bottom()])
#             # img = cv2.rectangle(img, (r.left(), r.top()), (r.right(), r.bottom()), [0, 255, 0], thickness=3)
#
# test_input_path = 'D:\has_glasses_data\MeGlass\data/7134850@N05_identity_2@8276957582_3.jpg'
# img = dlib.load_rgb_image(test_input_path)
# dets = detector(img, 1)
# for bb in dets:
#     r = bb.rect


class FDet:
    def __init__(self):
        self.m = dlib.get_frontal_face_detector()

    def get_bbox(self, img):
        sh = img.shape
        assert len(sh)==3
        dets = self.m(img, 1)
        if len(dets) > 0:
            bb = dets[0]
            bb = [bb.left(), bb.top(), bb.right(), bb.bottom()]
            return bb
        else:
            return [-1,-1,-1,-1]


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