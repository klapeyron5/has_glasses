import os
log_level = 3
log_level_values = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(log_level)
import tensorflow as tf
assert tf.__version__[0] == '2'
tf.get_logger().setLevel(log_level_values[log_level])
import numpy as np
from numpy.random import randint
import cv2
from numbers import Integral

dir_path = os.path.dirname(__file__)


class FDet:
    """
    Exported from tf1 from:
    https://github.com/TropComplique/FaceBoxes-tensorflow
    """
    def __init__(self):
        checkpoint_path = os.path.join(dir_path, 'fdet_saved_model')
        assert os.path.isdir(checkpoint_path), checkpoint_path + ' should exist'
        m = tf.saved_model.load(checkpoint_path)
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


def read_img_as_bytes_ndarray(img_path):
    """
    Returns img as base64 byte string and as np.ndarray
    :param img_path:
    :return: (img_bytes, img_ndarray)
    """
    img_bytes = open(img_path,'rb').read()
    img_ndarray = tf.io.decode_image(img_bytes).numpy()
    return img_bytes, img_ndarray


def is_any_int(x):
    """
    Checks if x is any integer type
    """
    return isinstance(x, Integral)


class Data_process_pipe:
    def __init__(self, funcs_names):
        possible_funcs = list(filter(lambda x: callable(self.__getattribute__(x)), dir(self)))

        self.funcs = []
        funcs_names_final = []
        for func_name in funcs_names:
            if func_name in possible_funcs:
                func = self.__getattribute__(func_name)
                self.funcs.append(func)
                funcs_names_final.append(func_name)
        self.get_config = lambda: funcs_names_final

    def __call__(self, **kwargs):
        for func in self.funcs:
            kwargs = func(**kwargs)
        return kwargs


class Pre(Data_process_pipe):

    READ_TF_IMG = 'read_tf_img'

    RESIZE_PROPORTIONAL_TF_BILINEAR = 'resize_proportional_tf_bilinear'
    PAD_TO_INPUT_CENTER = 'pad_to_input_center'
    STANDARDIZE = 'standardize'
    CROP_BB = 'crop_bb'
    FLIP = 'flip'
    ROTATE = 'rotate'

    def __init__(self, funcs_names, sample_size=None):
        self.init()
        super(Pre, self).__init__(funcs_names)
        if sample_size is not None:
            self._set_image_sample_size(sample_size)
        self.fdet = FDet()

    def init(self):
        pass

    def _set_image_sample_size(self, sample_size):
        assert is_any_int(sample_size)
        assert sample_size > 0
        self.sample_size = sample_size

    def read_tf_img(self, **kwargs):
        filepath = kwargs['filepath']
        _, img = read_img_as_bytes_ndarray(filepath)
        # assert len(img.shape) == 3, filepath
        kwargs['x'] = img
        return kwargs

    def resize_proportional_tf_bilinear(self, **kwargs):
        img = kwargs['x']
        shape = img.shape
        y, x = shape[0], shape[1]
        max_edge = max(x, y)
        scaling = self.sample_size / max_edge
        newx = int(round(scaling * x))
        newy = int(round(scaling * y))
        new_size = (newy, newx)
        try:
            img = tf.image.resize(img, new_size, method=tf.image.ResizeMethod.BILINEAR)
        except Exception:
            print()
        kwargs['x'] = img.numpy()
        return kwargs

    def pad_to_input_center(self, **kwargs):
        img = kwargs['x']
        y, x = img.shape[:2]
        assert y <= self.sample_size and x <= self.sample_size

        pad_y = self.sample_size-y
        residue_y = pad_y % 2
        pad_y //= 2

        pad_x = self.sample_size - x
        residue_x = pad_x % 2
        pad_x //= 2

        img = np.pad(img, [
            [pad_y, pad_y + residue_y],
            [pad_x, pad_x + residue_x],
            [0, 0]
        ], 'constant', constant_values=[0., 0.])
        kwargs['x'] = img
        return kwargs

    standardization_mean = 255 / 2
    def standardize(self, **kwargs):
        img = kwargs['x']
        img -= self.standardization_mean
        img /= 255
        kwargs['x'] = img
        return kwargs

    def crop_bb(self, **kwargs):
        img = kwargs['x']
        bb = kwargs.get('bb', None)
        if bb is None:
            l,t,r,b = self.fdet.get_bbox(img, thr=0.0)
            kwargs['bb'] = [l,t,r,b]
        else:
            l, t, r, b = bb
        if not(l==-1 or t>=b or l>=r):
            img = img[t:b,l:r]
        kwargs['x'] = img
        return kwargs

    def flip(self, **kwargs):
        img = kwargs['x']
        if randint(2):
            img = np.flip(img, 1)  # vertical flip
        kwargs['x'] = img
        return kwargs

    @staticmethod
    def get_interpolation_warpaffine():
        return randint(0,5)

    @staticmethod
    def augm_rotate(img, x, interpolation=1, border_value=(127, 127, 127)):
        image_center = tuple(np.array(img.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, x, 1.0)
        result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=interpolation, borderValue=border_value)  # INTER_LINEAR, CUBIC, AREA
        return result

    angle_low = -30
    angle_high = 31
    def rotate(self, **kwargs):
        img = kwargs['x']
        angle = randint(self.angle_low, self.angle_high)
        interp = self.get_interpolation_warpaffine()
        img = self.augm_rotate(img, angle, interp)
        kwargs['x'] = img
        return kwargs
