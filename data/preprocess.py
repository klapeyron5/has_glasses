from klapeyron_py_utils.data_pipe.data_process_pipe import Data_process_pipe
import numpy as np
from klapeyron_py_utils.image.io_image import read_img_as_bytes_ndarray
from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow()
from klapeyron_py_utils.types.common_types import is_any_int
# from bbox.yeephycho.bbox import FDet
from bbox.tropcomlique.bbox import FDet
import numpy as np
from numpy.random import randint
import cv2


class Pre(Data_process_pipe):

    READ_TF_IMG = 'read_tf_img'

    RESIZE_PROPORTIONAL_TF_BILINEAR = 'resize_proportional_tf_bilinear'
    RESIZE_COMMON = 'resize_common'
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

    def resize_common(self, **kwargs):
        if randint(2):
            kwargs = self.resize_proportion_cv2(**kwargs)
        else:
            kwargs = self.resize_proportional_tf_bilinear(**kwargs)
        return kwargs

    def resize_proportion_cv2(self, **kwargs):
        img = kwargs['x']
        shape = img.shape
        y, x = shape[0], shape[1]
        max_edge = max(x, y)
        scaling = self.sample_size / max_edge
        newx = int(round(scaling * x))
        newy = int(round(scaling * y))
        new_size = (newx, newy)
        img = img.astype(np.float32)
        img = cv2.resize(img, new_size, interpolation=self.get_interpolation_resize())
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
        else:
            img = np.zeros_like(img)  # TODO
        kwargs['x'] = img
        return kwargs

    def flip(self, **kwargs):
        img = kwargs['x']
        if randint(2):
            img = np.flip(img, 1)  # vertical flip
        kwargs['x'] = img
        return kwargs

    def get_interpolation_resize(self):
        return np.random.randint(0, 6)

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

# TODO
# rot90
# diff interps
# noise
# transparent backgr
# color augm