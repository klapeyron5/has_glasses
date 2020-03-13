import os
import sys
sys.path.append('./')
from utils import Pre
log_level = 3
log_level_values = ['DEBUG', 'INFO', 'WARNING', 'ERROR']
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(log_level)
import tensorflow as tf
assert tf.__version__[0] == '2'
tf.get_logger().setLevel(log_level_values[log_level])


dirpath = os.path.dirname(__file__)
saved_model_dirname = 'saved_model'
checkpoint_path = os.path.join(dirpath, saved_model_dirname)
assert os.path.isdir(checkpoint_path), checkpoint_path + ' should exist'


def callback(filepath):
    filepath = os.path.realpath(filepath)
    print(filepath)


def print_files_with_glass(directory, callback=callback, device='/device:GPU:0'):
    with tf.device(device):
        assert os.path.isdir(directory), directory + ' should exist'
        assert callable(callback)
        model = tf.saved_model.load(checkpoint_path)
        thr = 0.5
        size = 120
        preproc = Pre([Pre.READ_TF_IMG, Pre.CROP_BB, Pre.RESIZE_PROPORTIONAL_TF_BILINEAR, Pre.STANDARDIZE, Pre.PAD_TO_INPUT_CENTER], size)
        for pardir, dirname, files in os.walk(directory):
            for file in files:
                try:
                    realpath = os.path.join(pardir, file)
                    assert os.path.isfile(realpath)
                    x = preproc(**{'filepath': realpath})['x']
                    x = x.reshape((1,)+x.shape)
                    score = model(x).numpy()[0]
                    if score[1] >= thr:
                        callback(realpath)
                except Exception:
                    pass


def time_test(device='/device:GPU:0'):
    with tf.device(device):
        import numpy as np
        size = 120
        m = tf.saved_model.load(checkpoint_path)
        from time import time

        n_avg = 100
        x = np.random.normal(size=(1, size, size, 3))
        t = time()
        for _ in (m(x) for _ in range(n_avg)): pass
        wasted_time = (time() - t) / n_avg
        print('wasted_time:', round(wasted_time, 3), 'sec/sample')


if __name__ == '__main__':
    args = sys.argv[1:]
    arg0 = args[0]
    if len(args) > 1:
        device = args[1]
    else:
        device = '/device:GPU:0'
    devices = tf.config.experimental.list_logical_devices()
    errmsg = 'cant see device: '+str(device)
    try:
        assert any([device in x.name for x in devices])
    except Exception:
        print(errmsg)
    if os.path.isdir(arg0):
        directory = arg0
        print_files_with_glass(directory, device=device)
    else:
        assert arg0 == 'get_time'
        time_test(device)
