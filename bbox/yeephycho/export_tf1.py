import tensorflow as tf
assert tf.__version__[0] == '1'

from klapeyron_py_utils.tensorflow.export_tf1 import export_tf1, export_tf1_exp

PATH_TO_CKPT = '../../data/frozen_inference_graph_face.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=detection_graph, config=config)
# export_tf1(sess, 'image_tensor:0', 'detection_boxes:0')
export_tf1_exp(sess, 'image_tensor:0', ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0', 'num_detections:0'])
